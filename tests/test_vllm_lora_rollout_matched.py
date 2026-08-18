from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import rl_no_backward.vllm_lora_rollout as rollout_module
from rl_no_backward.vllm_lora_rollout import (
    ReloadableLoRAGenerator,
    create_standard_lora_vllm_engine,
)


class _FakeLLMConstructor:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs
        self.vllm_config = SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=2)
        )


def test_engine_constructor_pins_revision_fa2_and_fast_inprocess(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = _FakeLLMConstructor
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    engine = create_standard_lora_vllm_engine(
        "model",
        revision="commit",
        dtype="bfloat16",
        max_model_len=768,
        max_lora_rank=8,
        kv_cache_memory_bytes=1024,
        enforce_eager=False,
        flash_attn_version=2,
        max_num_seqs=64,
        seed=0,
    )
    assert isinstance(engine, _FakeLLMConstructor)
    assert _FakeLLMConstructor.kwargs["revision"] == "commit"
    assert _FakeLLMConstructor.kwargs["attention_config"] == {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 2,
    }
    assert _FakeLLMConstructor.kwargs["enable_lora"] is True
    assert rollout_module.os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert rollout_module.os.environ["VLLM_BATCH_INVARIANT"] == "0"


class _FakePeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.ones(1), requires_grad=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})

    def save_pretrained(self, destination, *, safe_serialization, selected_adapters):
        assert safe_serialization and selected_adapters == ["default"]
        destination.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
        destination.joinpath("adapter_model.safetensors").write_bytes(b"weights")


class _FakeEngine:
    def __init__(self) -> None:
        self.requests = []

    def add_lora(self, request):
        self.requests.append(request)
        return True

    def list_loras(self):
        return {1}


class _FakeLLM:
    def __init__(self) -> None:
        self.llm_engine = _FakeEngine()
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1
        return True


def test_same_id_reload_receipts_and_export_gc(monkeypatch, tmp_path) -> None:
    request_module = ModuleType("vllm.lora.request")

    class FakeRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    request_module.LoRARequest = FakeRequest
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)
    monkeypatch.setattr(
        rollout_module,
        "make_vllm_lora_request",
        lambda adapter_path, **kwargs: FakeRequest(
            lora_path=str(adapter_path), load_inplace=True, **kwargs
        ),
    )
    llm = _FakeLLM()
    generator = ReloadableLoRAGenerator(llm, retain_exports=2)
    model = _FakePeftModel()
    receipts = []
    for version in (1, 2, 3):
        with torch.no_grad():
            model.lora_B["default"].weight.add_(0.01)
        receipts.append(
            generator.sync(
                model,
                tmp_path,
                version=version,
                policy_version=f"v{version}",
            )
        )
    assert llm.resets == 3
    assert all(request.load_inplace for request in llm.llm_engine.requests)
    assert len({receipt.state_digest for receipt in receipts}) == 3
    assert len(list(tmp_path.glob("adapter-*"))) == 2
    assert generator.request.load_inplace is False
    assert all(receipt.adapter_path_transient for receipt in receipts)
    assert all(
        set(receipt.durable_hash_fields) == {"state_digest", "adapter_model_sha256"}
        for receipt in receipts
    )


def test_next_token_probe_records_policy_digest_and_selected_logprob(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class ProbeLLM:
        def generate(self, prompts, **kwargs):
            assert kwargs["sampling_params"].kwargs["max_tokens"] == 1
            candidate = SimpleNamespace(
                token_ids=[17],
                logprobs=[{17: SimpleNamespace(logprob=-0.125)}],
            )
            return [
                SimpleNamespace(
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    outputs=[candidate],
                )
            ]

    generator = ReloadableLoRAGenerator(ProbeLLM())
    generator.request = object()
    generator.policy_version = "policy-v4"
    generator.state_digest = "a" * 64
    probe = generator.probe_next_token((1, 2, 3))
    assert probe.policy_version == "policy-v4"
    assert probe.state_digest == "a" * 64
    assert probe.token_id == 17
    assert probe.selected_token_logprob == pytest.approx(-0.125)
