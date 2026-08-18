from __future__ import annotations

import os
import pickle
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from rl_no_backward.model import (
    AdapterWrappedLayer,
    ModelBundle,
    ResidualCoreAdapter,
)
from rl_no_backward.vllm_rollout import (
    VLLM_RESIDUAL_ARCHITECTURE,
    OnPolicyVLLMGenerator,
    ResidualAdapterCoreUpdate,
    ResidualAdapterLayerState,
    ResidualAdapterSnapshot,
    VLLMGroupedGeneration,
    VLLMResidualCoreAdapter,
    apply_adapter_to_vllm_split_state,
    apply_vllm_adapter_snapshot,
    apply_vllm_core_update,
    capture_residual_adapter_snapshot,
    create_vllm_engine,
    generate_vllm_greedy,
    generate_vllm_grouped,
    parse_vllm_greedy_outputs,
    parse_vllm_grouped_outputs,
    probe_vllm_repeat_determinism,
    sync_vllm_adapter_snapshot,
    vllm_hf_overrides,
)


def _layer_state(layer_index: int, core_offset: float = 0.0) -> ResidualAdapterLayerState:
    basis = torch.tensor(
        [
            [0.5, -0.2],
            [0.3, 0.4],
            [-0.1, 0.6],
            [0.7, 0.1],
        ]
    )
    core = torch.tensor([[0.2, -0.1], [0.05, 0.3]]) + core_offset
    return ResidualAdapterLayerState(
        layer_index=layer_index,
        p_basis=basis,
        q_basis=basis.flip(0),
        core=core,
        scale=0.4,
    )


def _snapshot(core_offset: float = 0.0) -> ResidualAdapterSnapshot:
    return ResidualAdapterSnapshot(
        (_layer_state(2, core_offset), _layer_state(3, core_offset + 0.1))
    )


class _FakeVLLMModel(nn.Module):
    def __init__(self, *, scale: float = 0.4) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                VLLMResidualCoreAdapter(4, 2, scale=scale, layer_index=2),
                VLLMResidualCoreAdapter(4, 2, scale=scale, layer_index=3),
            ]
        )


class _FakeLLM:
    def __init__(self, worker_count: int = 2, *, cache_reset: bool = True) -> None:
        self.workers = [_FakeVLLMModel() for _ in range(worker_count)]
        self.cache_reset = cache_reset
        self.calls: list[object] = []

    def sleep(self, *, level: int, mode: str) -> None:
        self.calls.append(("sleep", level, mode))

    def apply_model(self, function):
        self.calls.append("apply_model")
        return [function(worker) for worker in self.workers]

    def reset_prefix_cache(self) -> bool:
        self.calls.append("reset_prefix_cache")
        return self.cache_reset

    def wake_up(self, *, tags: list[str]) -> None:
        self.calls.append(("wake_up", tags))


def test_split_state_mapping_matches_post_block_adapter() -> None:
    torch.manual_seed(4)
    adapter = VLLMResidualCoreAdapter(4, 2, scale=0.4, layer_index=3)
    source = _layer_state(3)
    with torch.no_grad():
        adapter.p_basis.copy_(source.p_basis)
        adapter.q_basis.copy_(source.q_basis)
        adapter.core.copy_(source.core)

    branch = torch.randn(2, 3, 4)
    residual = torch.randn(2, 3, 4)
    mapped_branch, mapped_residual = apply_adapter_to_vllm_split_state(branch, residual, adapter)

    expected = adapter(branch + residual)
    torch.testing.assert_close(mapped_branch + mapped_residual, expected, atol=2e-6, rtol=2e-6)
    assert mapped_residual is residual


def test_vllm_adapter_state_is_buffer_only_for_strict_checkpoint_loading() -> None:
    adapter = VLLMResidualCoreAdapter(4, 2, scale=0.4, layer_index=3)

    assert dict(adapter.named_parameters()) == {}
    assert set(dict(adapter.named_buffers())) == {"p_basis", "q_basis", "core"}


def test_snapshot_is_sorted_canonical_and_detached() -> None:
    original = torch.ones(4, 2, dtype=torch.float64)
    first = ResidualAdapterLayerState(5, original, original, torch.eye(2), 1.0)
    second = _layer_state(2)
    snapshot = ResidualAdapterSnapshot((first, second))
    original.zero_()

    assert snapshot.layer_indices == (2, 5)
    assert snapshot.layers[1].p_basis.dtype == torch.float32
    assert torch.equal(snapshot.layers[1].p_basis, torch.ones(4, 2))
    assert snapshot.digest(include_bases=True) == snapshot.digest(include_bases=True)
    assert snapshot.digest(include_bases=False) != snapshot.digest(include_bases=True)


def test_core_update_digest_matches_snapshot_without_serializing_bases() -> None:
    snapshot = _snapshot()
    update = snapshot.core_update()

    assert isinstance(update, ResidualAdapterCoreUpdate)
    assert update.digest() == snapshot.digest(include_bases=False)
    assert len(pickle.dumps(update)) < len(pickle.dumps(snapshot)) / 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("p_basis", torch.tensor([[float("nan")], [0.0]])),
        ("core", torch.tensor([[float("inf")]])),
    ],
)
def test_layer_state_rejects_nonfinite_tensors(field: str, value: torch.Tensor) -> None:
    values = {
        "layer_index": 0,
        "p_basis": torch.ones(2, 1),
        "q_basis": torch.ones(2, 1),
        "core": torch.ones(1, 1),
        "scale": 1.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        ResidualAdapterLayerState(**values)


def test_capture_snapshot_uses_hf_decoder_layer_indices() -> None:
    torch.manual_seed(8)
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
    adapter_names: list[str] = []
    for index in (2, 3):
        basis = torch.linalg.qr(torch.randn(4, 2), mode="reduced").Q
        adapter = ResidualCoreAdapter(basis, basis, scale=0.4)
        adapter.core.data.fill_(index / 10)
        decoder.layers[index] = AdapterWrappedLayer(nn.Identity(), adapter)
        adapter_names.append(f"model.layers.{index}.adapter.core")
    model = nn.Module()
    model.model = decoder
    bundle = ModelBundle(
        model=model,
        tokenizer=None,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=adapter_names,
        device=torch.device("cpu"),
        model_name="fake-qwen",
    )

    snapshot = capture_residual_adapter_snapshot(bundle)

    assert snapshot.layer_indices == (2, 3)
    assert snapshot.rank == 2
    assert torch.equal(snapshot.layers[0].core, torch.full((2, 2), 0.2))


def test_apply_snapshot_copies_bases_then_allows_core_only_updates() -> None:
    model = _FakeVLLMModel()
    initial = _snapshot()
    receipt = apply_vllm_adapter_snapshot(
        model, snapshot=initial, version="initial", include_bases=True
    )
    bases_before = [adapter.p_basis.clone() for adapter in model.adapters]

    updated = _snapshot(core_offset=0.7)
    second_receipt = apply_vllm_core_update(model, update=updated.core_update(), version="step-1")

    assert receipt["digest"] == initial.digest(include_bases=True)
    assert second_receipt["digest"] == updated.digest(include_bases=False)
    for adapter, source, prior_basis in zip(
        model.adapters, updated.layers, bases_before, strict=True
    ):
        assert torch.equal(adapter.p_basis, prior_basis)
        assert torch.equal(adapter.core, source.core)


def test_apply_snapshot_validates_every_layer_before_mutating() -> None:
    model = _FakeVLLMModel()
    before = [adapter.core.clone() for adapter in model.adapters]
    model.adapters[1].scale = 9.0

    with pytest.raises(ValueError, match="scale mismatch"):
        apply_vllm_adapter_snapshot(
            model,
            snapshot=_snapshot(core_offset=0.8),
            version="bad",
            include_bases=True,
        )

    for adapter, original in zip(model.adapters, before, strict=True):
        assert torch.equal(adapter.core, original)


def test_sync_orders_pause_copy_cache_reset_and_resume() -> None:
    llm = _FakeLLM()
    snapshot = _snapshot()

    receipts = sync_vllm_adapter_snapshot(
        llm,
        snapshot,
        version="method=bp_grpo/seed=0/step=1",
        include_bases=True,
        pause_scheduler=True,
    )

    assert len(receipts) == 2
    assert llm.calls == [
        ("sleep", 0, "wait"),
        "apply_model",
        "reset_prefix_cache",
        ("wake_up", ["scheduling"]),
    ]


def test_sync_raises_when_cache_reset_fails() -> None:
    llm = _FakeLLM(cache_reset=False)

    with pytest.raises(RuntimeError, match="prefix cache"):
        sync_vllm_adapter_snapshot(
            llm,
            _snapshot(),
            version="step-2",
            include_bases=False,
        )

    assert llm.calls == ["apply_model", "reset_prefix_cache"]


def test_vllm_hf_overrides_select_custom_architecture() -> None:
    overrides = vllm_hf_overrides(_snapshot())

    assert overrides == {
        "architectures": [VLLM_RESIDUAL_ARCHITECTURE],
        "residual_core_adapter_layers": [2, 3],
        "residual_core_adapter_rank": 2,
        "residual_core_adapter_scale": 0.4,
    }


def _candidate(
    index: int,
    tokens: list[int],
    log_probs: list[float],
    reason: str,
    *,
    stop_reason: int | str | None = None,
):
    return SimpleNamespace(
        index=index,
        token_ids=tokens,
        logprobs=[
            {token_id: SimpleNamespace(logprob=log_prob)}
            for token_id, log_prob in zip(tokens, log_probs, strict=True)
        ],
        finish_reason=reason,
        stop_reason=stop_reason,
    )


def test_parse_grouped_outputs_sorts_candidates_and_preserves_terminal_eos() -> None:
    prompts = ((1, 2), (3,))
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=[
                _candidate(1, [8, 7], [-0.8, -0.7], "stop", stop_reason=7),
                _candidate(0, [4, 7], [-0.4, -0.7], "stop"),
            ],
        ),
        SimpleNamespace(
            prompt_token_ids=[3],
            outputs=[
                _candidate(0, [5], [-0.5], "length"),
                _candidate(1, [6, 7], [-0.6, -0.7], "stop", stop_reason=7),
            ],
        ),
    ]

    result = parse_vllm_grouped_outputs(
        outputs,
        prompts,
        group_size=2,
        pad_token_id=0,
        eos_token_ids=(7,),
        policy_version="step-3",
    )

    assert result.policy_version == "step-3"
    assert result.valid_response_tokens == 7
    assert result.response_input_ids.tolist() == [
        [[4, 7], [8, 7]],
        [[5, 0], [6, 7]],
    ]
    assert result.response_mask.tolist() == [
        [[True, True], [True, True]],
        [[True, False], [True, True]],
    ]
    torch.testing.assert_close(
        result.old_token_log_probs,
        torch.tensor([[[-0.4, -0.7], [-0.8, -0.7]], [[-0.5, 0.0], [-0.6, -0.7]]]),
    )


@pytest.mark.parametrize(
    ("tokens", "reason", "stop_reason", "message"),
    [
        ([4], "stop", 7, "omitted the terminal EOS"),
        ([4, 7, 9], "stop", 7, "tokens after EOS"),
        ([4, 7], "length", None, "labeled an EOS-terminated"),
        ([4, 7], "stop", 8, "unconfigured token ID"),
    ],
)
def test_parse_grouped_outputs_hard_gates_eos_objective_parity(
    tokens: list[int],
    reason: str,
    stop_reason: int | None,
    message: str,
) -> None:
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1],
            outputs=[
                _candidate(0, tokens, [-0.2] * len(tokens), reason, stop_reason=stop_reason),
                _candidate(1, [5, 7], [-0.3, -0.4], "stop", stop_reason=7),
            ],
        )
    ]

    with pytest.raises(RuntimeError, match=message):
        parse_vllm_grouped_outputs(
            outputs,
            ((1,),),
            group_size=2,
            pad_token_id=0,
            eos_token_ids=(7,),
            policy_version="step-3",
        )


def test_on_policy_generator_requires_full_initial_sync() -> None:
    generator = OnPolicyVLLMGenerator(_FakeLLM(worker_count=1))

    with pytest.raises(RuntimeError, match="first.*include PCA bases"):
        generator.sync(_snapshot(), version="initial")
    receipts = generator.sync(_snapshot(), version="initial", include_bases=True)

    assert len(receipts) == 1
    assert generator.policy_version == "initial"
    assert generator.state_digest == receipts[0]["digest"]


def test_on_policy_generator_uses_full_state_identity_for_core_updates() -> None:
    generator = OnPolicyVLLMGenerator(_FakeLLM(worker_count=1))
    initial = _snapshot()
    updated = _snapshot(core_offset=0.25)
    generator.sync(initial, version="initial", include_bases=True)

    receipts = generator.sync(updated, version="step-1")

    assert receipts[0]["digest"] == updated.core_update().digest()
    assert generator.state_digest == updated.digest(include_bases=True)
    assert generator.state_digest != receipts[0]["digest"]


def test_on_policy_generator_rejects_changed_bases_in_core_only_update() -> None:
    llm = _FakeLLM(worker_count=1)
    generator = OnPolicyVLLMGenerator(llm)
    initial = _snapshot()
    generator.sync(initial, version="initial", include_bases=True)
    changed_layer = ResidualAdapterLayerState(
        layer_index=initial.layers[0].layer_index,
        p_basis=initial.layers[0].p_basis + 0.1,
        q_basis=initial.layers[0].q_basis,
        core=initial.layers[0].core,
        scale=initial.layers[0].scale,
    )
    changed = ResidualAdapterSnapshot((changed_layer, initial.layers[1]))
    calls_before = list(llm.calls)

    with pytest.raises(RuntimeError, match="fixed bases/layout changed"):
        generator.sync(changed, version="unsafe-step")

    assert llm.calls == calls_before


def test_parse_greedy_outputs_preserves_eos_and_prompt_order() -> None:
    prompts = ((1, 2), (3,))
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[4, 7],
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            prompt_token_ids=[3],
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[5, 6],
                    finish_reason="length",
                    stop_reason=None,
                )
            ],
        ),
    ]

    result = parse_vllm_greedy_outputs(
        outputs, prompts, eos_token_ids=(7,), policy_version="eval-step-2"
    )

    assert result.response_token_ids == ((4, 7), (5, 6))
    assert result.finish_reasons == ("stop", "length")
    assert result.policy_version == "eval-step-2"


def test_generation_uses_only_explicit_eos_and_retains_terminal_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_params: list[object] = []

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            sampling_params.append(self)

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class FakeLLM:
        model_config = SimpleNamespace(logprobs_mode="processed_logprobs")

        @staticmethod
        def generate(prompts, *, sampling_params, use_tqdm):
            assert use_tqdm is False
            if isinstance(sampling_params, list):
                return [
                    SimpleNamespace(
                        prompt_token_ids=prompts[0]["prompt_token_ids"],
                        outputs=[
                            _candidate(0, [4, 7], [-0.2, -0.3], "stop", stop_reason=7),
                            _candidate(1, [5, 7], [-0.4, -0.5], "stop", stop_reason=7),
                        ],
                    )
                ]
            return [
                SimpleNamespace(
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=[6, 7],
                            finish_reason="stop",
                            stop_reason=7,
                        )
                    ],
                )
            ]

    sampled = generate_vllm_grouped(
        FakeLLM(),
        ((1, 2),),
        group_size=2,
        max_new_tokens=4,
        temperature=0.8,
        seed=9,
        pad_token_id=0,
        eos_token_ids=(7,),
        policy_version="step-1",
    )
    greedy = generate_vllm_greedy(
        FakeLLM(),
        ((1, 2),),
        max_new_tokens=4,
        eos_token_ids=(7,),
        policy_version="eval-step-1",
    )

    assert sampled.response_input_ids.tolist() == [[[4, 7], [5, 7]]]
    assert greedy.response_token_ids == ((6, 7),)
    assert len(sampling_params) == 2
    for params in sampling_params:
        assert params.stop_token_ids == [7]
        assert params.ignore_eos is True
        assert params.detokenize is False
        assert params.skip_special_tokens is False


def test_repeat_determinism_probe_compares_same_policy_seed_outputs() -> None:
    class FakePolicy:
        policy_version = "method=bp_grpo/seed=0/reset"
        state_digest = "same-policy-digest"

        def generate(self, prompt_token_ids, **kwargs):
            assert kwargs["seed"] == 20_001
            return VLLMGroupedGeneration(
                prompt_token_ids=prompt_token_ids,
                response_input_ids=torch.tensor([[[4, 7], [5, 7]]]),
                response_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                old_token_log_probs=torch.tensor([[[-0.2, -0.3], [-0.4, -0.5]]]),
                finish_reasons=(("stop", "stop"),),
                policy_version=self.policy_version,
            )

    report = probe_vllm_repeat_determinism(
        FakePolicy(),  # type: ignore[arg-type]
        ((1, 2),),
        group_size=2,
        max_new_tokens=4,
        temperature=0.8,
        seed=20_001,
        pad_token_id=0,
        eos_token_ids=(7,),
    )

    assert report.token_sequences_equal
    assert report.behavior_logprobs_bitwise_equal
    assert report.maximum_behavior_logprob_abs_delta == 0.0
    assert report.first_valid_response_tokens == report.second_valid_response_tokens == 4
    assert report.fully_repeatable


def test_repeat_determinism_probe_exposes_token_count_and_logprob_drift() -> None:
    class DriftingPolicy:
        policy_version = "method=fo_npg/seed=0/reset"
        state_digest = "same-policy-digest"
        calls = 0

        def generate(self, prompt_token_ids, **_kwargs):
            self.calls += 1
            second = self.calls == 2
            mask = torch.tensor([[[True, True], [True, not second]]])
            return VLLMGroupedGeneration(
                prompt_token_ids=prompt_token_ids,
                response_input_ids=torch.tensor([[[4, 7], [5, 7 if not second else 0]]]),
                response_mask=mask,
                old_token_log_probs=torch.tensor(
                    [[[-0.2, -0.3], [-0.4, -0.5 if not second else 0.0]]]
                ),
                finish_reasons=(("stop", "length" if second else "stop"),),
                policy_version=self.policy_version,
            )

    report = probe_vllm_repeat_determinism(
        DriftingPolicy(),  # type: ignore[arg-type]
        ((1, 2),),
        group_size=2,
        max_new_tokens=4,
        temperature=0.8,
        seed=20_001,
        pad_token_id=0,
        eos_token_ids=(7,),
    )

    assert not report.token_sequences_equal
    assert not report.behavior_logprobs_bitwise_equal
    assert report.maximum_behavior_logprob_abs_delta == pytest.approx(0.5)
    assert (report.first_valid_response_tokens, report.second_valid_response_tokens) == (4, 3)
    assert not report.fully_repeatable


def test_create_engine_pins_correctness_critical_options(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))

    class _Registry:
        architectures: ClassVar[set[str]] = set()

        @classmethod
        def get_supported_archs(cls) -> set[str]:
            assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
            assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
            assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
            return cls.architectures

        @staticmethod
        def register_model(architecture: str, qualname: str) -> None:
            captured["registration"] = (architecture, qualname)
            _Registry.architectures.add(architecture)

    def fake_llm(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="processed_logprobs"))

    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.22.0"
    fake_vllm.ModelRegistry = _Registry
    fake_vllm.LLM = fake_llm
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(
        "rl_no_backward.vllm_rollout.ensure_vllm_plugin_discoverable",
        lambda: None,
    )

    engine = create_vllm_engine(
        "Qwen/Qwen2.5-1.5B-Instruct",
        _snapshot(),
        revision="abc123",
        dtype="bfloat16",
        max_model_len=768,
        kv_cache_memory_bytes=123456,
        enforce_eager=True,
        flash_attn_version=2,
        allow_insecure_serialization=True,
        batch_invariant=True,
        disable_log_stats=True,
    )

    assert engine.model_config.logprobs_mode == "processed_logprobs"
    assert captured["registration"][0] == VLLM_RESIDUAL_ARCHITECTURE
    kwargs = captured["kwargs"]
    assert kwargs["skip_tokenizer_init"] is True
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["pipeline_parallel_size"] == 1
    assert kwargs["logprobs_mode"] == "processed_logprobs"
    assert kwargs["kv_cache_memory_bytes"] == 123456
    assert kwargs["attention_config"] == {"flash_attn_version": 2}
    assert kwargs["hf_overrides"]["residual_core_adapter_layers"] == [2, 3]
    assert kwargs["disable_log_stats"] is True
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"


def test_create_engine_rejects_batch_invariance_on_pre_hopper_gpu(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

    with pytest.raises(RuntimeError, match="compute capability 9.0"):
        create_vllm_engine(
            "Qwen/Qwen2.5-1.5B-Instruct",
            _snapshot(),
            revision="abc123",
            dtype="bfloat16",
            max_model_len=768,
            allow_insecure_serialization=True,
            batch_invariant=True,
        )

    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_create_engine_refuses_implicit_insecure_callable_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)

    with pytest.raises(RuntimeError, match="requires explicit trusted-local"):
        create_vllm_engine(
            "Qwen/Qwen2.5-1.5B-Instruct",
            _snapshot(),
            revision="abc123",
            dtype="bfloat16",
            max_model_len=768,
        )

    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ


def test_create_engine_allows_secure_inprocess_callable_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)

    class _Registry:
        architectures: ClassVar[set[str]] = set()

        @classmethod
        def get_supported_archs(cls) -> set[str]:
            assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
            assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ
            return cls.architectures

        @staticmethod
        def register_model(architecture: str, _qualname: str) -> None:
            _Registry.architectures.add(architecture)

    def fake_llm(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="processed_logprobs"))

    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.22.0"
    fake_vllm.ModelRegistry = _Registry
    fake_vllm.LLM = fake_llm
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(
        "rl_no_backward.vllm_rollout.ensure_vllm_plugin_discoverable",
        lambda: None,
    )

    engine = create_vllm_engine(
        "Qwen/Qwen2.5-1.5B-Instruct",
        _snapshot(),
        revision="abc123",
        dtype="bfloat16",
        max_model_len=768,
        allow_insecure_serialization=False,
        batch_invariant=False,
        enable_v1_multiprocessing=False,
    )

    assert engine.model_config.logprobs_mode == "processed_logprobs"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ
    assert captured["kwargs"]
