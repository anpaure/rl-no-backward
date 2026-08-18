from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_no_backward.model as model_module
from rl_no_backward.model import (
    AdapterWrappedLayer,
    compile_model_forward_in_place,
    installed_flash_attn_version,
    load_model_bundle,
    model_forward_is_compiled,
    resolved_attention_implementation,
    validate_attention_implementation,
    validate_compile_mode,
)


class TinyForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.weight


def test_forward_compilation_preserves_parameter_names_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def fake_compile(function: object, *, mode: str, fullgraph: bool, dynamic: bool):
        calls.append((mode, fullgraph, dynamic))

        def compiled(*args: object, **kwargs: object):
            return function(*args, **kwargs)  # type: ignore[operator]

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = TinyForwardModel()
    names_before = tuple(dict(model.named_parameters()))

    compile_model_forward_in_place(model, "reduce-overhead")

    assert tuple(dict(model.named_parameters())) == names_before
    assert model_forward_is_compiled(model)
    assert model(torch.tensor(3.0)).item() == pytest.approx(6.0)
    assert calls == [("reduce-overhead", False, True)]

    compile_model_forward_in_place(model, "reduce-overhead")
    assert calls == [("reduce-overhead", False, True)]
    with pytest.raises(ValueError, match="already compiled"):
        compile_model_forward_in_place(model, "max-autotune")


def test_runtime_control_validators_reject_unrecognized_strings() -> None:
    validate_attention_implementation("flash_attention_2")
    validate_compile_mode("max-autotune-no-cudagraphs")
    with pytest.raises(ValueError, match="attention_implementation"):
        validate_attention_implementation("custom")
    with pytest.raises(ValueError, match="compile_model_forward_mode"):
        validate_compile_mode("custom")


class TinyDecoderLayer(nn.Module):
    def forward(self, hidden_states: Tensor, *args: object, **kwargs: object) -> tuple[Tensor]:
        del args, kwargs
        return (hidden_states,)


class TinyCausalLM(nn.Module):
    def __init__(self, attention_implementation: str) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyDecoderLayer()])
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.config = SimpleNamespace(
            _attn_implementation=attention_implementation,
            _commit_hash="fixed-revision",
        )

    def forward(self, input_ids: Tensor, **kwargs: object) -> SimpleNamespace:
        del kwargs
        hidden = torch.nn.functional.one_hot(input_ids % 2, num_classes=2).float()
        hidden = self.model.layers[0](hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, candidate: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [int(candidate) + 2]


def test_model_loader_passes_attention_and_compiles_after_adapter_insertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_load_kwargs: dict[str, object] = {}
    compile_parameter_names: list[tuple[str, ...]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(_name: str, **kwargs: object) -> TinyCausalLM:
            observed_load_kwargs.update(kwargs)
            return TinyCausalLM(str(kwargs["attn_implementation"]))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, **kwargs: object) -> TinyTokenizer:
            del kwargs
            return TinyTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = AutoModel
    fake_transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        model_module,
        "calibrate_activation_bases",
        lambda **_kwargs: {0: torch.tensor([[1.0], [0.0]])},
    )

    def fake_compile(function: object, *, mode: str, fullgraph: bool, dynamic: bool):
        del mode, fullgraph
        assert dynamic
        owning_model = function.__self__  # type: ignore[attr-defined]
        compile_parameter_names.append(tuple(dict(owning_model.named_parameters())))

        def compiled(*args: object, **kwargs: object):
            return function(*args, **kwargs)  # type: ignore[operator]

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    bundle = load_model_bundle(
        model_name="tiny",
        calibration_prompts=["calibrate"],
        candidates=["0", "1"],
        adapter_rank=1,
        adapter_layers=1,
        dtype="float32",
        device="cpu",
        attention_implementation="sdpa",
        compile_model_forward=True,
        compile_model_forward_mode="reduce-overhead",
    )

    assert observed_load_kwargs["attn_implementation"] == "sdpa"
    assert isinstance(bundle.model.model.layers[0], AdapterWrappedLayer)
    assert bundle.adapter_names == ["model.layers.0.adapter.core"]
    assert compile_parameter_names == [("model.layers.0.adapter.core", "lm_head.weight")]
    assert tuple(dict(bundle.model.named_parameters())) == compile_parameter_names[0]
    assert resolved_attention_implementation(bundle.model) == "sdpa"
    assert model_forward_is_compiled(bundle.model)


def test_flash_version_helper_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "version", lambda _name: "2.8.3")
    assert installed_flash_attn_version() == "2.8.3"

    def missing(_name: str) -> str:
        raise model_module.PackageNotFoundError

    monkeypatch.setattr(model_module, "version", missing)
    assert installed_flash_attn_version() is None
