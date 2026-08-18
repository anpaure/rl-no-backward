from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn

from rl_no_backward import standard_lora
from rl_no_backward.standard_lora import (
    StandardLoRAConfig,
    assert_lora_frozen,
    export_vllm_lora_adapter,
    frozen_base_parameter_digest,
    load_shared_lora_initialization,
    lora_parameter_layout,
    lora_parameter_vector,
    lora_state_digest,
    named_lora_parameters,
    named_trainable_lora_parameters,
    qwen_projection_lora_parameter_count,
    save_shared_lora_initialization,
    set_lora_parameter_vector,
)


class _FakeLoRALinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, rank: int) -> None:
        super().__init__()
        self.base_layer = nn.Linear(input_size, output_size, bias=False)
        self.base_layer.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(input_size, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, output_size, bias=False)})


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _FakeLoRALinear(5, 7, 2)
        self.v_proj = _FakeLoRALinear(5, 3, 2)
        self.norm = nn.LayerNorm(5)
        self.norm.requires_grad_(False)

    def save_pretrained(
        self,
        destination,
        *,
        safe_serialization: bool,
        selected_adapters: list[str],
    ) -> None:
        assert safe_serialization is True
        assert selected_adapters == ["default"]
        destination.joinpath("adapter_config.json").write_text("{}\n", encoding="utf-8")
        destination.joinpath("adapter_model.safetensors").write_bytes(b"safe")


def test_default_qwen_parameter_count_is_explicit() -> None:
    config = StandardLoRAConfig()
    assert qwen_projection_lora_parameter_count(
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=128,
        config=config,
    ) == 1_089_536


def test_config_validates_layer_scope_and_mapping() -> None:
    config = StandardLoRAConfig.from_mapping(
        {
            "rank": 2,
            "alpha": 4,
            "target_modules": ["q_proj", "v_proj"],
            "layer_indices": [26, 27],
        }
    )
    assert config.layer_indices == (26, 27)
    config.validate_model_depth(28)
    with pytest.raises(ValueError, match="exceed model depth"):
        config.validate_model_depth(27)
    with pytest.raises(ValueError, match="sorted and unique"):
        StandardLoRAConfig(layer_indices=(27, 26))


def test_vector_round_trip_and_non_lora_trainable_guard() -> None:
    model = _FakePolicy()
    layout = lora_parameter_layout(model)
    assert layout.parameter_count == 40
    original = lora_parameter_vector(model)
    changed = torch.arange(original.numel(), dtype=torch.float32)
    set_lora_parameter_vector(model, changed)
    torch.testing.assert_close(lora_parameter_vector(model), changed)
    set_lora_parameter_vector(model, original)
    torch.testing.assert_close(lora_parameter_vector(model), original)

    model.norm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="non-LoRA"):
        lora_parameter_layout(model)


def test_shared_initialization_has_verified_digest(tmp_path) -> None:
    model = _FakePolicy()
    config = StandardLoRAConfig()
    destination = save_shared_lora_initialization(tmp_path / "initial.pt", model, config)
    expected = lora_state_digest(model)
    with pytest.raises(FileExistsError):
        save_shared_lora_initialization(destination, model, config)
    set_lora_parameter_vector(model, torch.zeros_like(lora_parameter_vector(model)))
    assert lora_state_digest(model) != expected
    assert load_shared_lora_initialization(destination, model, config) == expected
    assert lora_state_digest(model) == expected


def test_vllm_export_is_immutable_and_receipted(tmp_path) -> None:
    model = _FakePolicy()
    destination, receipt = export_vllm_lora_adapter(model, tmp_path, version=7)
    assert destination.name.startswith("adapter-000007-")
    assert receipt["parameter_count"] == 40
    assert receipt["adapter_model_sha256"] == (
        "8b3369944dd2a3fab39e32d1aeb1f763946a458ae3e6368a46432adc8f3a0860"
    )
    assert receipt["vllm_reload_semantics"] == "LoRARequest(load_inplace=True)"
    assert receipt["path_is_transient"] is True
    assert set(receipt["durable_hash_fields"]) == {"state_digest", "adapter_model_sha256"}
    assert destination.joinpath("rl_no_backward_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        export_vllm_lora_adapter(model, tmp_path, version=7)


def test_frozen_forward_only_lora_can_move_digest_and_export(tmp_path) -> None:
    model = _FakePolicy()
    for _, parameter in named_lora_parameters(model):
        parameter.requires_grad_(False)
        parameter.grad = None
    assert_lora_frozen(model)
    with pytest.raises(ValueError, match="unexpectedly frozen"):
        named_trainable_lora_parameters(model)

    before = lora_parameter_vector(model)
    before_digest = lora_state_digest(model)
    set_lora_parameter_vector(model, before + 0.01)
    assert lora_state_digest(model) != before_digest
    destination, receipt = export_vllm_lora_adapter(model, tmp_path, version=8)
    assert destination.is_dir()
    assert receipt["state_digest"] == lora_state_digest(model)
    assert_lora_frozen(model)


def test_frozen_base_digest_ignores_lora_but_detects_base_change() -> None:
    model = _FakePolicy()
    baseline = frozen_base_parameter_digest(model)
    set_lora_parameter_vector(model, lora_parameter_vector(model) + 0.25)
    assert frozen_base_parameter_digest(model) == baseline
    with torch.no_grad():
        model.q_proj.base_layer.weight.add_(0.01)
    assert frozen_base_parameter_digest(model) != baseline


def test_shared_lora_helpers_never_call_backward_or_autograd_grad() -> None:
    tree = ast.parse(inspect.getsource(standard_lora))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
            forbidden.append("backward")
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "autograd"
        ):
            forbidden.append("autograd")
    assert forbidden == []
