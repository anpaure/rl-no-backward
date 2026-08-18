"""Shared PEFT LoRA parameterization for matched BP and forward-only RL.

This module intentionally contains no optimizer or rollout code.  It defines
the *single* policy parameterization that both sides of the headline
comparison must use, plus deterministic state/vector helpers.  Keeping this
contract separate prevents either trainer from silently changing the layer
scope, target projections, initialization, or trainable-parameter budget.

PEFT is imported lazily so the main lightweight test environment does not need
the optional standard-GRPO dependencies installed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class StandardLoRAConfig:
    """A conventional, explicitly scoped causal-LM LoRA parameterization.

    The headline default is the widely recognizable PEFT setup requested for
    the comparison: query and value projections in *all* 28 Qwen2.5-1.5B
    blocks, rank eight, alpha sixteen.  It has exactly 1,089,536 trainable
    parameters.  Smaller layer/rank scopes are permitted for smoke tests but
    must remain explicit in artifact metadata.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    layer_indices: tuple[int, ...] = tuple(range(28))
    bias: str = "none"

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int) or self.alpha < 1:
            raise ValueError("LoRA alpha must be a positive integer")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must lie in [0, 1)")
        if not self.target_modules or any(
            not isinstance(name, str) or not name.strip() for name in self.target_modules
        ):
            raise ValueError("target_modules must contain non-empty module names")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")
        if not self.layer_indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.layer_indices
        ):
            raise ValueError("layer_indices must contain non-negative integers")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("layer_indices must be sorted and unique")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("LoRA bias must be none, all, or lora_only")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> StandardLoRAConfig:
        payload = dict(value or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown standard LoRA config keys: {unknown}")
        if "target_modules" in payload:
            payload["target_modules"] = tuple(payload["target_modules"])
        if "layer_indices" in payload:
            payload["layer_indices"] = tuple(payload["layer_indices"])
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_modules"] = list(self.target_modules)
        value["layer_indices"] = list(self.layer_indices)
        return value

    def validate_model_depth(self, num_hidden_layers: int) -> None:
        if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int):
            raise TypeError("num_hidden_layers must be an integer")
        invalid = [index for index in self.layer_indices if index >= num_hidden_layers]
        if invalid:
            raise ValueError(
                f"LoRA layer indices {invalid} exceed model depth {num_hidden_layers}"
            )

    def build_peft_config(self, *, num_hidden_layers: int) -> Any:
        """Build the official :class:`peft.LoraConfig` lazily."""

        self.validate_model_depth(num_hidden_layers)
        try:
            from peft import LoraConfig, TaskType
        except ImportError as error:  # pragma: no cover - optional GPU dependency
            raise ImportError(
                "standard LoRA training requires PEFT; run scripts/install_standard_grpo_h100.sh"
            ) from error
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=float(self.dropout),
            target_modules=list(self.target_modules),
            layers_to_transform=list(self.layer_indices),
            layers_pattern="layers",
            bias=self.bias,
            inference_mode=False,
            init_lora_weights=True,
        )


@dataclass(frozen=True, slots=True)
class LoRAParameterLayout:
    """Stable layout used to flatten and restore the shared LoRA policy."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    dtypes: tuple[str, ...]

    @property
    def parameter_count(self) -> int:
        return sum(self.numels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "shapes": [list(shape) for shape in self.shapes],
            "numels": list(self.numels),
            "dtypes": list(self.dtypes),
            "parameter_count": self.parameter_count,
        }


def _is_lora_parameter_name(name: str) -> bool:
    qualified = f".{name}"
    return ".lora_A." in qualified or ".lora_B." in qualified


def _rng_devices(model: nn.Module) -> list[int]:
    devices = {
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda" and parameter.device.index is not None
    }
    return sorted(devices)


def attach_standard_lora(
    base_model: nn.Module,
    config: StandardLoRAConfig,
    *,
    initialization_seed: int,
) -> nn.Module:
    """Wrap a base causal LM with official PEFT using deterministic init.

    Both compared methods must either call this with the same seed or, more
    strongly, load the same artifact produced by
    :func:`save_shared_lora_initialization` immediately afterward.
    """

    if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int):
        raise TypeError("initialization_seed must be an integer")
    model_config = getattr(base_model, "config", None)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int):
        raise TypeError("base model config does not expose num_hidden_layers")
    peft_config = config.build_peft_config(num_hidden_layers=num_hidden_layers)
    try:
        from peft import get_peft_model
    except ImportError as error:  # pragma: no cover - optional GPU dependency
        raise ImportError(
            "standard LoRA training requires PEFT; run scripts/install_standard_grpo_h100.sh"
        ) from error

    # Do not perturb the caller's global CPU/CUDA RNG streams.  PEFT's default
    # initialization uses Kaiming-uniform A and all-zero B matrices.
    python_rng_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=_rng_devices(base_model)):
            torch.manual_seed(initialization_seed)
            random.seed(initialization_seed)
            model = get_peft_model(base_model, peft_config)
    finally:
        random.setstate(python_rng_state)

    layout = lora_parameter_layout(model)
    named_trainable_lora_parameters(model)
    expected_tensor_count = len(config.layer_indices) * len(config.target_modules) * 2
    if len(layout.names) != expected_tensor_count:
        raise RuntimeError(
            f"PEFT created {len(layout.names)} trainable LoRA tensors; "
            f"expected {expected_tensor_count}"
        )
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")
    resolved_layers = {
        int(match.group(1))
        for name in layout.names
        if (match := layer_pattern.search(name)) is not None
    }
    if resolved_layers != set(config.layer_indices):
        raise RuntimeError(
            f"PEFT resolved LoRA layers {sorted(resolved_layers)}; "
            f"expected {list(config.layer_indices)}"
        )
    for target in config.target_modules:
        if not any(f".{target}." in name for name in layout.names):
            raise RuntimeError(f"PEFT did not create LoRA tensors for {target!r}")
    if set(config.target_modules) == {"q_proj", "v_proj"}:
        hidden_size = getattr(model_config, "hidden_size", None)
        attention_heads = getattr(model_config, "num_attention_heads", None)
        key_value_heads = getattr(model_config, "num_key_value_heads", None)
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None and isinstance(hidden_size, int) and isinstance(attention_heads, int):
            head_dim = hidden_size // attention_heads
        if not all(
            isinstance(value, int)
            for value in (hidden_size, attention_heads, key_value_heads, head_dim)
        ):
            raise TypeError("base model config lacks Qwen attention dimensions")
        expected_count = qwen_projection_lora_parameter_count(
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=key_value_heads,
            head_dim=head_dim,
            config=config,
        )
        if layout.parameter_count != expected_count:
            raise RuntimeError(
                f"PEFT created {layout.parameter_count:,} LoRA parameters; "
                f"Qwen projection shapes imply {expected_count:,}"
            )
    return model


def named_lora_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return the structurally locked PEFT LoRA tensors, frozen or trainable."""

    parameters = tuple(
        sorted(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if _is_lora_parameter_name(name)
        )
    )
    if not parameters:
        raise ValueError("model has no structural LoRA A/B parameters")
    return parameters


def _reject_non_lora_trainables(model: nn.Module) -> None:
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        if not _is_lora_parameter_name(name)
    ]
    if unexpected:
        raise ValueError(f"non-LoRA parameters are trainable: {unexpected[:8]}")


def named_trainable_lora_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return LoRA tensors after asserting the reverse-mode trainability contract."""

    _reject_non_lora_trainables(model)
    parameters = named_lora_parameters(model)
    frozen = [name for name, parameter in parameters if not parameter.requires_grad]
    if frozen:
        raise ValueError(f"LoRA parameters are unexpectedly frozen: {frozen[:8]}")
    return parameters


def assert_lora_frozen(model: nn.Module) -> None:
    """Hard-gate the strict forward-only LoRA state."""

    _reject_non_lora_trainables(model)
    parameters = named_lora_parameters(model)
    trainable = [name for name, parameter in parameters if parameter.requires_grad]
    gradients = [name for name, parameter in parameters if parameter.grad is not None]
    if trainable or gradients:
        raise ValueError(
            f"forward-only LoRA is not frozen/gradient-free: trainable={trainable[:8]}, "
            f"gradients={gradients[:8]}"
        )


def lora_parameter_layout(model: nn.Module) -> LoRAParameterLayout:
    _reject_non_lora_trainables(model)
    parameters = named_lora_parameters(model)
    return LoRAParameterLayout(
        names=tuple(name for name, _ in parameters),
        shapes=tuple(tuple(parameter.shape) for _, parameter in parameters),
        numels=tuple(parameter.numel() for _, parameter in parameters),
        dtypes=tuple(str(parameter.dtype).removeprefix("torch.") for _, parameter in parameters),
    )


@torch.no_grad()
def lora_parameter_vector(model: nn.Module, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Copy all trainable LoRA tensors into one optimizer-neutral vector."""

    tensors = [parameter.detach().reshape(-1).to(dtype=dtype) for _, parameter in named_lora_parameters(model)]
    return torch.cat(tensors)


@torch.no_grad()
def set_lora_parameter_vector(model: nn.Module, vector: Tensor) -> None:
    """Restore a vector into the exact shared LoRA layout without autograd."""

    if not isinstance(vector, Tensor) or vector.ndim != 1:
        raise TypeError("LoRA parameter vector must be a rank-one torch.Tensor")
    parameters = named_lora_parameters(model)
    expected = sum(parameter.numel() for _, parameter in parameters)
    if vector.numel() != expected:
        raise ValueError(f"LoRA vector has {vector.numel()} values; expected {expected}")
    offset = 0
    for _, parameter in parameters:
        size = parameter.numel()
        parameter.copy_(vector[offset : offset + size].reshape(parameter.shape).to(parameter))
        offset += size


@torch.no_grad()
def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a CPU snapshot containing only trainable LoRA tensors."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_lora_parameters(model)
    }


@torch.no_grad()
def load_lora_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Load an exact LoRA-only snapshot, rejecting missing or extra tensors."""

    parameters = dict(named_lora_parameters(model))
    missing = sorted(set(parameters) - set(state))
    extra = sorted(set(state) - set(parameters))
    if missing or extra:
        raise ValueError(f"LoRA state mismatch: missing={missing[:8]}, extra={extra[:8]}")
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != tuple(parameter.shape):
            shape = None if not isinstance(value, Tensor) else tuple(value.shape)
            raise ValueError(f"LoRA tensor {name!r} has shape {shape}; expected {tuple(parameter.shape)}")
        parameter.copy_(value.to(parameter))


def lora_state_digest(model_or_state: nn.Module | Mapping[str, Tensor]) -> str:
    """Canonical SHA-256 receipt for cross-method initialization checks."""

    state = lora_state_dict(model_or_state) if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_base_parameter_digest(model: nn.Module) -> str:
    """Hash every non-LoRA parameter to prove the shared base stayed immutable."""

    digest = hashlib.sha256()
    digest.update(b"rl-no-backward-frozen-base-v1\0")
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if _is_lora_parameter_name(name):
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    if count == 0:
        raise ValueError("model has no frozen base parameters to hash")
    return digest.hexdigest()


def save_shared_lora_initialization(
    path: str | Path,
    model: nn.Module,
    config: StandardLoRAConfig,
) -> Path:
    """Serialize the one initialization that every matched method must load."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite shared LoRA initialization {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    payload = {
        "schema_version": 1,
        "parameterization": config.as_dict(),
        "layout": lora_parameter_layout(model).as_dict(),
        "state_digest": lora_state_digest(state),
        "state": state,
    }
    torch.save(payload, destination)
    return destination


def load_shared_lora_initialization(
    path: str | Path,
    model: nn.Module,
    config: StandardLoRAConfig,
) -> str:
    """Load and validate the shared initialization; return its digest."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("unsupported shared LoRA initialization artifact")
    if payload.get("parameterization") != config.as_dict():
        raise ValueError("shared LoRA initialization parameterization does not match this run")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("shared LoRA initialization has no tensor state")
    expected_digest = payload.get("state_digest")
    actual_digest = lora_state_digest(state)
    if actual_digest != expected_digest:
        raise ValueError("shared LoRA initialization digest is invalid")
    load_lora_state_dict(model, state)
    if lora_state_digest(model) != expected_digest:
        raise RuntimeError("loaded LoRA initialization did not reproduce its digest")
    return str(expected_digest)


def export_vllm_lora_adapter(
    model: nn.Module,
    output_root: str | Path,
    *,
    version: int,
    adapter_name: str = "default",
) -> tuple[Path, dict[str, Any]]:
    """Export one immutable PEFT adapter directory for vLLM 0.22 reloads.

    A fresh versioned path avoids readers observing partially overwritten
    safetensors.  The caller can place ``output_root`` on tmpfs and pass the
    resulting directory to ``LoRARequest(..., load_inplace=True)``.  Only the
    current on-policy adapter needs to be reloaded once per rollout; finite-
    difference rescoring remains in the inference-only HF policy.
    """

    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("LoRA export version must be a non-negative integer")
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError("adapter_name must be a non-empty string")
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("model must be a PEFT model with save_pretrained")
    layout = lora_parameter_layout(model)
    digest = lora_state_digest(model)
    destination = Path(output_root) / f"adapter-{version:06d}-{digest[:12]}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite LoRA export {destination}")
    destination.mkdir(parents=True)
    save_pretrained(
        destination,
        safe_serialization=True,
        selected_adapters=[adapter_name],
    )
    required = (destination / "adapter_config.json", destination / "adapter_model.safetensors")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"PEFT export is incomplete; missing {missing}")
    weights_path = destination / "adapter_model.safetensors"
    file_digest = hashlib.sha256()
    with weights_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            file_digest.update(chunk)
    receipt = {
        "schema_version": 1,
        "version": version,
        "adapter_name": adapter_name,
        "state_digest": digest,
        "adapter_model_sha256": file_digest.hexdigest(),
        "parameter_count": layout.parameter_count,
        "path": str(destination.resolve()),
        "path_is_transient": True,
        "durable_hash_fields": ["state_digest", "adapter_model_sha256"],
        "vllm_reload_semantics": "LoRARequest(load_inplace=True)",
    }
    (destination / "rl_no_backward_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination, receipt


def make_vllm_lora_request(
    adapter_path: str | Path,
    *,
    lora_name: str = "matched-policy",
    lora_int_id: int = 1,
) -> Any:
    """Create vLLM 0.22's in-place RL adapter reload request lazily."""

    path = Path(adapter_path)
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"not a PEFT adapter directory: {path}")
    try:
        from vllm.lora.request import LoRARequest
    except ImportError as error:  # pragma: no cover - optional GPU dependency
        raise ImportError("vLLM LoRA rollout requires vllm>=0.22") from error
    return LoRARequest(
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        lora_path=str(path.resolve()),
        load_inplace=True,
    )


def qwen_projection_lora_parameter_count(
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    config: StandardLoRAConfig,
) -> int:
    """Calculate the expected Qwen q/v LoRA count without loading weights."""

    if set(config.target_modules) != {"q_proj", "v_proj"}:
        raise ValueError("closed-form count is defined only for q_proj/v_proj")
    q_out = num_attention_heads * head_dim
    v_out = num_key_value_heads * head_dim
    per_layer = config.rank * (hidden_size + q_out) + config.rank * (hidden_size + v_out)
    return len(config.layer_indices) * per_layer


__all__ = [
    "LoRAParameterLayout",
    "StandardLoRAConfig",
    "assert_lora_frozen",
    "attach_standard_lora",
    "export_vllm_lora_adapter",
    "frozen_base_parameter_digest",
    "load_lora_state_dict",
    "load_shared_lora_initialization",
    "lora_parameter_layout",
    "lora_parameter_vector",
    "lora_state_dict",
    "lora_state_digest",
    "make_vllm_lora_request",
    "named_lora_parameters",
    "named_trainable_lora_parameters",
    "qwen_projection_lora_parameter_count",
    "save_shared_lora_initialization",
    "set_lora_parameter_vector",
]
