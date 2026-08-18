"""Frozen language model with tiny activation-calibrated residual adapters.

Only each adapter's ``core`` matrix is trainable.  The base model and PCA bases
remain frozen.  This keeps the searchable parameter vector genuinely small and
makes strict inference-only optimization practical.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualCoreAdapter(nn.Module):
    """Apply ``h + scale * P C Q^T LN(h)`` with fixed P/Q and trainable C."""

    def __init__(self, p_basis: Tensor, q_basis: Tensor, scale: float = 1.0) -> None:
        super().__init__()
        if p_basis.shape != q_basis.shape or p_basis.ndim != 2:
            raise ValueError("P and Q must have the same [hidden_size, rank] shape")
        self.register_buffer("p_basis", p_basis.float().contiguous())
        self.register_buffer("q_basis", q_basis.float().contiguous())
        self.core = nn.Parameter(torch.zeros(p_basis.shape[1], p_basis.shape[1]))
        self.scale = float(scale)

    @property
    def rank(self) -> int:
        return self.core.shape[0]

    def forward(self, hidden_states: Tensor) -> Tensor:
        original_dtype = hidden_states.dtype
        normalized = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],))
        coordinates = normalized @ self.q_basis
        delta = (coordinates @ self.core.T) @ self.p_basis.T
        return hidden_states + (self.scale * delta).to(original_dtype)


class AdapterWrappedLayer(nn.Module):
    """Wrap a Hugging Face decoder layer without changing its call signature."""

    def __init__(self, layer: nn.Module, adapter: ResidualCoreAdapter) -> None:
        super().__init__()
        self.layer = layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        if isinstance(output, tuple):
            return (self.adapter(output[0]), *output[1:])
        return self.adapter(output)


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: object
    candidate_token_ids: Tensor
    adapter_names: list[str]
    device: torch.device
    model_name: str

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        named = dict(self.model.named_parameters())
        return [named[name] for name in self.adapter_names]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters)


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve the common decoder-layer container used by Qwen/Llama-like models."""

    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if isinstance(layers, nn.ModuleList):
            return layers
    raise TypeError("Unsupported model: could not locate decoder ModuleList")


def _top_activation_basis(samples: Tensor, rank: int) -> Tensor:
    """Return deterministic top eigenvectors of the uncentered activation moment."""

    if samples.ndim != 2:
        raise ValueError("activation samples must have shape [tokens, hidden_size]")
    normalized = F.layer_norm(samples.float(), (samples.shape[-1],))
    covariance = normalized.T @ normalized / max(normalized.shape[0], 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -rank:]
    # Eigenvector signs are arbitrary. Canonicalize them for reproducible hashes.
    pivot_rows = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivot_rows, torch.arange(rank, device=basis.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs).contiguous()


@torch.inference_mode()
def calibrate_activation_bases(
    model: nn.Module,
    tokenizer: object,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    rank: int,
    device: torch.device,
    max_tokens: int = 4096,
) -> dict[int, Tensor]:
    """Collect frozen activations and estimate a PCA basis for each target layer."""

    layers = _decoder_layers(model)
    captured: dict[int, list[Tensor]] = {index: [] for index in layer_indices}
    handles = []
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    calibration_mask = encoded.get("attention_mask")

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            detached = hidden.detach()
            if (
                calibration_mask is not None
                and detached.ndim == 3
                and calibration_mask.shape == detached.shape[:2]
            ):
                detached = detached[calibration_mask.bool()]
            else:
                detached = detached.reshape(-1, detached.shape[-1])
            captured[index].append(detached.float().cpu())

        return hook

    for index in layer_indices:
        handles.append(layers[index].register_forward_hook(make_hook(index)))

    try:
        model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    bases: dict[int, Tensor] = {}
    for index in layer_indices:
        samples = torch.cat(captured[index], dim=0)[:max_tokens].to(device)
        if samples.shape[0] < rank:
            raise ValueError(f"layer {index} has only {samples.shape[0]} calibration samples")
        bases[index] = _top_activation_basis(samples, rank).cpu()
    return bases


def _candidate_ids(tokenizer: object, candidates: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate {candidate!r} is not one token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError("candidate strings map to duplicate token ids")
    return ids


def load_model_bundle(
    model_name: str,
    calibration_prompts: Sequence[str],
    candidates: Sequence[str],
    adapter_rank: int = 16,
    adapter_layers: int = 4,
    adapter_scale: float = 1.0,
    dtype: str = "bfloat16",
    device: str = "cuda",
    revision: str | None = None,
) -> ModelBundle:
    """Load a frozen causal LM, calibrate bases, and insert residual-core adapters."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_device = torch.device(device)
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        # Eager attention avoids cuDNN-SDPA planner failures observed on the
        # remote H100 software stack for short, heavily padded prompt batches.
        attn_implementation="eager",
    ).to(target_device)
    model.eval()
    model.requires_grad_(False)

    layers = _decoder_layers(model)
    if adapter_layers > len(layers):
        raise ValueError(f"requested {adapter_layers} adapters for {len(layers)} layers")
    indices = list(range(len(layers) - adapter_layers, len(layers)))
    bases = calibrate_activation_bases(
        model=model,
        tokenizer=tokenizer,
        prompts=calibration_prompts,
        layer_indices=indices,
        rank=adapter_rank,
        device=target_device,
    )

    for index in indices:
        basis = bases[index].to(target_device)
        adapter = ResidualCoreAdapter(basis, basis, scale=adapter_scale).to(target_device)
        layers[index] = AdapterWrappedLayer(layers[index], adapter)

    adapter_names = [
        name for name, parameter in model.named_parameters() if name.endswith("adapter.core")
    ]
    if len(adapter_names) != adapter_layers:
        raise RuntimeError(f"expected {adapter_layers} adapter cores, found {adapter_names}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in adapter_names)

    ids = torch.tensor(_candidate_ids(tokenizer, candidates), device=target_device)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        candidate_token_ids=ids,
        adapter_names=adapter_names,
        device=target_device,
        model_name=model_name,
    )


def tokenize_prompts(
    bundle: ModelBundle, prompts: Sequence[str], max_length: int = 128
) -> dict[str, Tensor]:
    encoded = bundle.tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {name: value.to(bundle.device) for name, value in encoded.items()}


def candidate_log_probs(bundle: ModelBundle, encoded: dict[str, Tensor]) -> Tensor:
    """Return log probabilities for the constrained one-token action space."""

    outputs = bundle.model(**encoded, use_cache=False)
    logits = outputs.logits[:, -1, :].index_select(-1, bundle.candidate_token_ids)
    return logits.float().log_softmax(dim=-1)


def parameter_vector(bundle: ModelBundle) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in bundle.trainable_parameters])


@torch.no_grad()
def set_parameter_vector(bundle: ModelBundle, vector: Tensor) -> None:
    offset = 0
    for parameter in bundle.trainable_parameters:
        size = parameter.numel()
        parameter.copy_(vector[offset : offset + size].view_as(parameter))
        offset += size
    if offset != vector.numel():
        raise ValueError(f"vector has {vector.numel()} entries but model consumed {offset}")


def set_adapter_grad_enabled(bundle: ModelBundle, enabled: bool) -> None:
    for parameter in bundle.trainable_parameters:
        parameter.requires_grad_(enabled)


def orthonormalize(columns: Iterable[Tensor], dimension: int, device: torch.device) -> Tensor:
    materialized = [
        column.reshape(dimension, 1).to(device=device, dtype=torch.float32) for column in columns
    ]
    if not materialized:
        return torch.empty(dimension, 0, device=device)
    matrix = torch.cat(materialized, dim=1)
    return torch.linalg.qr(matrix, mode="reduced").Q
