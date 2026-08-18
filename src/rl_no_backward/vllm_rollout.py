"""Optional vLLM rollout support for the residual-core policy.

The training model keeps each Qwen decoder block in ordinary Hugging Face
form, where a block returns its fully materialized hidden state.  vLLM keeps
the same state split into an MLP branch and a residual tensor.  If ``b`` and
``r`` are those tensors, the Hugging Face block output is ``h = b + r``.
Consequently the post-block residual adapter can be represented in vLLM as::

    delta = adapter_delta(b + r)
    return b + delta, r

The next fused residual add observes ``b + delta + r``, which is algebraically
the same policy as ``adapter(b + r)``.  This module contains the dependency-free
state and synchronization side of that integration.  The vLLM-specific Qwen
model lives in :mod:`rl_no_backward.vllm_qwen_model` and is imported lazily so
the normal package and CPU test suite do not require vLLM.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import ModelBundle, ResidualCoreAdapter, _decoder_layers
from .vllm_plugin import (
    VLLM_RESIDUAL_ARCHITECTURE,
    configure_trusted_vllm_callable_serialization,
    configure_vllm_batch_invariance,
    configure_vllm_v1_multiprocessing,
    ensure_vllm_plugin_discoverable,
    register_residual_qwen_model,
)


def _canonical_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class ResidualAdapterLayerState:
    """Canonical CPU state for one post-block residual-core adapter."""

    layer_index: int
    p_basis: Tensor
    q_basis: Tensor
    core: Tensor
    scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise ValueError("adapter scale must be finite")
        p_basis = _canonical_tensor("p_basis", self.p_basis)
        q_basis = _canonical_tensor("q_basis", self.q_basis)
        core = _canonical_tensor("core", self.core)
        if p_basis.ndim != 2 or p_basis.shape != q_basis.shape:
            raise ValueError("P and Q must have the same [hidden_size, rank] shape")
        rank = p_basis.shape[1]
        if rank < 1 or core.shape != (rank, rank):
            raise ValueError("core must have shape [rank, rank] matching the bases")
        object.__setattr__(self, "p_basis", p_basis)
        object.__setattr__(self, "q_basis", q_basis)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "scale", scale)

    @property
    def rank(self) -> int:
        return int(self.core.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.p_basis.shape[0])


@dataclass(frozen=True, slots=True)
class ResidualAdapterSnapshot:
    """Ordered, immutable-by-convention snapshot of every adapter tensor."""

    layers: tuple[ResidualAdapterLayerState, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("an adapter snapshot must contain at least one layer")
        ordered = tuple(sorted(self.layers, key=lambda layer: layer.layer_index))
        indices = tuple(layer.layer_index for layer in ordered)
        if len(indices) != len(set(indices)):
            raise ValueError("adapter layer indices must be unique")
        ranks = {layer.rank for layer in ordered}
        if len(ranks) != 1:
            raise ValueError("all residual-core adapters must use the same rank")
        object.__setattr__(self, "layers", ordered)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers)

    @property
    def rank(self) -> int:
        return self.layers[0].rank

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(layer.scale for layer in self.layers)

    def digest(self, *, include_bases: bool) -> str:
        """Return a deterministic digest for synchronization receipts."""

        digest = hashlib.sha256()
        digest.update(b"rl-no-backward-residual-adapter-v1\0")
        for layer in self.layers:
            digest.update(struct.pack("<q", layer.layer_index))
            digest.update(struct.pack("<d", layer.scale))
            tensors = (
                (("p_basis", layer.p_basis), ("q_basis", layer.q_basis)) if include_bases else ()
            )
            tensors = (*tensors, ("core", layer.core))
            for name, tensor in tensors:
                digest.update(name.encode("ascii") + b"\0")
                digest.update(struct.pack("<q", tensor.ndim))
                digest.update(struct.pack(f"<{tensor.ndim}q", *tensor.shape))
                digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def fixed_digest(self) -> str:
        """Identify the immutable adapter layout, bases, and scales."""

        digest = hashlib.sha256()
        digest.update(b"rl-no-backward-residual-adapter-fixed-v1\0")
        for layer in self.layers:
            digest.update(struct.pack("<q", layer.layer_index))
            digest.update(struct.pack("<d", layer.scale))
            for name, tensor in (("p_basis", layer.p_basis), ("q_basis", layer.q_basis)):
                digest.update(name.encode("ascii") + b"\0")
                digest.update(struct.pack("<q", tensor.ndim))
                digest.update(struct.pack(f"<{tensor.ndim}q", *tensor.shape))
                digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def core_update(self) -> ResidualAdapterCoreUpdate:
        """Return the compact recurrent payload (256 FP32 values at rank 8)."""

        return ResidualAdapterCoreUpdate(
            tuple(
                ResidualAdapterCoreLayerState(
                    layer_index=layer.layer_index,
                    core=layer.core,
                    scale=layer.scale,
                )
                for layer in self.layers
            )
        )


@dataclass(frozen=True, slots=True)
class ResidualAdapterCoreLayerState:
    """Compact mutable state for one already-initialized vLLM adapter."""

    layer_index: int
    core: Tensor
    scale: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        core = _canonical_tensor("core", self.core)
        if core.ndim != 2 or core.shape[0] < 1 or core.shape[0] != core.shape[1]:
            raise ValueError("core must be a non-empty square matrix")
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise ValueError("adapter scale must be finite")
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, slots=True)
class ResidualAdapterCoreUpdate:
    """Core-only update suitable for vLLM's small control-plane RPC."""

    layers: tuple[ResidualAdapterCoreLayerState, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a core update must contain at least one layer")
        ordered = tuple(sorted(self.layers, key=lambda layer: layer.layer_index))
        indices = tuple(layer.layer_index for layer in ordered)
        if len(indices) != len(set(indices)):
            raise ValueError("core-update layer indices must be unique")
        object.__setattr__(self, "layers", ordered)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"rl-no-backward-residual-adapter-v1\0")
        for layer in self.layers:
            digest.update(struct.pack("<q", layer.layer_index))
            digest.update(struct.pack("<d", layer.scale))
            digest.update(b"core\0")
            digest.update(struct.pack("<q", layer.core.ndim))
            digest.update(struct.pack(f"<{layer.core.ndim}q", *layer.core.shape))
            digest.update(layer.core.numpy().tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VLLMGroupedGeneration:
    """A prompt-major group sampled by one installed vLLM policy version."""

    prompt_token_ids: tuple[tuple[int, ...], ...]
    response_input_ids: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor
    finish_reasons: tuple[tuple[str | None, ...], ...]
    policy_version: str

    def __post_init__(self) -> None:
        if self.response_input_ids.ndim != 3:
            raise ValueError("response_input_ids must have shape [batch, group, tokens]")
        if self.response_mask.shape != self.response_input_ids.shape:
            raise ValueError("response_mask must match response_input_ids")
        if self.old_token_log_probs.shape != self.response_input_ids.shape:
            raise ValueError("old_token_log_probs must match response_input_ids")
        if self.response_mask.dtype != torch.bool:
            raise TypeError("response_mask must have boolean dtype")
        if self.old_token_log_probs.requires_grad:
            raise ValueError("old_token_log_probs must be detached")
        valid_log_probs = self.old_token_log_probs[self.response_mask]
        if not torch.isfinite(valid_log_probs).all():
            raise ValueError("sampled token log probabilities must be finite")
        batch_size, group_size, _ = self.response_input_ids.shape
        if len(self.prompt_token_ids) != batch_size:
            raise ValueError("prompt count does not match generated responses")
        if len(self.finish_reasons) != batch_size or any(
            len(group) != group_size for group in self.finish_reasons
        ):
            raise ValueError("finish reasons must have shape [batch][group]")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")

    @property
    def batch_size(self) -> int:
        return int(self.response_input_ids.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.response_input_ids.shape[1])

    @property
    def valid_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())


@dataclass(frozen=True, slots=True)
class VLLMRepeatDeterminismReport:
    """Exact repeatability evidence for two same-policy, same-seed requests."""

    policy_version: str
    policy_state_digest: str
    seed: int
    response_input_ids_equal: bool
    response_masks_equal: bool
    behavior_logprobs_bitwise_equal: bool
    maximum_behavior_logprob_abs_delta: float | None
    first_valid_response_tokens: int
    second_valid_response_tokens: int

    @property
    def token_sequences_equal(self) -> bool:
        return self.response_input_ids_equal and self.response_masks_equal

    @property
    def fully_repeatable(self) -> bool:
        return self.token_sequences_equal and self.behavior_logprobs_bitwise_equal


@dataclass(frozen=True, slots=True)
class VLLMGreedyGeneration:
    """Greedy token sequences produced by one installed policy version."""

    prompt_token_ids: tuple[tuple[int, ...], ...]
    response_token_ids: tuple[tuple[int, ...], ...]
    finish_reasons: tuple[str | None, ...]
    policy_version: str

    def __post_init__(self) -> None:
        count = len(self.prompt_token_ids)
        if count == 0 or len(self.response_token_ids) != count:
            raise ValueError("greedy responses must match a non-empty prompt batch")
        if len(self.finish_reasons) != count:
            raise ValueError("greedy finish reasons must match the prompt batch")
        if any(not response for response in self.response_token_ids):
            raise ValueError("vLLM returned an empty greedy completion")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")


def capture_residual_adapter_snapshot(bundle: ModelBundle) -> ResidualAdapterSnapshot:
    """Copy the calibrated Hugging Face adapter state to canonical CPU tensors."""

    states: list[ResidualAdapterLayerState] = []
    for layer_index, layer in enumerate(_decoder_layers(bundle.model)):
        adapter = getattr(layer, "adapter", None)
        if adapter is None:
            continue
        if not isinstance(adapter, ResidualCoreAdapter):
            raise TypeError(
                f"decoder layer {layer_index} has an unsupported adapter {type(adapter).__name__}"
            )
        states.append(
            ResidualAdapterLayerState(
                layer_index=layer_index,
                p_basis=adapter.p_basis,
                q_basis=adapter.q_basis,
                core=adapter.core,
                scale=adapter.scale,
            )
        )
    if len(states) != len(bundle.adapter_names):
        raise ValueError(
            "decoder adapter layout does not match ModelBundle.adapter_names: "
            f"found {len(states)} adapters for {len(bundle.adapter_names)} names"
        )
    return ResidualAdapterSnapshot(tuple(states))


class VLLMResidualCoreAdapter(nn.Module):
    """Inference-only residual-core adapter stored inside the vLLM model.

    Bases and core use FP32 exactly like :class:`ResidualCoreAdapter`.  The
    buffers start at zero and must be populated before the first adapted
    rollout with :func:`sync_vllm_adapter_snapshot`.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        scale: float,
        layer_index: int,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        self.layer_index = int(layer_index)
        self.scale = float(scale)
        self.register_buffer("p_basis", torch.zeros(hidden_size, rank, dtype=torch.float32))
        self.register_buffer("q_basis", torch.zeros(hidden_size, rank, dtype=torch.float32))
        # Keep inference-only state as buffers. vLLM 0.22 strictly checks that
        # every named parameter came from the base checkpoint; a new Parameter
        # would therefore make an otherwise valid Qwen checkpoint fail loading.
        self.register_buffer("core", torch.zeros(rank, rank, dtype=torch.float32))

    @property
    def rank(self) -> int:
        return int(self.core.shape[0])

    def delta(self, hidden_states: Tensor) -> Tensor:
        """Return only ``scale * P C Q^T LN(h)`` in the input dtype."""

        original_dtype = hidden_states.dtype
        normalized = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],))
        coordinates = normalized @ self.q_basis
        delta = (coordinates @ self.core.T) @ self.p_basis.T
        return (self.scale * delta).to(original_dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.delta(hidden_states)


def apply_adapter_to_vllm_split_state(
    branch: Tensor,
    residual: Tensor,
    adapter: VLLMResidualCoreAdapter,
) -> tuple[Tensor, Tensor]:
    """Apply a post-block adapter while preserving vLLM's split state."""

    if branch.shape != residual.shape:
        raise ValueError("branch and residual tensors must have the same shape")
    full_hidden_state = branch + residual
    return branch + adapter.delta(full_hidden_state), residual


def _vllm_adapters(model: nn.Module) -> dict[int, VLLMResidualCoreAdapter]:
    result: dict[int, VLLMResidualCoreAdapter] = {}
    for module in model.modules():
        if not isinstance(module, VLLMResidualCoreAdapter):
            continue
        if module.layer_index in result:
            raise ValueError(f"duplicate vLLM adapter for layer {module.layer_index}")
        result[module.layer_index] = module
    return result


@torch.inference_mode()
def apply_vllm_adapter_snapshot(
    model: nn.Module,
    *,
    snapshot: ResidualAdapterSnapshot,
    version: str,
    include_bases: bool,
) -> dict[str, Any]:
    """Worker-side callable used by ``LLM.apply_model``.

    All shapes and layer identities are checked before the first mutation.
    The returned receipt is deliberately tiny and can safely cross vLLM's
    control plane.
    """

    if not isinstance(version, str) or not version.strip():
        raise ValueError("adapter version must be a non-empty string")
    targets = _vllm_adapters(model)
    expected_indices = set(snapshot.layer_indices)
    if set(targets) != expected_indices:
        raise ValueError(
            "vLLM adapter layers do not match the snapshot: "
            f"model={sorted(targets)}, snapshot={sorted(expected_indices)}"
        )

    # Validate the complete update before mutating any worker tensor.
    for source in snapshot.layers:
        target = targets[source.layer_index]
        if target.p_basis.shape != source.p_basis.shape:
            raise ValueError(f"P basis shape mismatch at layer {source.layer_index}")
        if target.q_basis.shape != source.q_basis.shape:
            raise ValueError(f"Q basis shape mismatch at layer {source.layer_index}")
        if target.core.shape != source.core.shape:
            raise ValueError(f"core shape mismatch at layer {source.layer_index}")
        if target.scale != source.scale:
            raise ValueError(f"adapter scale mismatch at layer {source.layer_index}")

    for source in snapshot.layers:
        target = targets[source.layer_index]
        if include_bases:
            target.p_basis.copy_(source.p_basis.to(target.p_basis.device))
            target.q_basis.copy_(source.q_basis.to(target.q_basis.device))
        target.core.copy_(source.core.to(target.core.device))

    # Read back the small update.  Besides detecting a failed copy, this makes
    # CUDA completion part of the RPC boundary before generation can resume.
    for source in snapshot.layers:
        target = targets[source.layer_index]
        if not torch.equal(target.core.detach().float().cpu(), source.core):
            raise RuntimeError(f"core verification failed at layer {source.layer_index}")
        if include_bases:
            if not torch.equal(target.p_basis.detach().float().cpu(), source.p_basis):
                raise RuntimeError(f"P basis verification failed at layer {source.layer_index}")
            if not torch.equal(target.q_basis.detach().float().cpu(), source.q_basis):
                raise RuntimeError(f"Q basis verification failed at layer {source.layer_index}")

    return {
        "version": version,
        "digest": snapshot.digest(include_bases=include_bases),
        "include_bases": bool(include_bases),
        "layer_indices": list(snapshot.layer_indices),
    }


@torch.inference_mode()
def apply_vllm_core_update(
    model: nn.Module,
    *,
    update: ResidualAdapterCoreUpdate,
    version: str,
) -> dict[str, Any]:
    """Apply only mutable cores after the one-time full-state installation."""

    if not isinstance(version, str) or not version.strip():
        raise ValueError("adapter version must be a non-empty string")
    targets = _vllm_adapters(model)
    expected_indices = set(update.layer_indices)
    if set(targets) != expected_indices:
        raise ValueError(
            "vLLM adapter layers do not match the core update: "
            f"model={sorted(targets)}, update={sorted(expected_indices)}"
        )
    for source in update.layers:
        target = targets[source.layer_index]
        if target.core.shape != source.core.shape:
            raise ValueError(f"core shape mismatch at layer {source.layer_index}")
        if target.scale != source.scale:
            raise ValueError(f"adapter scale mismatch at layer {source.layer_index}")
    for source in update.layers:
        targets[source.layer_index].core.copy_(
            source.core.to(targets[source.layer_index].core.device)
        )
    for source in update.layers:
        actual = targets[source.layer_index].core.detach().float().cpu()
        if not torch.equal(actual, source.core):
            raise RuntimeError(f"core verification failed at layer {source.layer_index}")
    return {
        "version": version,
        "digest": update.digest(),
        "include_bases": False,
        "layer_indices": list(update.layer_indices),
    }


def sync_vllm_adapter_snapshot(
    llm: Any,
    snapshot: ResidualAdapterSnapshot,
    *,
    version: str,
    include_bases: bool,
    pause_scheduler: bool = False,
) -> list[dict[str, Any]]:
    """Synchronously install one policy version in every vLLM worker.

    Offline ``LLM.generate`` is blocking, so the fastest correct caller invokes
    this only after generation has returned and leaves ``pause_scheduler``
    false.  Async callers must request a level-0 ``mode='wait'`` pause so no
    trajectory spans two policy versions.  Prefix/KV reuse is invalidated
    before the new version is committed.
    """

    if not isinstance(version, str) or not version.strip():
        raise ValueError("adapter version must be a non-empty string")
    paused = False
    if pause_scheduler:
        llm.sleep(level=0, mode="wait")
        paused = True
    try:
        if include_bases:
            worker_update = partial(
                apply_vllm_adapter_snapshot,
                snapshot=snapshot,
                version=version,
                include_bases=True,
            )
            expected_digest = snapshot.digest(include_bases=True)
        else:
            compact_update = snapshot.core_update()
            worker_update = partial(
                apply_vllm_core_update,
                update=compact_update,
                version=version,
            )
            expected_digest = compact_update.digest()
        receipts = llm.apply_model(worker_update)
        if not isinstance(receipts, list) or not receipts:
            raise RuntimeError("vLLM returned no adapter synchronization receipts")
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise TypeError("vLLM returned a malformed synchronization receipt")
            if receipt.get("version") != version or receipt.get("digest") != expected_digest:
                raise RuntimeError(f"vLLM adapter synchronization mismatch: {receipt!r}")
        cache_reset = llm.reset_prefix_cache()
        if cache_reset is False:
            raise RuntimeError("vLLM refused to reset its prefix cache after a policy update")
        return receipts
    finally:
        if paused:
            llm.wake_up(tags=["scheduling"])


def _selected_output_log_probs(output: Any, token_ids: tuple[int, ...]) -> tuple[float, ...]:
    logprob_rows = getattr(output, "logprobs", None)
    if not isinstance(logprob_rows, list) or len(logprob_rows) != len(token_ids):
        raise ValueError("vLLM must return one log-probability row per sampled token")
    selected: list[float] = []
    for token_id, row in zip(token_ids, logprob_rows, strict=True):
        if not isinstance(row, dict) or token_id not in row:
            raise ValueError(f"vLLM log probabilities omit sampled token {token_id}")
        value = getattr(row[token_id], "logprob", None)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("vLLM sampled-token log probability must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("vLLM sampled-token log probability must be finite")
        selected.append(numeric)
    return tuple(selected)


def _validate_completion_eos(
    candidate: Any,
    token_ids: tuple[int, ...],
    eos_token_ids: set[int],
) -> str:
    """Require vLLM's returned token sequence to preserve HF stop semantics.

    ``CompletionOutput.finish_reason`` is ``"stop"`` for both the primary EOS
    and explicit stop-token IDs in vLLM 0.22.  The primary EOS has a null
    ``stop_reason``; an explicit stop token reports its integer ID.  This
    integration configures no stop strings, so every stop must end with one of
    the requested EOS IDs.  Silently accepting an omitted EOS would change the
    response mask, behavior log-probability sum, and GRPO objective.
    """

    reason_value = getattr(candidate, "finish_reason", None)
    reason = None if reason_value is None else str(reason_value)
    stop_reason = getattr(candidate, "stop_reason", None)
    eos_positions = [index for index, token_id in enumerate(token_ids) if token_id in eos_token_ids]

    if reason is None:
        raise RuntimeError("vLLM returned a non-final completion from blocking generation")
    if reason not in {"stop", "eos", "length"}:
        raise RuntimeError(f"vLLM completion ended with unsupported reason {reason!r}")
    if len(eos_positions) > 1 or (eos_positions and eos_positions[0] != len(token_ids) - 1):
        raise RuntimeError("vLLM returned tokens after EOS; objective parity is not guaranteed")

    ended_with_eos = bool(eos_positions)
    if reason in {"stop", "eos"}:
        if not eos_token_ids:
            raise RuntimeError("vLLM stopped although no EOS token IDs were configured")
        if not ended_with_eos:
            raise RuntimeError(
                "vLLM omitted the terminal EOS token; objective parity is not guaranteed"
            )
        if isinstance(stop_reason, int) and stop_reason not in eos_token_ids:
            raise RuntimeError(
                f"vLLM stopped on unconfigured token ID {stop_reason}; "
                "objective parity is not guaranteed"
            )
        if isinstance(stop_reason, str):
            raise RuntimeError("vLLM stopped on a string despite token-only stopping configuration")
    elif ended_with_eos:
        raise RuntimeError("vLLM labeled an EOS-terminated completion as length-capped")
    elif stop_reason is not None:
        raise RuntimeError("vLLM returned a stop reason for a length-capped completion")
    return reason


def parse_vllm_grouped_outputs(
    request_outputs: Any,
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    group_size: int,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
    policy_version: str,
    device: torch.device | str = "cpu",
) -> VLLMGroupedGeneration:
    """Validate final vLLM outputs and build padded rollout tensors."""

    if not isinstance(request_outputs, list) or len(request_outputs) != len(prompt_token_ids):
        raise ValueError("vLLM returned the wrong number of prompt outputs")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")

    grouped_tokens: list[list[tuple[int, ...]]] = []
    grouped_log_probs: list[list[tuple[float, ...]]] = []
    finish_reasons: list[tuple[str | None, ...]] = []
    eos_set = set(eos_token_ids)
    maximum_length = 0
    for expected_prompt, request_output in zip(prompt_token_ids, request_outputs, strict=True):
        actual_prompt = tuple(int(token_id) for token_id in request_output.prompt_token_ids)
        if actual_prompt != expected_prompt:
            raise ValueError("vLLM changed or reordered a tokenized prompt")
        candidates = sorted(request_output.outputs, key=lambda candidate: candidate.index)
        if len(candidates) != group_size:
            raise ValueError("vLLM returned the wrong number of candidates for a prompt")
        if [candidate.index for candidate in candidates] != list(range(group_size)):
            raise ValueError("vLLM candidate indices are not contiguous")

        prompt_tokens: list[tuple[int, ...]] = []
        prompt_log_probs: list[tuple[float, ...]] = []
        prompt_finish_reasons: list[str | None] = []
        for candidate in candidates:
            tokens = tuple(int(token_id) for token_id in candidate.token_ids)
            if not tokens:
                raise ValueError("vLLM returned an empty completion")
            reason = _validate_completion_eos(candidate, tokens, eos_set)
            selected_log_probs = _selected_output_log_probs(candidate, tokens)
            maximum_length = max(maximum_length, len(tokens))
            prompt_tokens.append(tokens)
            prompt_log_probs.append(selected_log_probs)
            prompt_finish_reasons.append(reason)
        grouped_tokens.append(prompt_tokens)
        grouped_log_probs.append(prompt_log_probs)
        finish_reasons.append(tuple(prompt_finish_reasons))

    target_device = torch.device(device)
    response_ids = torch.full(
        (len(prompt_token_ids), group_size, maximum_length),
        int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    response_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    old_log_probs = torch.zeros_like(response_ids, dtype=torch.float32)
    for prompt_index, (token_group, logprob_group) in enumerate(
        zip(grouped_tokens, grouped_log_probs, strict=True)
    ):
        for group_index, (tokens, log_probs) in enumerate(
            zip(token_group, logprob_group, strict=True)
        ):
            length = len(tokens)
            response_ids[prompt_index, group_index, :length] = torch.tensor(
                tokens, dtype=torch.long, device=target_device
            )
            response_mask[prompt_index, group_index, :length] = True
            old_log_probs[prompt_index, group_index, :length] = torch.tensor(
                log_probs, dtype=torch.float32, device=target_device
            )

    return VLLMGroupedGeneration(
        prompt_token_ids=prompt_token_ids,
        response_input_ids=response_ids,
        response_mask=response_mask,
        old_token_log_probs=old_log_probs,
        finish_reasons=tuple(finish_reasons),
        policy_version=policy_version,
    )


def generate_vllm_grouped(
    llm: Any,
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
    policy_version: str,
    device: torch.device | str = "cpu",
) -> VLLMGroupedGeneration:
    """Sample unrestricted grouped completions and retain behavior log-probs."""

    if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
        raise ValueError("prompt_token_ids must contain non-empty token sequences")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be numeric")
    if float(temperature) <= 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    mode = getattr(getattr(llm, "model_config", None), "logprobs_mode", None)
    if mode != "processed_logprobs":
        raise ValueError(
            "vLLM must use logprobs_mode='processed_logprobs' so old log-probs "
            "match the temperature-scaled behavior policy"
        )

    try:
        from vllm import SamplingParams
    except ImportError as error:  # pragma: no cover - exercised on the CUDA host
        raise RuntimeError("vLLM is required for grouped generation") from error

    prompts = [{"prompt_token_ids": list(prompt)} for prompt in prompt_token_ids]
    params = [
        SamplingParams(
            n=group_size,
            max_tokens=max_new_tokens,
            temperature=float(temperature),
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            seed=seed + prompt_index,
            stop_token_ids=list(eos_token_ids),
            # vLLM otherwise merges every EOS from the repository's generation
            # config into stop_token_ids.  HF rollouts deliberately override
            # those defaults with exactly ``eos_token_ids``.  ``ignore_eos``
            # disables that implicit merge; the explicit stop-token list still
            # stops generation and is preserved in returned token IDs.
            ignore_eos=True,
            logprobs=0,
            detokenize=False,
            skip_special_tokens=False,
        )
        for prompt_index in range(len(prompts))
    ]
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return parse_vllm_grouped_outputs(
        outputs,
        prompt_token_ids,
        group_size=group_size,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
        policy_version=policy_version,
        device=device,
    )


def parse_vllm_greedy_outputs(
    request_outputs: Any,
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    eos_token_ids: tuple[int, ...],
    policy_version: str,
) -> VLLMGreedyGeneration:
    """Validate final single-candidate outputs from greedy evaluation."""

    if not isinstance(request_outputs, list) or len(request_outputs) != len(prompt_token_ids):
        raise ValueError("vLLM returned the wrong number of greedy prompt outputs")
    eos_set = set(eos_token_ids)
    responses: list[tuple[int, ...]] = []
    finish_reasons: list[str | None] = []
    for expected_prompt, request_output in zip(prompt_token_ids, request_outputs, strict=True):
        actual_prompt = tuple(int(token_id) for token_id in request_output.prompt_token_ids)
        if actual_prompt != expected_prompt:
            raise ValueError("vLLM changed or reordered a greedy tokenized prompt")
        if len(request_output.outputs) != 1 or request_output.outputs[0].index != 0:
            raise ValueError("greedy vLLM evaluation requires exactly candidate index zero")
        candidate = request_output.outputs[0]
        tokens = tuple(int(token_id) for token_id in candidate.token_ids)
        if not tokens:
            raise ValueError("vLLM returned an empty greedy completion")
        reason = _validate_completion_eos(candidate, tokens, eos_set)
        responses.append(tokens)
        finish_reasons.append(reason)
    return VLLMGreedyGeneration(
        prompt_token_ids=prompt_token_ids,
        response_token_ids=tuple(responses),
        finish_reasons=tuple(finish_reasons),
        policy_version=policy_version,
    )


def generate_vllm_greedy(
    llm: Any,
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    policy_version: str,
) -> VLLMGreedyGeneration:
    """Generate deterministic single-candidate token sequences for evaluation."""

    if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
        raise ValueError("prompt_token_ids must contain non-empty token sequences")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    try:
        from vllm import SamplingParams
    except ImportError as error:  # pragma: no cover - exercised on the CUDA host
        raise RuntimeError("vLLM is required for greedy generation") from error

    prompts = [{"prompt_token_ids": list(prompt)} for prompt in prompt_token_ids]
    params = SamplingParams(
        n=1,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        stop_token_ids=list(eos_token_ids),
        # Match the explicit HF evaluation EOS set, not additional EOS IDs in
        # the model repository's generation_config.json.
        ignore_eos=True,
        detokenize=False,
        skip_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return parse_vllm_greedy_outputs(
        outputs,
        prompt_token_ids,
        eos_token_ids=eos_token_ids,
        policy_version=policy_version,
    )


def vllm_hf_overrides(snapshot: ResidualAdapterSnapshot) -> dict[str, Any]:
    """Build the Hugging Face config overrides for the custom vLLM model."""

    unique_scales = set(snapshot.scales)
    if len(unique_scales) != 1:
        raise ValueError("the vLLM model currently requires one common adapter scale")
    return {
        "architectures": [VLLM_RESIDUAL_ARCHITECTURE],
        "residual_core_adapter_layers": list(snapshot.layer_indices),
        "residual_core_adapter_rank": snapshot.rank,
        "residual_core_adapter_scale": snapshot.scales[0],
    }


def create_vllm_engine(
    model_name: str,
    snapshot: ResidualAdapterSnapshot,
    *,
    revision: str | None,
    dtype: str,
    max_model_len: int,
    kv_cache_memory_bytes: int = 2 * 1024**3,
    enforce_eager: bool = True,
    flash_attn_version: int = 2,
    allow_insecure_serialization: bool = False,
    batch_invariant: bool = False,
    enable_v1_multiprocessing: bool = True,
    seed: int = 0,
    **engine_overrides: Any,
) -> Any:
    """Create the pinned single-GPU offline engine without loading a tokenizer."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be non-empty")
    if max_model_len < 2:
        raise ValueError("max_model_len must be at least two")
    if kv_cache_memory_bytes < 1:
        raise ValueError("kv_cache_memory_bytes must be positive")
    if (
        isinstance(flash_attn_version, bool)
        or not isinstance(flash_attn_version, int)
        or flash_attn_version not in {2, 3}
    ):
        raise ValueError("flash_attn_version must be 2 or 3")
    if not isinstance(batch_invariant, bool):
        raise TypeError("batch_invariant must be boolean")
    if not isinstance(enable_v1_multiprocessing, bool):
        raise TypeError("enable_v1_multiprocessing must be boolean")
    if batch_invariant:
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM batch invariance requires an NVIDIA CUDA GPU")
        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            raise RuntimeError(
                "vLLM 0.22 batch invariance requires compute capability 9.0 or newer; "
                f"found {capability[0]}.{capability[1]}"
            )
    # Process mode and the related environment controls must be fixed before
    # plugin discovery imports vLLM.  In multiprocess mode, apply_model sends a
    # Python callable over local IPC and therefore requires the explicit
    # insecure-serialization opt-in.  The in-process client calls it directly.
    configure_vllm_v1_multiprocessing(enabled=enable_v1_multiprocessing)
    configure_vllm_batch_invariance(enabled=batch_invariant)
    configure_trusted_vllm_callable_serialization(
        enabled=allow_insecure_serialization,
    )
    if enable_v1_multiprocessing and not allow_insecure_serialization:
        raise RuntimeError(
            "the multiprocess mutable residual-core vLLM backend requires explicit "
            "trusted-local callable serialization; set "
            "vllm_allow_insecure_serialization=true only when EngineCore and workers "
            "are local trusted processes, or disable V1 multiprocessing"
        )
    reserved = {
        "attention_config",
        "dtype",
        "enforce_eager",
        "hf_overrides",
        "kv_cache_memory_bytes",
        "logprobs_mode",
        "max_model_len",
        "model",
        "pipeline_parallel_size",
        "revision",
        "seed",
        "skip_tokenizer_init",
        "tensor_parallel_size",
    }
    collisions = sorted(reserved.intersection(engine_overrides))
    if collisions:
        raise ValueError(f"engine_overrides cannot replace protected settings: {collisions}")

    # The serialization environment is now fixed and inherited by spawned
    # children.  Only after that boundary may any code import vLLM.
    # The direct call makes parent-process config inspection deterministic.
    # Installed general-plugin metadata performs the same idempotent
    # registration in a spawned EngineCore and any worker processes.
    ensure_vllm_plugin_discoverable()
    register_residual_qwen_model()
    try:
        from vllm import LLM
    except ImportError as error:  # pragma: no cover - registration already checks this
        raise RuntimeError("vLLM is required to create the rollout engine") from error
    return LLM(
        model=model_name,
        revision=revision,
        dtype=dtype,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        skip_tokenizer_init=True,
        hf_overrides=vllm_hf_overrides(snapshot),
        attention_config={"flash_attn_version": flash_attn_version},
        max_model_len=max_model_len,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        enforce_eager=enforce_eager,
        logprobs_mode="processed_logprobs",
        seed=seed,
        **engine_overrides,
    )


class OnPolicyVLLMGenerator:
    """Stateful guard that never labels generations with an unsynced policy."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.policy_version: str | None = None
        self.state_digest: str | None = None
        self._bases_synced = False
        self._fixed_state_digest: str | None = None

    def sync(
        self,
        snapshot: ResidualAdapterSnapshot,
        *,
        version: str,
        include_bases: bool = False,
        pause_scheduler: bool = False,
    ) -> list[dict[str, Any]]:
        if not self._bases_synced and not include_bases:
            raise RuntimeError("the first vLLM synchronization must include PCA bases")
        fixed_digest = snapshot.fixed_digest()
        if (
            self._fixed_state_digest is not None
            and fixed_digest != self._fixed_state_digest
            and not include_bases
        ):
            raise RuntimeError(
                "the HF adapter's fixed bases/layout changed; a core-only vLLM update is unsafe"
            )
        receipts = sync_vllm_adapter_snapshot(
            self.llm,
            snapshot,
            version=version,
            include_bases=include_bases,
            pause_scheduler=pause_scheduler,
        )
        self._bases_synced = True
        self._fixed_state_digest = fixed_digest
        self.policy_version = version
        # Policy identity always covers bases and cores.  Worker receipts use a
        # compact core-only digest after initialization, but exposing that as a
        # state identity would make equal policies hash differently by sync mode.
        self.state_digest = snapshot.digest(include_bases=True)
        return receipts

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        **sampling_kwargs: Any,
    ) -> VLLMGroupedGeneration:
        if self.policy_version is None:
            raise RuntimeError("install an adapter snapshot before vLLM generation")
        return generate_vllm_grouped(
            self.llm,
            prompt_token_ids,
            policy_version=self.policy_version,
            **sampling_kwargs,
        )

    def generate_greedy(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
    ) -> VLLMGreedyGeneration:
        if self.policy_version is None:
            raise RuntimeError("install an adapter snapshot before vLLM generation")
        return generate_vllm_greedy(
            self.llm,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            policy_version=self.policy_version,
        )


@torch.inference_mode()
def probe_vllm_repeat_determinism(
    rollout_policy: OnPolicyVLLMGenerator,
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
    device: torch.device | str = "cpu",
) -> VLLMRepeatDeterminismReport:
    """Generate twice without an update and compare every sampled tensor.

    This is an explicit diagnostic rather than a training-path assertion: it
    spends two rollout batches and advances vLLM's internal request counter,
    but it neither updates nor resynchronizes the policy.  A common seed only
    predicts equal output while the policy and prompt batch are also equal;
    normal method-specific optimizer updates invalidate that comparison.

    CUDA graph replay should not intentionally alter the per-request sampling
    seed.  If the first optimizer-step rollout differs across matched methods,
    this probe distinguishes engine-level repeatability from stale policy
    synchronization, prompt-order differences, or method updates.
    """

    policy_version = rollout_policy.policy_version
    policy_state_digest = rollout_policy.state_digest
    if policy_version is None or policy_state_digest is None:
        raise RuntimeError("repeat determinism probing requires a synchronized vLLM policy")
    generation_kwargs = {
        "group_size": group_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "pad_token_id": pad_token_id,
        "eos_token_ids": eos_token_ids,
        "device": device,
    }
    first = rollout_policy.generate(prompt_token_ids, **generation_kwargs)
    second = rollout_policy.generate(prompt_token_ids, **generation_kwargs)
    if (
        first.policy_version != policy_version
        or second.policy_version != policy_version
        or rollout_policy.policy_version != policy_version
        or rollout_policy.state_digest != policy_state_digest
    ):
        raise RuntimeError("vLLM policy identity changed during repeat determinism probing")

    first_ids = first.response_input_ids.detach().cpu()
    second_ids = second.response_input_ids.detach().cpu()
    first_mask = first.response_mask.detach().cpu()
    second_mask = second.response_mask.detach().cpu()
    first_logprobs = first.old_token_log_probs.detach().float().cpu()
    second_logprobs = second.old_token_log_probs.detach().float().cpu()
    ids_equal = torch.equal(first_ids, second_ids)
    masks_equal = torch.equal(first_mask, second_mask)
    logprobs_equal = torch.equal(first_logprobs, second_logprobs)
    if first_logprobs.shape == second_logprobs.shape:
        maximum_delta = float((first_logprobs - second_logprobs).abs().max().item())
    else:
        maximum_delta = None
    return VLLMRepeatDeterminismReport(
        policy_version=policy_version,
        policy_state_digest=policy_state_digest,
        seed=seed,
        response_input_ids_equal=ids_equal,
        response_masks_equal=masks_equal,
        behavior_logprobs_bitwise_equal=logprobs_equal,
        maximum_behavior_logprob_abs_delta=maximum_delta,
        first_valid_response_tokens=first.valid_response_tokens,
        second_valid_response_tokens=second.valid_response_tokens,
    )


def register_vllm_residual_qwen_model() -> None:
    """Backward-compatible direct registration for callers outside the runner."""

    register_residual_qwen_model()


__all__ = [
    "VLLM_RESIDUAL_ARCHITECTURE",
    "OnPolicyVLLMGenerator",
    "ResidualAdapterCoreLayerState",
    "ResidualAdapterCoreUpdate",
    "ResidualAdapterLayerState",
    "ResidualAdapterSnapshot",
    "VLLMGreedyGeneration",
    "VLLMGroupedGeneration",
    "VLLMRepeatDeterminismReport",
    "VLLMResidualCoreAdapter",
    "apply_adapter_to_vllm_split_state",
    "apply_vllm_adapter_snapshot",
    "apply_vllm_core_update",
    "capture_residual_adapter_snapshot",
    "create_vllm_engine",
    "generate_vllm_greedy",
    "generate_vllm_grouped",
    "parse_vllm_greedy_outputs",
    "parse_vllm_grouped_outputs",
    "probe_vllm_repeat_determinism",
    "register_vllm_residual_qwen_model",
    "sync_vllm_adapter_snapshot",
    "vllm_hf_overrides",
]
