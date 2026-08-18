"""Exact frozen-prefix caching for Qwen teacher-forced sequence scoring.

The residual adapters occupy a contiguous suffix of decoder blocks.  For a
fixed rollout, everything before the first wrapped block is therefore frozen
and independent of the policy parameters.  This module captures the exact
hidden state and keyword arguments entering that block once, then replays only
the adapted suffix, final norm, and language-model head.

No Transformers mask or rotary implementation is duplicated here.  A
temporary forward pre-hook observes the arguments prepared by the installed
Qwen implementation, then raises a private sentinel to stop before any adapted
block executes.  Structural guards make unsupported layouts a clean opt-in
fallback instead of a silently approximate scorer.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import AdapterWrappedLayer, ModelBundle, _decoder_layers, model_forward_is_compiled


class FrozenPrefixUnsupportedError(RuntimeError):
    """Raised when a model cannot be replayed exactly by this fast path."""


class _PrefixCaptured(BaseException):
    """Private non-Exception sentinel used to stop the model at the boundary."""


@dataclass(frozen=True, slots=True)
class FrozenPrefixStructure:
    """Validated references required to replay the adapted Qwen suffix."""

    model: nn.Module
    suffix_layers: tuple[AdapterWrappedLayer, ...]
    final_norm: nn.Module
    lm_head: nn.Module
    first_adapted_layer: int
    total_layers: int
    adapter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenPrefixCache:
    """Adapter-independent inputs captured at the first adapted block."""

    structure: FrozenPrefixStructure
    hidden_states: Tensor
    layer_kwargs: Mapping[str, Any]
    batch_size: int
    sequence_length: int
    full_prefix_calls: int = 1


@dataclass(frozen=True, slots=True)
class FrozenPrefixBuildResult:
    cache: FrozenPrefixCache | None
    fallback_reason: str | None


def _unsupported(message: str) -> tuple[None, str]:
    return None, message


def resolve_frozen_prefix_structure(
    bundle: ModelBundle,
) -> tuple[FrozenPrefixStructure | None, str | None]:
    """Resolve the exact supported Qwen layout without running the model."""

    if model_forward_is_compiled(bundle.model):
        return _unsupported("compiled model.forward is incompatible with boundary capture")
    if bundle.model.training:
        return _unsupported("frozen-prefix scoring requires model.eval()")
    config = getattr(bundle.model, "config", None)
    if getattr(config, "model_type", None) != "qwen2":
        return _unsupported("frozen-prefix scoring currently supports Qwen2/Qwen2.5 only")

    try:
        layers = _decoder_layers(bundle.model)
    except TypeError as error:
        return _unsupported(str(error))
    wrapped_indices = [
        index for index, layer in enumerate(layers) if isinstance(layer, AdapterWrappedLayer)
    ]
    if not wrapped_indices:
        return _unsupported("no AdapterWrappedLayer suffix was found")
    first = wrapped_indices[0]
    expected_indices = list(range(first, len(layers)))
    if wrapped_indices != expected_indices:
        return _unsupported("adapters must wrap one contiguous decoder suffix")
    if first == 0:
        return _unsupported("the model has no frozen decoder prefix to cache")

    suffix_layers = tuple(layers[index] for index in expected_indices)
    if len(suffix_layers) != len(bundle.adapter_names):
        return _unsupported("adapter names do not match the wrapped decoder suffix")
    suffix_core_ids = {id(layer.adapter.core) for layer in suffix_layers}
    try:
        bundle_parameter_ids = {id(parameter) for parameter in bundle.trainable_parameters}
    except KeyError:
        return _unsupported("ModelBundle adapter names do not resolve in the live model")
    if suffix_core_ids != bundle_parameter_ids:
        return _unsupported("trainable parameters are not exactly the suffix adapter cores")
    live_gradient_parameter_ids = {
        id(parameter) for parameter in bundle.model.parameters() if parameter.requires_grad
    }
    if not live_gradient_parameter_ids.issubset(suffix_core_ids):
        return _unsupported("only suffix adapter cores may require gradients")
    if any(parameter.requires_grad for layer in layers[:first] for parameter in layer.parameters()):
        return _unsupported("a parameter in the proposed frozen prefix requires gradients")

    backbone = getattr(bundle.model, "model", None)
    if backbone is None or getattr(backbone, "layers", None) is not layers:
        return _unsupported("unsupported Qwen causal-LM/backbone nesting")
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(bundle.model, "lm_head", None)
    if not isinstance(final_norm, nn.Module) or not isinstance(lm_head, nn.Module):
        return _unsupported("Qwen final norm or lm_head could not be resolved")

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) < len(layers):
            return _unsupported("config.layer_types is shorter than the decoder stack")
        suffix_types = {str(layer_types[index]) for index in expected_indices}
        if len(suffix_types) != 1:
            return _unsupported("adapted suffix mixes attention-mask layer types")

    return (
        FrozenPrefixStructure(
            model=bundle.model,
            suffix_layers=suffix_layers,
            final_norm=final_norm,
            lm_head=lm_head,
            first_adapted_layer=first,
            total_layers=len(layers),
            adapter_names=tuple(bundle.adapter_names),
        ),
        None,
    )


def _detach_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


@torch.no_grad()
def build_frozen_prefix_cache(
    bundle: ModelBundle,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    structure: FrozenPrefixStructure | None = None,
) -> FrozenPrefixCache:
    """Execute Qwen through its frozen prefix and capture the suffix inputs."""

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [B, T] shapes")
    model_device = bundle.trainable_parameters[0].device
    if input_ids.device != model_device or attention_mask.device != model_device:
        raise ValueError("prefix-cache inputs and model bundle must be on the same device")
    if structure is None:
        structure, reason = resolve_frozen_prefix_structure(bundle)
        if structure is None:
            raise FrozenPrefixUnsupportedError(reason or "unsupported frozen-prefix structure")

    captured: dict[str, Any] = {}

    def capture_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) == 1:
            hidden_states = args[0]
        elif not args and "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            kwargs = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        else:
            raise FrozenPrefixUnsupportedError(
                "first adapted layer received an unsupported positional signature"
            )
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
            raise FrozenPrefixUnsupportedError(
                "first adapted layer did not receive [B, T, H] hidden states"
            )
        if kwargs.get("past_key_values") is not None or kwargs.get("past_key_value") is not None:
            raise FrozenPrefixUnsupportedError("teacher-forced prefix capture requires no KV cache")
        if kwargs.get("use_cache") not in {None, False}:
            raise FrozenPrefixUnsupportedError("teacher-forced prefix capture requires use_cache=False")
        captured["hidden_states"] = hidden_states.detach()
        captured["layer_kwargs"] = _detach_tree(dict(kwargs))
        raise _PrefixCaptured

    handle = structure.suffix_layers[0].register_forward_pre_hook(
        capture_hook,
        prepend=True,
        with_kwargs=True,
    )
    try:
        try:
            structure.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except _PrefixCaptured:
            pass
    finally:
        handle.remove()
    if "hidden_states" not in captured or "layer_kwargs" not in captured:
        raise FrozenPrefixUnsupportedError("Qwen did not execute the first adapted layer")

    hidden_states = captured["hidden_states"]
    if hidden_states.shape[:2] != input_ids.shape:
        raise FrozenPrefixUnsupportedError("captured prefix hidden-state shape is incompatible")
    return FrozenPrefixCache(
        structure=structure,
        hidden_states=hidden_states,
        layer_kwargs=captured["layer_kwargs"],
        batch_size=input_ids.shape[0],
        sequence_length=input_ids.shape[1],
    )


def maybe_build_frozen_prefix_cache(
    bundle: ModelBundle,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> FrozenPrefixBuildResult:
    """Build a cache when structurally supported, otherwise report fallback."""

    structure, reason = resolve_frozen_prefix_structure(bundle)
    if structure is None:
        return FrozenPrefixBuildResult(cache=None, fallback_reason=reason)
    try:
        cache = build_frozen_prefix_cache(
            bundle,
            input_ids,
            attention_mask,
            structure=structure,
        )
    except FrozenPrefixUnsupportedError as error:
        return FrozenPrefixBuildResult(cache=None, fallback_reason=str(error))
    return FrozenPrefixBuildResult(cache=cache, fallback_reason=None)


def _slice_batch_tree(value: Any, start: int, stop: int, full_batch_size: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim >= 2 and value.shape[0] == full_batch_size:
            return value[start:stop]
        return value
    if isinstance(value, tuple):
        return tuple(_slice_batch_tree(item, start, stop, full_batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_tree(item, start, stop, full_batch_size) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _slice_batch_tree(item, start, stop, full_batch_size)
            for key, item in value.items()
        }
    return value


def _repeat_batch_tree(value: Any, repeats: int, selected_batch_size: int) -> Any:
    if repeats == 1:
        return value
    if isinstance(value, Tensor):
        if value.ndim >= 2 and value.shape[0] == selected_batch_size:
            return value.repeat((repeats,) + (1,) * (value.ndim - 1))
        return value
    if isinstance(value, tuple):
        return tuple(_repeat_batch_tree(item, repeats, selected_batch_size) for item in value)
    if isinstance(value, list):
        return [_repeat_batch_tree(item, repeats, selected_batch_size) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _repeat_batch_tree(item, repeats, selected_batch_size)
            for key, item in value.items()
        }
    return value


def replay_frozen_suffix_logits(
    bundle: ModelBundle,
    cache: FrozenPrefixCache,
    *,
    start: int = 0,
    stop: int | None = None,
    repeats: int = 1,
    logits_to_keep: Tensor | slice | None = None,
) -> Tensor:
    """Replay cached rows through adapted suffix, final norm, and LM head."""

    if cache.structure.model is not bundle.model:
        raise ValueError("frozen-prefix cache belongs to a different model instance")
    if tuple(bundle.adapter_names) != cache.structure.adapter_names:
        raise ValueError("adapter parameter layout changed after prefix capture")
    if stop is None:
        stop = cache.batch_size
    if not 0 <= start < stop <= cache.batch_size:
        raise ValueError("cached suffix row range is invalid")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    selected_batch_size = stop - start
    hidden_states = cache.hidden_states[start:stop]
    layer_kwargs = _slice_batch_tree(
        cache.layer_kwargs,
        start,
        stop,
        cache.batch_size,
    )
    if repeats > 1:
        hidden_states = hidden_states.repeat((repeats, 1, 1))
        layer_kwargs = _repeat_batch_tree(layer_kwargs, repeats, selected_batch_size)

    for layer in cache.structure.suffix_layers:
        output = layer(hidden_states, **layer_kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output
    hidden_states = cache.structure.final_norm(hidden_states)
    if logits_to_keep is not None:
        hidden_states = hidden_states[:, logits_to_keep, :]
    logits = cache.structure.lm_head(hidden_states)
    if logits.ndim != 3 or logits.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("cached Qwen suffix returned incompatible logits")
    return logits


def qwen_teacher_forcing_logits_to_keep(
    bundle: ModelBundle,
    prompt_width: int,
    response_width: int,
) -> Tensor | None:
    """Return exact response-prediction positions for Qwen 5.x projection.

    Transformers 5.x accepts a tensor-valued ``logits_to_keep`` and applies
    the LM head only at these sequence positions.  Older Qwen and generic toy
    models fall back to the full logits path.
    """

    if getattr(getattr(bundle.model, "config", None), "model_type", None) != "qwen2":
        return None
    try:
        parameters = inspect.signature(bundle.model.forward).parameters
    except (TypeError, ValueError):
        return None
    if "logits_to_keep" not in parameters:
        return None
    return torch.arange(
        prompt_width - 1,
        prompt_width + response_width - 1,
        device=bundle.device,
        dtype=torch.long,
    )


def _selected_token_log_probs(logits: Tensor, targets: Tensor, temperature: float) -> Tensor:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    work = logits.float() / float(temperature)
    selected = work.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(work, dim=-1)


def cached_prefix_flat_token_log_probs(
    bundle: ModelBundle,
    cache: FrozenPrefixCache,
    response_input_ids: Tensor,
    response_mask: Tensor,
    *,
    prompt_width: int,
    temperature: float,
    micro_batch_size: int | None,
) -> Tensor:
    """Score flat fixed responses by replaying only the cached Qwen suffix."""

    if response_input_ids.ndim != 2 or response_mask.shape != response_input_ids.shape:
        raise ValueError("response tensors must have matching [B, T] shapes")
    total_examples, response_width = response_input_ids.shape
    if cache.batch_size != total_examples:
        raise ValueError("frozen-prefix cache batch size does not match the rollout")
    if cache.sequence_length < prompt_width + response_width:
        raise ValueError("frozen-prefix cache sequence length does not match the rollout")
    if micro_batch_size is None:
        micro_batch_size = total_examples
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size < 1
    ):
        raise ValueError("micro_batch_size must be a positive integer")

    chunks: list[Tensor] = []
    logits_to_keep = qwen_teacher_forcing_logits_to_keep(
        bundle,
        prompt_width,
        response_width,
    )
    for start in range(0, total_examples, micro_batch_size):
        stop = min(start + micro_batch_size, total_examples)
        logits = replay_frozen_suffix_logits(
            bundle,
            cache,
            start=start,
            stop=stop,
            logits_to_keep=logits_to_keep,
        )
        response_logits = (
            logits
            if logits_to_keep is not None
            else logits[:, prompt_width - 1 : prompt_width + response_width - 1, :]
        )
        token_log_probs = _selected_token_log_probs(
            response_logits,
            response_input_ids[start:stop],
            temperature,
        )
        chunks.append(token_log_probs.masked_fill(~response_mask[start:stop], 0.0))
    return torch.cat(chunks, dim=0)


__all__ = [
    "FrozenPrefixBuildResult",
    "FrozenPrefixCache",
    "FrozenPrefixStructure",
    "FrozenPrefixUnsupportedError",
    "build_frozen_prefix_cache",
    "cached_prefix_flat_token_log_probs",
    "maybe_build_frozen_prefix_cache",
    "qwen_teacher_forcing_logits_to_keep",
    "replay_frozen_suffix_logits",
    "resolve_frozen_prefix_structure",
]
