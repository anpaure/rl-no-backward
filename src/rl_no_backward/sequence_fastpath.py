"""Opt-in inference fast paths for fixed-rollout directional rescoring.

The default optimizer deliberately uses a simple, one-policy-at-a-time
implementation.  This module contains the production vectorized alternative,
enabled only by explicit configuration.  It never generates trajectories: it
only rescores the same stored token sequences under several adapter
perturbations.

The frozen transformer sees an ordinary larger batch.  A
``BatchedProbeResidualCoreAdapter`` selects a different tiny adapter core for
each batch row, allowing several positive/negative finite-difference policies
to share one model invocation.  Base-model weights are neither copied nor
stacked.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .frozen_prefix import qwen_teacher_forcing_logits_to_keep, replay_frozen_suffix_logits
from .model import ModelBundle, ResidualCoreAdapter
from .sequence_policy import SequenceRolloutBatch


def selected_token_log_probs(logits: Tensor, targets: Tensor, temperature: float) -> Tensor:
    """Return selected log probabilities with FP32 normalization.

    Keeping this reduction as a small pure function gives ``torch.compile`` a
    useful fusion boundary: on CUDA, the cast, temperature scaling, gather, and
    log-sum-exp can be fused without changing the model's logits or sampling
    semantics.  Compilation is intentionally left to the caller so importing
    this module never triggers a compiler or GPU initialization.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match every non-vocabulary logits dimension")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    work = logits.float() / float(temperature)
    selected = work.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(work, dim=-1)


class BatchedProbeResidualCoreAdapter(ResidualCoreAdapter):
    """Residual core adapter with an inference-only per-example probe mode.

    In ordinary operation this is exactly ``ResidualCoreAdapter``.  Inside
    ``use_probe_cores``, ``cores[b]`` is applied to hidden-state batch row
    ``b``.  The context is deliberately local and restores prior state even if
    the model call raises.
    """

    _probe_cores: Tensor | None

    def __init__(self, p_basis: Tensor, q_basis: Tensor, scale: float = 1.0) -> None:
        super().__init__(p_basis, q_basis, scale)
        self._probe_cores = None

    @contextmanager
    def use_probe_cores(self, cores: Tensor):
        if cores.ndim != 3 or cores.shape[1:] != self.core.shape:
            raise ValueError(
                "probe cores must have shape [batch, rank, rank] matching the adapter"
            )
        if cores.device != self.core.device:
            raise ValueError("probe cores and adapter must be on the same device")
        previous = self._probe_cores
        self._probe_cores = cores
        try:
            yield self
        finally:
            self._probe_cores = previous

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self._probe_cores is None:
            return super().forward(hidden_states)
        if hidden_states.ndim != 3:
            raise ValueError("batched probe mode expects hidden states with shape [B, T, H]")
        if hidden_states.shape[0] != self._probe_cores.shape[0]:
            raise ValueError("one probe core is required for every hidden-state batch row")

        original_dtype = hidden_states.dtype
        normalized = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],))
        coordinates = normalized @ self.q_basis
        # The ordinary adapter computes coordinates @ core.T.  Here each batch
        # row has its own core while sequence positions share that row's core.
        output_coordinates = torch.einsum(
            "bti,boi->bto", coordinates, self._probe_cores.float()
        )
        delta = output_coordinates @ self.p_basis.T
        return hidden_states + (self.scale * delta).to(original_dtype)


@torch.no_grad()
def enable_batched_probe_adapters(bundle: ModelBundle) -> int:
    """Replace residual-core adapters with probe-capable equivalents in place.

    The ordinary forward path, parameter names, parameter values, and
    ``requires_grad`` state are preserved.  Keeping this conversion explicit
    lets the default model-loading path remain completely unchanged while an
    optimized runner can opt in after activation calibration.

    Returns the number of adapters converted.  Calling the function again is
    idempotent.
    """

    expected_parameter_count = bundle.parameter_count
    modules = dict(bundle.model.named_modules())
    replacements: list[tuple[torch.nn.Module, str, BatchedProbeResidualCoreAdapter]] = []
    for parameter_name in bundle.adapter_names:
        suffix = ".core"
        if not parameter_name.endswith(suffix):
            raise TypeError(f"unsupported trainable adapter parameter {parameter_name!r}")
        module_name = parameter_name[: -len(suffix)]
        adapter = modules.get(module_name)
        if isinstance(adapter, BatchedProbeResidualCoreAdapter):
            continue
        if not isinstance(adapter, ResidualCoreAdapter):
            raise TypeError(f"{module_name!r} must be a ResidualCoreAdapter")

        parent_name, separator, attribute_name = module_name.rpartition(".")
        parent = bundle.model.get_submodule(parent_name) if separator else bundle.model
        replacement = BatchedProbeResidualCoreAdapter(
            adapter.p_basis,
            adapter.q_basis,
            scale=adapter.scale,
        ).to(device=adapter.core.device)
        replacement.core.copy_(adapter.core)
        replacement.core.requires_grad_(adapter.core.requires_grad)
        replacement.train(adapter.training)
        replacements.append((parent, attribute_name or module_name, replacement))

    for parent, attribute_name, replacement in replacements:
        setattr(parent, attribute_name, replacement)

    # ModelBundle resolves trainable parameters by name on every access.  Check
    # that replacement did not alter that public layout.
    if bundle.parameter_count != expected_parameter_count:
        raise RuntimeError("batched-probe conversion changed the adapter parameter layout")
    return len(replacements)


@dataclass(frozen=True, slots=True)
class FusedProbeConfig:
    """Memory controls for vectorized finite-difference rescoring."""

    directions_per_forward: int = 2
    examples_per_forward: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("directions_per_forward", self.directions_per_forward),
            ("examples_per_forward", self.examples_per_forward),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def maximum_model_batch(self) -> int:
        """Maximum rows in a model call (both signs are included)."""

        return 2 * self.directions_per_forward * self.examples_per_forward


@dataclass(frozen=True, slots=True)
class FusedProbeResult:
    positive_token_log_probs: Tensor
    negative_token_log_probs: Tensor
    model_calls: int
    full_prefix_calls: int
    suffix_calls: int


@dataclass(frozen=True, slots=True)
class _AdapterSlice:
    adapter: BatchedProbeResidualCoreAdapter
    start: int
    stop: int


def _adapter_slices(bundle: ModelBundle) -> tuple[_AdapterSlice, ...]:
    modules = dict(bundle.model.named_modules())
    offset = 0
    result: list[_AdapterSlice] = []
    for parameter_name in bundle.adapter_names:
        suffix = ".core"
        if not parameter_name.endswith(suffix):
            raise TypeError(f"unsupported trainable adapter parameter {parameter_name!r}")
        module_name = parameter_name[: -len(suffix)]
        adapter = modules.get(module_name)
        if not isinstance(adapter, BatchedProbeResidualCoreAdapter):
            raise TypeError(
                f"{module_name!r} must be a BatchedProbeResidualCoreAdapter for fused probes"
            )
        size = adapter.core.numel()
        result.append(_AdapterSlice(adapter=adapter, start=offset, stop=offset + size))
        offset += size
    if offset != bundle.parameter_count:
        raise ValueError("adapter layout does not match the bundle's trainable parameter vector")
    return tuple(result)


@contextmanager
def _probe_core_context(
    layout: tuple[_AdapterSlice, ...],
    probe_vectors: Tensor,
    examples_per_probe: int,
):
    with ExitStack() as stack:
        for item in layout:
            cores = probe_vectors[:, item.start : item.stop].view(
                probe_vectors.shape[0], *item.adapter.core.shape
            )
            per_row_cores = cores.repeat_interleave(examples_per_probe, dim=0)
            stack.enter_context(item.adapter.use_probe_cores(per_row_cores))
        yield


@torch.inference_mode()
def fused_directional_token_log_probs(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    center: Tensor,
    basis: Tensor,
    radius: float,
    config: FusedProbeConfig | None = None,
) -> FusedProbeResult:
    """Rescore fixed responses under several paired perturbations per call.

    Results have shape ``[B, G, T, D]``, matching the production central-
    difference statistics function.  Direction chunks contain both signs, and
    example chunks are repeated perturbation-major.  Only adapter cores differ
    between repeated rows; prompts, response tokens, masks, and temperature are
    exactly shared.
    """

    if config is None:
        config = FusedProbeConfig()
    if basis.ndim != 2 or basis.shape[0] != center.numel() or basis.shape[1] < 1:
        raise ValueError("basis must have shape [parameter_count, directions>=1]")
    if center.numel() != bundle.parameter_count:
        raise ValueError("center size does not match the bundle's adapter parameters")
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    model_device = bundle.trainable_parameters[0].device
    if center.device != model_device or basis.device != model_device:
        raise ValueError("center, basis, and model bundle must be on the same device")

    layout = _adapter_slices(bundle)
    flat_input_ids = rollout.flat_input_ids
    flat_attention_mask = rollout.flat_attention_mask
    flat_targets = rollout.flat_response_input_ids
    flat_response_mask = rollout.flat_response_mask
    prompt_width = rollout.prompt_input_ids.shape[1]
    response_width = rollout.max_response_length
    example_count = rollout.environment_samples
    direction_count = basis.shape[1]
    logits_to_keep = qwen_teacher_forcing_logits_to_keep(
        bundle,
        prompt_width,
        response_width,
    )

    positive = torch.empty(
        example_count,
        response_width,
        direction_count,
        device=bundle.device,
        dtype=torch.float32,
    )
    negative = torch.empty_like(positive)
    model_calls = 0

    for direction_start in range(0, direction_count, config.directions_per_forward):
        direction_stop = min(
            direction_start + config.directions_per_forward, direction_count
        )
        direction_chunk = basis[:, direction_start:direction_stop].T
        positive_vectors = center.unsqueeze(0) + radius * direction_chunk
        negative_vectors = center.unsqueeze(0) - radius * direction_chunk
        probe_vectors = torch.cat([positive_vectors, negative_vectors], dim=0)
        chunk_directions = direction_stop - direction_start
        probe_count = 2 * chunk_directions

        for example_start in range(0, example_count, config.examples_per_forward):
            example_stop = min(
                example_start + config.examples_per_forward, example_count
            )
            examples_in_chunk = example_stop - example_start
            expanded_targets = flat_targets[example_start:example_stop].repeat(
                probe_count, 1
            )
            expanded_response_mask = flat_response_mask[
                example_start:example_stop
            ].repeat(probe_count, 1)

            with _probe_core_context(layout, probe_vectors, examples_in_chunk):
                if rollout.frozen_prefix_cache is None:
                    expanded_input_ids = flat_input_ids[
                        example_start:example_stop
                    ].repeat(probe_count, 1)
                    expanded_attention_mask = flat_attention_mask[
                        example_start:example_stop
                    ].repeat(probe_count, 1)
                    model_kwargs: dict[str, object] = {
                        "input_ids": expanded_input_ids,
                        "attention_mask": expanded_attention_mask,
                        "use_cache": False,
                    }
                    if logits_to_keep is not None:
                        model_kwargs["logits_to_keep"] = logits_to_keep
                    outputs = bundle.model(**model_kwargs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                else:
                    logits = replay_frozen_suffix_logits(
                        bundle,
                        rollout.frozen_prefix_cache,
                        start=example_start,
                        stop=example_stop,
                        repeats=probe_count,
                        logits_to_keep=logits_to_keep,
                    )
            response_logits = (
                logits
                if logits_to_keep is not None
                else logits[:, prompt_width - 1 : prompt_width + response_width - 1, :]
            )
            token_log_probs = selected_token_log_probs(
                response_logits,
                expanded_targets,
                rollout.sampling_temperature,
            ).masked_fill(~expanded_response_mask, 0.0)
            token_log_probs = token_log_probs.view(
                probe_count, examples_in_chunk, response_width
            )

            plus = token_log_probs[:chunk_directions].permute(1, 2, 0)
            minus = token_log_probs[chunk_directions:].permute(1, 2, 0)
            positive[
                example_start:example_stop, :, direction_start:direction_stop
            ] = plus
            negative[
                example_start:example_stop, :, direction_start:direction_stop
            ] = minus
            model_calls += 1

    output_shape = (
        rollout.batch_size,
        rollout.group_size,
        response_width,
        direction_count,
    )
    return FusedProbeResult(
        positive_token_log_probs=positive.view(output_shape),
        negative_token_log_probs=negative.view(output_shape),
        model_calls=model_calls,
        full_prefix_calls=model_calls if rollout.frozen_prefix_cache is None else 0,
        # Full-model calls execute both stages; cached calls execute suffix only.
        suffix_calls=model_calls,
    )


__all__ = [
    "BatchedProbeResidualCoreAdapter",
    "FusedProbeConfig",
    "FusedProbeResult",
    "enable_batched_probe_adapters",
    "fused_directional_token_log_probs",
    "selected_token_log_probs",
]
