"""Non-updating projected-gradient oracle for the all-layer matched LoRA policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .common import make_search_basis
from .matched_grpo_objective import MatchedGRPOObjectiveConfig, matched_token_grpo_surrogate
from .matched_lora_backprop import matched_grpo_streaming_backward
from .matched_lora_forward_only import MatchedForwardConfig
from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled, set_parameter_vector
from .sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs


@dataclass(frozen=True, slots=True)
class ProjectedGradientGateReport:
    directions: int
    finite_difference_mu: float
    cosine_similarity: float
    relative_l2_error: float
    exact_projected_norm: float
    finite_difference_norm: float
    parameter_digest_before: str
    parameter_digest_after: str
    parameter_integrity: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector_digest(vector: Tensor) -> str:
    value = vector.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def fixed_rollout_projected_gradient_gate(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    objective_config: MatchedGRPOObjectiveConfig,
    forward_config: MatchedForwardConfig,
    generator: torch.Generator,
    *,
    prompt_groups_per_micro_batch: int = 1,
) -> ProjectedGradientGateReport:
    """Compare q-direction central differences with ``basis.T @ exact_gradient``.

    This diagnostic performs reverse mode only in its dedicated oracle process,
    never in a forward-only trial.  It restores the byte-equivalent parameter
    vector and original ``requires_grad`` flags before returning.
    """

    center = parameter_vector(bundle).float().clone()
    before_digest = _vector_digest(center)
    original_requires_grad = [parameter.requires_grad for parameter in bundle.trainable_parameters]
    exact_objective_gradient: Tensor | None = None
    basis: Tensor | None = None
    finite_difference_coordinates: list[Tensor] = []
    try:
        set_adapter_grad_enabled(bundle, True)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        matched_grpo_streaming_backward(
            bundle,
            rollout,
            sampler_token_log_probs,
            objective_config,
            prompt_groups_per_micro_batch=prompt_groups_per_micro_batch,
        )
        loss_gradient = torch.cat(
            [parameter.grad.detach().float().reshape(-1) for parameter in bundle.trainable_parameters]
        )
        exact_objective_gradient = -loss_gradient
        if not torch.isfinite(exact_objective_gradient).all() or exact_objective_gradient.norm() == 0:
            raise RuntimeError("exact matched-GRPO gradient is non-finite or zero")
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        set_adapter_grad_enabled(bundle, False)
        basis = make_search_basis(
            center.numel(),
            forward_config.directions,
            generator,
            bundle.device,
        )
        with torch.inference_mode():
            for direction_index in range(forward_config.directions):
                direction = basis[:, direction_index]
                set_parameter_vector(
                    bundle,
                    center + forward_config.finite_difference_mu * direction,
                )
                positive = teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=forward_config.scoring_micro_batch_size,
                )
                positive_objective = matched_token_grpo_surrogate(
                    positive,
                    rollout.old_token_log_probs,
                    sampler_token_log_probs,
                    rollout.advantages,
                    rollout.response_mask,
                    objective_config,
                )
                set_parameter_vector(
                    bundle,
                    center - forward_config.finite_difference_mu * direction,
                )
                negative = teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=forward_config.scoring_micro_batch_size,
                )
                negative_objective = matched_token_grpo_surrogate(
                    negative,
                    rollout.old_token_log_probs,
                    sampler_token_log_probs,
                    rollout.advantages,
                    rollout.response_mask,
                    objective_config,
                )
                finite_difference_coordinates.append(
                    (positive_objective - negative_objective)
                    / (2.0 * forward_config.finite_difference_mu)
                )
    finally:
        set_parameter_vector(bundle, center)
        for parameter, requires_grad in zip(
            bundle.trainable_parameters,
            original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
    if exact_objective_gradient is None or basis is None:
        raise RuntimeError("projected-gradient oracle did not produce exact coordinates")
    finite = torch.stack(finite_difference_coordinates).float()
    exact = basis.T @ exact_objective_gradient
    if not torch.isfinite(finite).all() or finite.norm() == 0 or exact.norm() == 0:
        raise RuntimeError("projected-gradient coordinates are non-finite or zero")
    cosine = float(torch.nn.functional.cosine_similarity(finite, exact, dim=0).item())
    relative_l2 = float(((finite - exact).norm() / exact.norm().clamp_min(1.0e-12)).item())
    after_digest = _vector_digest(parameter_vector(bundle))
    if not math.isfinite(cosine) or not math.isfinite(relative_l2):
        raise RuntimeError("projected-gradient comparison is non-finite")
    return ProjectedGradientGateReport(
        directions=forward_config.directions,
        finite_difference_mu=forward_config.finite_difference_mu,
        cosine_similarity=cosine,
        relative_l2_error=relative_l2,
        exact_projected_norm=float(exact.norm().item()),
        finite_difference_norm=float(finite.norm().item()),
        parameter_digest_before=before_digest,
        parameter_digest_after=after_digest,
        parameter_integrity=before_digest == after_digest,
    )


__all__ = ["ProjectedGradientGateReport", "fixed_rollout_projected_gradient_gate"]
