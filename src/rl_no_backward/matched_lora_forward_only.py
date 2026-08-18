"""Strict inference-only finite-direction NPG for the matched PEFT LoRA policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .common import fisher_condition_number, make_search_basis
from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
    matched_token_grpo_surrogate,
    sampled_hf_policy_kl,
)
from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled, set_parameter_vector
from .sequence_policy import (
    SequenceRolloutBatch,
    central_difference_score_statistics,
    teacher_forced_token_log_probs,
)


@dataclass(frozen=True, slots=True)
class MatchedForwardConfig:
    directions: int = 8
    finite_difference_mu: float = 1.0
    fisher_damping: float = 0.2
    kl_budget: float = 0.001
    max_step_norm: float = 0.1
    line_search_steps: int = 6
    line_search_decay: float = 0.5
    minimum_surrogate_improvement: float = -1.0e-8
    scoring_micro_batch_size: int = 16

    def __post_init__(self) -> None:
        for name in ("directions", "line_search_steps", "scoring_micro_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "finite_difference_mu",
            "kl_budget",
            "max_step_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.fisher_damping) or self.fisher_damping < 0:
            raise ValueError("fisher_damping must be non-negative and finite")
        if not 0 < self.line_search_decay < 1:
            raise ValueError("line_search_decay must lie in (0, 1)")
        if not math.isfinite(self.minimum_surrogate_improvement):
            raise ValueError("minimum_surrogate_improvement must be finite")


@dataclass(frozen=True, slots=True)
class MatchedForwardStepResult:
    accepted: bool
    reward_mean: float
    zero_advantage_fraction: float
    empirical_kl: float
    surrogate_improvement: float
    step_norm: float
    projected_gradient_norm: float
    fisher_condition: float
    line_search_trials: int
    policy_evaluations: int
    forward_calls: int
    backward_calls: int
    environment_samples: int
    teacher_forced_examples: int
    scored_tokens: int
    derivative_variance: float


@torch.inference_mode()
def matched_forward_npg_step(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    generator: torch.Generator,
    objective_config: MatchedGRPOObjectiveConfig,
    config: MatchedForwardConfig,
) -> MatchedForwardStepResult:
    """Update all LoRA weights using paired fixed-rollout forward probes only."""

    if rollout.frozen_prefix_cache is not None:
        raise ValueError("all-layer LoRA is incompatible with frozen-prefix scoring")
    if sampler_token_log_probs.shape != rollout.old_token_log_probs.shape:
        raise ValueError("sampler log-probability shape mismatch")
    bundle.model.eval()
    set_adapter_grad_enabled(bundle, False)
    for policy_parameter in bundle.trainable_parameters:
        policy_parameter.grad = None
    if any(
        policy_parameter.requires_grad or policy_parameter.grad is not None
        for policy_parameter in bundle.trainable_parameters
    ):
        raise RuntimeError("forward-only policy parameters must be frozen with no gradients")
    center = parameter_vector(bundle).float().clone()
    if config.directions > center.numel():
        raise ValueError("directions exceed LoRA parameter count")
    basis = make_search_basis(
        center.numel(),
        config.directions,
        generator,
        bundle.device,
    )

    positive_logps: list[Tensor] = []
    negative_logps: list[Tensor] = []
    positive_objectives: list[Tensor] = []
    negative_objectives: list[Tensor] = []
    try:
        for direction_index in range(config.directions):
            direction = basis[:, direction_index]
            set_parameter_vector(bundle, center + config.finite_difference_mu * direction)
            positive = teacher_forced_token_log_probs(
                bundle,
                rollout,
                micro_batch_size=config.scoring_micro_batch_size,
            )
            positive_logps.append(positive)
            positive_objectives.append(
                matched_token_grpo_surrogate(
                    positive,
                    rollout.old_token_log_probs,
                    sampler_token_log_probs,
                    rollout.advantages,
                    rollout.response_mask,
                    objective_config,
                )
            )

            set_parameter_vector(bundle, center - config.finite_difference_mu * direction)
            negative = teacher_forced_token_log_probs(
                bundle,
                rollout,
                micro_batch_size=config.scoring_micro_batch_size,
            )
            negative_logps.append(negative)
            negative_objectives.append(
                matched_token_grpo_surrogate(
                    negative,
                    rollout.old_token_log_probs,
                    sampler_token_log_probs,
                    rollout.advantages,
                    rollout.response_mask,
                    objective_config,
                )
            )
    finally:
        set_parameter_vector(bundle, center)

    positive_stack = torch.stack(positive_logps, dim=-1)
    negative_stack = torch.stack(negative_logps, dim=-1)
    inference_weights = detached_inference_correction(
        rollout.old_token_log_probs,
        sampler_token_log_probs,
        rollout.response_mask,
        objective_config,
    )
    statistics = central_difference_score_statistics(
        positive_stack,
        negative_stack,
        rollout,
        config.finite_difference_mu,
        length_normalize=False,
        sampling_weights=inference_weights,
    )
    objective_coordinates = (
        torch.stack(positive_objectives) - torch.stack(negative_objectives)
    ) / (2.0 * config.finite_difference_mu)
    fisher_scale = (torch.trace(statistics.fisher) / config.directions).clamp_min(1.0e-8)
    regularized_fisher = statistics.fisher + config.fisher_damping * fisher_scale * torch.eye(
        config.directions,
        dtype=statistics.fisher.dtype,
        device=statistics.fisher.device,
    )
    natural_coordinates = torch.linalg.solve(regularized_fisher, objective_coordinates)
    parameter_direction = basis @ natural_coordinates
    direction_norm = float(parameter_direction.norm().item())
    curvature = float(
        torch.dot(natural_coordinates, statistics.fisher @ natural_coordinates).item()
    )
    if direction_norm <= 1.0e-12 or not math.isfinite(direction_norm):
        initial_scale = 0.0
    else:
        norm_scale = config.max_step_norm / direction_norm
        trust_scale = (
            math.sqrt(2.0 * config.kl_budget / curvature)
            if curvature > 1.0e-12 and math.isfinite(curvature)
            else norm_scale
        )
        initial_scale = min(norm_scale, trust_scale)

    initial_objective = matched_token_grpo_surrogate(
        rollout.old_token_log_probs,
        rollout.old_token_log_probs,
        sampler_token_log_probs,
        rollout.advantages,
        rollout.response_mask,
        objective_config,
    )
    accepted = False
    applied_kl = 0.0
    applied_improvement = 0.0
    trials = 0
    search_evaluations = 0
    if initial_scale > 0:
        try:
            for trial_index in range(config.line_search_steps):
                trials = trial_index + 1
                scale = initial_scale * config.line_search_decay**trial_index
                set_parameter_vector(bundle, center + scale * parameter_direction)
                candidate_logps = teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=config.scoring_micro_batch_size,
                )
                search_evaluations += 1
                candidate_objective = matched_token_grpo_surrogate(
                    candidate_logps,
                    rollout.old_token_log_probs,
                    sampler_token_log_probs,
                    rollout.advantages,
                    rollout.response_mask,
                    objective_config,
                )
                candidate_kl = sampled_hf_policy_kl(
                    candidate_logps,
                    rollout.old_token_log_probs,
                    rollout.response_mask,
                    sampling_weights=inference_weights,
                )
                improvement = float((candidate_objective - initial_objective).item())
                kl_value = float(candidate_kl.item())
                if (
                    math.isfinite(improvement)
                    and math.isfinite(kl_value)
                    and improvement >= config.minimum_surrogate_improvement
                    and kl_value <= config.kl_budget * 1.05
                ):
                    accepted = True
                    applied_improvement = improvement
                    applied_kl = kl_value
                    break
        finally:
            if not accepted:
                set_parameter_vector(bundle, center)

    final_vector = parameter_vector(bundle).float()
    step_norm = float((final_vector - center).norm().item()) if accepted else 0.0
    probe_evaluations = 2 * config.directions
    policy_evaluations = probe_evaluations + search_evaluations
    micro_batches = math.ceil(rollout.environment_samples / config.scoring_micro_batch_size)
    directional_rewards = torch.stack(positive_objectives) - torch.stack(negative_objectives)
    zero_groups = rollout.advantages.abs().amax(dim=1).eq(0)
    return MatchedForwardStepResult(
        accepted=accepted,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=float(zero_groups.float().mean().item()),
        empirical_kl=applied_kl,
        surrogate_improvement=applied_improvement,
        step_norm=step_norm,
        projected_gradient_norm=float(objective_coordinates.norm().item()),
        fisher_condition=fisher_condition_number(statistics.fisher),
        line_search_trials=trials,
        policy_evaluations=policy_evaluations,
        forward_calls=policy_evaluations * micro_batches,
        backward_calls=0,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=policy_evaluations * rollout.environment_samples,
        scored_tokens=policy_evaluations * rollout.valid_response_tokens,
        derivative_variance=float(directional_rewards.var(unbiased=False).item()),
    )


__all__ = [
    "MatchedForwardConfig",
    "MatchedForwardStepResult",
    "matched_forward_npg_step",
]
