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
    grpo_objective_before: float
    grpo_objective_after: float
    grpo_loss_before: float
    grpo_loss_after: float
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
    focus_bootstrap_b_only: bool | None = None
    focus_a_rank: int | None = None
    focus_b_rank: int | None = None
    focus_a_update_count: int | None = None
    focus_b_update_count: int | None = None
    focus_state_numel: int | None = None
    focus_state_numel_cap: int | None = None
    focus_first_half_prompts: int | None = None
    focus_second_half_prompts: int | None = None
    focus_cross_sketch_count: int | None = None
    focus_state_update_policy_evaluations: int | None = None
    line_search_candidate_grpo_objective: float | None = None
    line_search_candidate_grpo_loss: float | None = None
    line_search_candidate_surrogate_improvement: float | None = None
    line_search_candidate_empirical_kl: float | None = None

    def __post_init__(self) -> None:
        objective_values = (
            self.grpo_objective_before,
            self.grpo_objective_after,
            self.grpo_loss_before,
            self.grpo_loss_after,
            self.surrogate_improvement,
        )
        if not all(math.isfinite(value) for value in objective_values):
            raise ValueError("GRPO objective telemetry must be finite")
        if not math.isclose(
            self.grpo_loss_before,
            -self.grpo_objective_before,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ) or not math.isclose(
            self.grpo_loss_after,
            -self.grpo_objective_after,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError("GRPO loss telemetry must equal the negative objective")
        if not math.isclose(
            self.surrogate_improvement,
            self.grpo_objective_after - self.grpo_objective_before,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError("surrogate improvement does not match the applied objective")

        candidate_values = (
            self.line_search_candidate_grpo_objective,
            self.line_search_candidate_grpo_loss,
            self.line_search_candidate_surrogate_improvement,
            self.line_search_candidate_empirical_kl,
        )
        if any(value is not None for value in candidate_values):
            if any(value is None for value in candidate_values):
                raise ValueError(
                    "line-search candidate telemetry must be all present or all absent"
                )
            candidate_objective = self.line_search_candidate_grpo_objective
            candidate_loss = self.line_search_candidate_grpo_loss
            candidate_improvement = self.line_search_candidate_surrogate_improvement
            candidate_kl = self.line_search_candidate_empirical_kl
            assert candidate_objective is not None
            assert candidate_loss is not None
            assert candidate_improvement is not None
            assert candidate_kl is not None
            if not all(math.isfinite(value) for value in candidate_values if value is not None):
                raise ValueError("line-search candidate telemetry must be finite")
            if not math.isclose(
                candidate_loss,
                -candidate_objective,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise ValueError("line-search candidate loss must equal the negative objective")
            if not math.isclose(
                candidate_improvement,
                candidate_objective - self.grpo_objective_before,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise ValueError("line-search candidate improvement is inconsistent")

        if self.accepted:
            if self.line_search_candidate_grpo_objective is None:
                raise ValueError("an accepted forward-only step must record its candidate")
            if not math.isclose(
                self.grpo_objective_after,
                self.line_search_candidate_grpo_objective,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ) or not math.isclose(
                self.empirical_kl,
                self.line_search_candidate_empirical_kl or 0.0,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise ValueError("accepted candidate telemetry differs from the applied policy")
        elif (
            self.grpo_objective_after != self.grpo_objective_before
            or self.grpo_loss_after != self.grpo_loss_before
            or self.surrogate_improvement != 0.0
            or self.empirical_kl != 0.0
            or self.step_norm != 0.0
        ):
            raise ValueError("a rejected forward-only step must leave applied metrics unchanged")


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

            set_parameter_vector(bundle, center - config.finite_difference_mu * direction)
            negative = teacher_forced_token_log_probs(
                bundle,
                rollout,
                micro_batch_size=config.scoring_micro_batch_size,
            )
            negative_logps.append(negative)
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
        length_normalize=True,
        sampling_weights=inference_weights,
    )
    # At the rollout center the unclipped and clipped PPO ratios both equal one.
    # The exact fixed-rollout GRPO directional derivative is therefore the
    # advantage-weighted token log-probability score, including the detached
    # inference correction and the objective's per-completion normalization.
    # Forming that statistic directly avoids catastrophic cancellation from
    # subtracting two nearly identical batch-reduced scalar objectives.
    objective_coordinates = statistics.gradient
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
    initial_objective_value = float(initial_objective.item())
    accepted = False
    applied_kl = 0.0
    applied_improvement = 0.0
    trials = 0
    search_evaluations = 0
    terminal_candidate_objective: float | None = None
    terminal_candidate_improvement: float | None = None
    terminal_candidate_kl: float | None = None
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
                terminal_candidate_objective = float(candidate_objective.item())
                terminal_candidate_improvement = (
                    terminal_candidate_objective - initial_objective_value
                )
                terminal_candidate_kl = kl_value
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

    applied_objective = terminal_candidate_objective if accepted else initial_objective_value
    if applied_objective is None:  # pragma: no cover - accepted implies a candidate
        raise RuntimeError("accepted line search did not retain its candidate objective")
    applied_improvement = applied_objective - initial_objective_value
    final_vector = parameter_vector(bundle).float()
    step_norm = float((final_vector - center).norm().item()) if accepted else 0.0
    probe_evaluations = 2 * config.directions
    policy_evaluations = probe_evaluations + search_evaluations
    micro_batches = math.ceil(rollout.environment_samples / config.scoring_micro_batch_size)
    zero_groups = rollout.advantages.abs().amax(dim=1).eq(0)
    return MatchedForwardStepResult(
        accepted=accepted,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=float(zero_groups.float().mean().item()),
        empirical_kl=applied_kl,
        surrogate_improvement=applied_improvement,
        grpo_objective_before=initial_objective_value,
        grpo_objective_after=applied_objective,
        grpo_loss_before=-initial_objective_value,
        grpo_loss_after=-applied_objective,
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
        derivative_variance=float(objective_coordinates.var(unbiased=False).item()),
        line_search_candidate_grpo_objective=terminal_candidate_objective,
        line_search_candidate_grpo_loss=(
            -terminal_candidate_objective if terminal_candidate_objective is not None else None
        ),
        line_search_candidate_surrogate_improvement=terminal_candidate_improvement,
        line_search_candidate_empirical_kl=terminal_candidate_kl,
    )


__all__ = [
    "MatchedForwardConfig",
    "MatchedForwardStepResult",
    "matched_forward_npg_step",
]
