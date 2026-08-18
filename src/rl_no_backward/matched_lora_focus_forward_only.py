"""Strict inference-only FOCUS-NPG for the matched PEFT LoRA policy.

The optimizer uses the same fixed-rollout token-score gradient, empirical
Fisher, natural-gradient solve, and line search as :mod:`matched_lora_forward_only`.
Only the q=8 search basis changes: standard LoRA receives one B-only bootstrap
round and then disjoint A4/B4 probes guided by rank-two cross-sketch state. At
the production finite radius and BF16 precision this is an approximate geometry
heuristic, not a claim of an unbiased covariance estimator.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .common import fisher_condition_number
from .matched_focus import (
    MatchedFocusState,
    build_focus_probe_plan,
    canonical_lora_partition,
    requires_b_only_bootstrap,
)
from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
    matched_token_grpo_surrogate,
    sampled_hf_policy_kl,
)
from .matched_lora_forward_only import MatchedForwardConfig, MatchedForwardStepResult
from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled, set_parameter_vector
from .sequence_policy import (
    ProjectedScoreStatistics,
    SequenceRolloutBatch,
    central_difference_score_statistics,
    teacher_forced_token_log_probs,
)


def deterministic_prompt_half_coordinates(
    statistics: ProjectedScoreStatistics,
    rollout: SequenceRolloutBatch,
) -> tuple[Tensor, Tensor]:
    """Return objective coordinates from deterministic, disjoint prompt halves.

    Even and odd prompt positions form the two halves.  The completion scores
    were already computed from the q=8 paired token-log-probability probes, so
    covariance observation performs no additional model evaluation.
    """

    if rollout.batch_size < 2 or rollout.batch_size % 2:
        raise ValueError("matched FOCUS requires an even prompt batch of at least two")
    if statistics.completion_scores.shape[:2] != (
        rollout.batch_size,
        rollout.group_size,
    ):
        raise ValueError("projected completion scores do not match the rollout")
    contributions = rollout.advantages.unsqueeze(-1) * statistics.completion_scores
    first = contributions[0::2].mean(dim=(0, 1))
    second = contributions[1::2].mean(dim=(0, 1))
    if (
        first.shape != second.shape
        or not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
    ):
        raise RuntimeError("FOCUS prompt-half coordinates are malformed or non-finite")
    return first, second


@torch.inference_mode()
def matched_focus_npg_step(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    generator: torch.Generator,
    objective_config: MatchedGRPOObjectiveConfig,
    config: MatchedForwardConfig,
    state: MatchedFocusState,
) -> MatchedForwardStepResult:
    """Update all LoRA weights with stateful q=8 FOCUS probes and no backward pass."""

    if rollout.frozen_prefix_cache is not None:
        raise ValueError("all-layer LoRA is incompatible with frozen-prefix scoring")
    if sampler_token_log_probs.shape != rollout.old_token_log_probs.shape:
        raise ValueError("sampler log-probability shape mismatch")
    if config.directions != 8:
        raise ValueError("matched FOCUS requires exactly eight search directions")
    if rollout.batch_size < 2 or rollout.batch_size % 2:
        raise ValueError("matched FOCUS requires an even prompt batch of at least two")
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
    partition = canonical_lora_partition(bundle.model)
    if state.partition.layout_digest != partition.layout_digest:
        raise ValueError("FOCUS state belongs to a different LoRA parameter layout")
    if tuple(bundle.adapter_names) != tuple(span.name for span in partition.spans):
        raise RuntimeError("bundle parameter order differs from the canonical LoRA layout")
    if center.numel() != partition.full_dimension:
        raise RuntimeError("bundle parameter count differs from the canonical LoRA layout")
    bootstrap_b_only = state.family_state("B").update_count == 0 and requires_b_only_bootstrap(
        center, partition
    )
    plan = build_focus_probe_plan(
        partition,
        state,
        generator,
        bootstrap_b_only=bootstrap_b_only,
        device=bundle.device,
        dtype=torch.float32,
    )
    basis = plan.basis

    positive_logps: list[Tensor] = []
    negative_logps: list[Tensor] = []
    try:
        for direction_index in range(config.directions):
            direction = basis[:, direction_index]
            set_parameter_vector(bundle, center + config.finite_difference_mu * direction)
            positive_logps.append(
                teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=config.scoring_micro_batch_size,
                )
            )
            set_parameter_vector(bundle, center - config.finite_difference_mu * direction)
            negative_logps.append(
                teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=config.scoring_micro_batch_size,
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
        length_normalize=True,
        sampling_weights=inference_weights,
    )
    objective_coordinates = statistics.gradient

    # These two observations reuse the token scores above.  Updating before the
    # line search is intentional: rejected policy proposals still contribute an
    # on-policy covariance observation to the next search basis.
    first_half, second_half = deterministic_prompt_half_coordinates(statistics, rollout)
    cross_sketches = state.update_from_half_batch_coordinates(
        plan,
        first_half,
        second_half,
    )

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
    a_state = state.family_state("A")
    b_state = state.family_state("B")
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
        focus_bootstrap_b_only=bootstrap_b_only,
        focus_a_rank=a_state.rank,
        focus_b_rank=b_state.rank,
        focus_a_update_count=a_state.update_count,
        focus_b_update_count=b_state.update_count,
        focus_state_numel=state.persistent_numel,
        focus_state_numel_cap=state.persistent_numel_cap,
        focus_first_half_prompts=rollout.batch_size // 2,
        focus_second_half_prompts=rollout.batch_size // 2,
        focus_cross_sketch_count=len(cross_sketches),
        focus_state_update_policy_evaluations=0,
        line_search_candidate_grpo_objective=terminal_candidate_objective,
        line_search_candidate_grpo_loss=(
            -terminal_candidate_objective if terminal_candidate_objective is not None else None
        ),
        line_search_candidate_surrogate_improvement=terminal_candidate_improvement,
        line_search_candidate_empirical_kl=terminal_candidate_kl,
    )


__all__ = [
    "deterministic_prompt_half_coordinates",
    "matched_focus_npg_step",
]
