"""Strict inference-only projected policy optimizers for sequence rollouts.

Every model evaluation in this module is reached from a function decorated
with :func:`torch.inference_mode`.  The module contains no reverse-mode or
autograd API calls; its directional information comes exclusively from paired
ordinary forward evaluations of fixed candidate sequences.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .common import fisher_condition_number, make_search_basis
from .model import (
    ModelBundle,
    parameter_vector,
    set_adapter_grad_enabled,
    set_parameter_vector,
)
from .sequence_fastpath import FusedProbeConfig, fused_directional_token_log_probs
from .sequence_policy import (
    ProjectedScoreStatistics,
    SequenceRolloutBatch,
    central_difference_score_statistics,
    clipped_grpo_surrogate,
    teacher_forced_token_log_probs,
)

ForwardSequenceMethod = Literal["fo_pg", "fo_npg", "focus_npg"]


@dataclass(frozen=True, slots=True)
class ForwardSequenceConfig:
    """Settings for symmetric-probe sequence FO-PG and natural-gradient steps."""

    method: ForwardSequenceMethod = "focus_npg"
    directions: int = 8
    finite_difference_mu: float = 1.0
    fisher_damping: float = 0.1
    kl_budget: float = 0.02
    clip_epsilon: float = 0.2
    max_step_norm: float = 1.0
    line_search_steps: int = 6
    line_search_decay: float = 0.5
    minimum_surrogate_improvement: float = -1e-8
    active_rank: int = 4
    history_size: int = 16
    length_normalize_scores: bool = True
    scoring_micro_batch_size: int | None = None
    use_fused_probes: bool = False
    fused_probe_directions_per_forward: int = 2
    fused_probe_examples_per_forward: int = 4

    def __post_init__(self) -> None:
        if self.method not in {"fo_pg", "fo_npg", "focus_npg"}:
            raise ValueError(f"unsupported forward sequence method {self.method!r}")
        for name, value, minimum in (
            ("directions", self.directions, 1),
            ("line_search_steps", self.line_search_steps, 1),
            ("active_rank", self.active_rank, 1),
            ("history_size", self.history_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value, allow_zero in (
            ("finite_difference_mu", self.finite_difference_mu, False),
            ("fisher_damping", self.fisher_damping, True),
            ("kl_budget", self.kl_budget, False),
            ("clip_epsilon", self.clip_epsilon, False),
            ("max_step_norm", self.max_step_norm, False),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            lower_bound_ok = value >= 0 if allow_zero else value > 0
            if not lower_bound_ok or not math.isfinite(float(value)):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier} and finite")
        if (
            isinstance(self.line_search_decay, bool)
            or not isinstance(self.line_search_decay, (int, float))
            or not 0 < self.line_search_decay < 1
        ):
            raise ValueError("line_search_decay must lie strictly between zero and one")
        if isinstance(self.minimum_surrogate_improvement, bool) or not isinstance(
            self.minimum_surrogate_improvement, (int, float)
        ):
            raise TypeError("minimum_surrogate_improvement must be a real number")
        if not math.isfinite(float(self.minimum_surrogate_improvement)):
            raise ValueError("minimum_surrogate_improvement must be finite")
        if not isinstance(self.length_normalize_scores, bool):
            raise TypeError("length_normalize_scores must be boolean")
        if not isinstance(self.use_fused_probes, bool):
            raise TypeError("use_fused_probes must be boolean")
        for name, value in (
            (
                "fused_probe_directions_per_forward",
                self.fused_probe_directions_per_forward,
            ),
            ("fused_probe_examples_per_forward", self.fused_probe_examples_per_forward),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.scoring_micro_batch_size is not None and (
            isinstance(self.scoring_micro_batch_size, bool)
            or not isinstance(self.scoring_micro_batch_size, int)
            or self.scoring_micro_batch_size < 1
        ):
            raise ValueError("scoring_micro_batch_size must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class ForwardSequenceStepResult:
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
    full_prefix_calls: int = 0
    suffix_calls: int = 0


class SequenceActiveSubspace:
    """Online low-rank basis of recent forward-only gradient sketches."""

    def __init__(self, rank: int, history_size: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        self.rank = rank
        self.history: deque[Tensor] = deque(maxlen=history_size)
        self.basis: Tensor | None = None

    @torch.inference_mode()
    def update(self, sketch: Tensor) -> None:
        flat_sketch = sketch.float().reshape(-1)
        norm = flat_sketch.norm()
        if not torch.isfinite(norm) or norm <= 1e-12:
            return
        if self.history and self.history[0].numel() != flat_sketch.numel():
            raise ValueError("active-subspace sketch dimension changed")
        self.history.append((flat_sketch / norm).detach())
        matrix = torch.stack(list(self.history), dim=1)
        left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        self.basis = left[:, : min(self.rank, left.shape[1])].contiguous()


def _micro_batches_per_evaluation(
    rollout: SequenceRolloutBatch, micro_batch_size: int | None
) -> int:
    if micro_batch_size is None:
        return 1
    return math.ceil(rollout.environment_samples / micro_batch_size)


@torch.inference_mode()
def sampled_sequence_kl(new_token_log_probs: Tensor, rollout: SequenceRolloutBatch) -> Tensor:
    """Return the non-negative k3 estimator of KL(old policy || new policy)."""

    if new_token_log_probs.shape != rollout.response_input_ids.shape:
        raise ValueError("new_token_log_probs must have shape [B, G, T]")
    log_ratio = new_token_log_probs.float() - rollout.old_token_log_probs.float()
    token_kl = torch.expm1(log_ratio) - log_ratio
    token_kl = token_kl.masked_fill(~rollout.response_mask, 0.0)
    lengths = rollout.response_lengths.clamp_min(1).to(token_kl.dtype)
    return (token_kl.sum(dim=-1) / lengths).mean()


@torch.inference_mode()
def directional_sequence_score_statistics(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    center: Tensor,
    basis: Tensor,
    finite_difference_mu: float,
    *,
    length_normalize: bool = True,
    scoring_micro_batch_size: int | None = None,
    fused_probe_config: FusedProbeConfig | None = None,
) -> tuple[ProjectedScoreStatistics, int]:
    """Symmetrically probe fixed response log probabilities in every basis direction."""

    if basis.ndim != 2 or basis.shape[0] != center.numel() or basis.shape[1] == 0:
        raise ValueError("basis must have shape [parameter_count, directions>=1]")
    if finite_difference_mu <= 0 or not math.isfinite(finite_difference_mu):
        raise ValueError("finite_difference_mu must be positive and finite")
    statistics, policy_evaluations, _, _, _ = _directional_sequence_score_statistics_with_counts(
        bundle,
        rollout,
        center,
        basis,
        finite_difference_mu,
        length_normalize=length_normalize,
        scoring_micro_batch_size=scoring_micro_batch_size,
        fused_probe_config=fused_probe_config,
    )
    return statistics, policy_evaluations


@torch.inference_mode()
def _directional_sequence_score_statistics_with_counts(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    center: Tensor,
    basis: Tensor,
    finite_difference_mu: float,
    *,
    length_normalize: bool,
    scoring_micro_batch_size: int | None,
    fused_probe_config: FusedProbeConfig | None,
) -> tuple[ProjectedScoreStatistics, int, int, int, int]:
    """Return projected statistics plus logical evaluations and model calls."""

    if basis.ndim != 2 or basis.shape[0] != center.numel() or basis.shape[1] == 0:
        raise ValueError("basis must have shape [parameter_count, directions>=1]")
    if finite_difference_mu <= 0 or not math.isfinite(finite_difference_mu):
        raise ValueError("finite_difference_mu must be positive and finite")

    policy_evaluations = 2 * basis.shape[1]
    if fused_probe_config is not None:
        try:
            fused = fused_directional_token_log_probs(
                bundle,
                rollout,
                center,
                basis,
                finite_difference_mu,
                fused_probe_config,
            )
        finally:
            # Match the scalar implementation's public postcondition even
            # when a caller supplied a center different from the live policy.
            set_parameter_vector(bundle, center)
        statistics = central_difference_score_statistics(
            fused.positive_token_log_probs,
            fused.negative_token_log_probs,
            rollout,
            finite_difference_mu,
            length_normalize=length_normalize,
        )
        return (
            statistics,
            policy_evaluations,
            fused.model_calls,
            fused.full_prefix_calls,
            fused.suffix_calls,
        )

    positive: list[Tensor] = []
    negative: list[Tensor] = []
    try:
        for index in range(basis.shape[1]):
            direction = basis[:, index]
            set_parameter_vector(bundle, center + finite_difference_mu * direction)
            positive.append(
                teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=scoring_micro_batch_size,
                )
            )
            set_parameter_vector(bundle, center - finite_difference_mu * direction)
            negative.append(
                teacher_forced_token_log_probs(
                    bundle,
                    rollout,
                    micro_batch_size=scoring_micro_batch_size,
                )
            )
    finally:
        set_parameter_vector(bundle, center)

    statistics = central_difference_score_statistics(
        torch.stack(positive, dim=-1),
        torch.stack(negative, dim=-1),
        rollout,
        finite_difference_mu,
        length_normalize=length_normalize,
    )
    model_calls = policy_evaluations * _micro_batches_per_evaluation(
        rollout, scoring_micro_batch_size
    )
    if rollout.frozen_prefix_cache is None:
        return statistics, policy_evaluations, model_calls, model_calls, model_calls
    return statistics, policy_evaluations, model_calls, 0, model_calls


@torch.inference_mode()
def _solve_projected_coordinates(
    gradient: Tensor,
    fisher: Tensor,
    config: ForwardSequenceConfig,
) -> Tensor:
    if config.method == "fo_pg":
        return gradient
    scale = (torch.trace(fisher) / max(fisher.shape[0], 1)).clamp_min(1e-8)
    damping = config.fisher_damping * scale
    regularized = fisher + damping * torch.eye(
        fisher.shape[0], device=fisher.device, dtype=fisher.dtype
    )
    return torch.linalg.solve(regularized, gradient)


@torch.inference_mode()
def _initial_step_scale(
    coordinates: Tensor,
    fisher: Tensor,
    parameter_direction: Tensor,
    config: ForwardSequenceConfig,
) -> float:
    direction_norm = float(parameter_direction.norm().item())
    if direction_norm <= 1e-12 or not math.isfinite(direction_norm):
        return 0.0
    curvature = float(torch.dot(coordinates, fisher @ coordinates).item())
    if not math.isfinite(curvature):
        return 0.0
    norm_limited_scale = config.max_step_norm / direction_norm
    if curvature <= 1e-12:
        return norm_limited_scale
    trust_region_scale = math.sqrt(2.0 * config.kl_budget / curvature)
    return min(norm_limited_scale, trust_region_scale)


@torch.inference_mode()
def _sequence_line_search(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    center: Tensor,
    parameter_direction: Tensor,
    initial_scale: float,
    config: ForwardSequenceConfig,
) -> tuple[bool, float, float, int, int]:
    initial_surrogate = float(
        clipped_grpo_surrogate(rollout.old_token_log_probs, rollout, config.clip_epsilon).item()
    )
    accepted = False
    last_kl = 0.0
    last_improvement = 0.0
    evaluations = 0
    trials = 0
    try:
        for trial in range(config.line_search_steps):
            trials = trial + 1
            scale = initial_scale * config.line_search_decay**trial
            set_parameter_vector(bundle, center + scale * parameter_direction)
            new_token_log_probs = teacher_forced_token_log_probs(
                bundle,
                rollout,
                micro_batch_size=config.scoring_micro_batch_size,
            )
            evaluations += 1
            kl = sampled_sequence_kl(new_token_log_probs, rollout)
            surrogate = clipped_grpo_surrogate(new_token_log_probs, rollout, config.clip_epsilon)
            last_kl = float(kl.item())
            last_improvement = float(surrogate.item()) - initial_surrogate
            if (
                math.isfinite(last_kl)
                and math.isfinite(last_improvement)
                and last_kl <= config.kl_budget * 1.05
                and last_improvement >= config.minimum_surrogate_improvement
            ):
                accepted = True
                return (
                    True,
                    last_kl,
                    last_improvement,
                    trials,
                    evaluations,
                )
    finally:
        if not accepted:
            set_parameter_vector(bundle, center)
    return False, last_kl, last_improvement, trials, evaluations


@torch.inference_mode()
def forward_sequence_step(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    generator: torch.Generator,
    config: ForwardSequenceConfig,
    active_subspace: SequenceActiveSubspace | None = None,
) -> ForwardSequenceStepResult:
    """Take one strict inference-only FO-PG/FO-NPG/FOCUS-NPG sequence step."""

    bundle.model.eval()
    set_adapter_grad_enabled(bundle, False)
    for parameter in bundle.trainable_parameters:
        parameter.grad = None
    center = parameter_vector(bundle).float().clone()
    if center.numel() == 0:
        raise ValueError("forward sequence optimization requires adapter parameters")
    if config.directions > center.numel():
        raise ValueError("directions cannot exceed the adapter parameter count")

    active_basis = None
    if config.method == "focus_npg" and active_subspace is not None:
        active_basis = active_subspace.basis
    basis = make_search_basis(
        center.numel(),
        config.directions,
        generator,
        bundle.device,
        active_basis=active_basis,
    )
    fused_probe_config = None
    if config.use_fused_probes:
        fused_probe_config = FusedProbeConfig(
            directions_per_forward=config.fused_probe_directions_per_forward,
            examples_per_forward=config.fused_probe_examples_per_forward,
        )
    (
        statistics,
        probe_evaluations,
        probe_forward_calls,
        probe_full_prefix_calls,
        probe_suffix_calls,
    ) = _directional_sequence_score_statistics_with_counts(
        bundle,
        rollout,
        center,
        basis,
        config.finite_difference_mu,
        length_normalize=config.length_normalize_scores,
        scoring_micro_batch_size=config.scoring_micro_batch_size,
        fused_probe_config=fused_probe_config,
    )
    coordinates = _solve_projected_coordinates(statistics.gradient, statistics.fisher, config)
    parameter_direction = basis @ coordinates
    initial_scale = _initial_step_scale(coordinates, statistics.fisher, parameter_direction, config)

    if initial_scale > 0:
        accepted, kl, improvement, trials, search_evaluations = _sequence_line_search(
            bundle,
            rollout,
            center,
            parameter_direction,
            initial_scale,
            config,
        )
    else:
        set_parameter_vector(bundle, center)
        accepted = False
        kl = 0.0
        improvement = 0.0
        trials = 0
        search_evaluations = 0

    if config.method == "focus_npg" and active_subspace is not None:
        active_subspace.update(basis @ statistics.gradient)

    # These are metrics for the update actually applied to the policy.  The
    # line search restores the center on rejection, so applied KL/improvement
    # are exactly zero rather than the last rejected candidate's diagnostics.
    if not accepted:
        kl = 0.0
        improvement = 0.0

    weighted_scores = (rollout.advantages.unsqueeze(-1) * statistics.completion_scores).reshape(
        -1, statistics.completion_scores.shape[-1]
    )
    derivative_variance = float(weighted_scores.var(dim=0, unbiased=False).mean().item())
    policy_evaluations = probe_evaluations + search_evaluations
    search_forward_calls = search_evaluations * _micro_batches_per_evaluation(
        rollout, config.scoring_micro_batch_size
    )
    forward_calls = probe_forward_calls + search_forward_calls
    if rollout.frozen_prefix_cache is None:
        full_prefix_calls = probe_full_prefix_calls + search_forward_calls
        suffix_calls = probe_suffix_calls + search_forward_calls
    else:
        full_prefix_calls = probe_full_prefix_calls
        suffix_calls = probe_suffix_calls + search_forward_calls
    final_parameters = parameter_vector(bundle).float()
    actual_step_norm = float((final_parameters - center).norm().item())
    if not accepted:
        actual_step_norm = 0.0
    return ForwardSequenceStepResult(
        accepted=accepted,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=rollout.zero_advantage_fraction,
        empirical_kl=kl,
        surrogate_improvement=improvement,
        step_norm=actual_step_norm,
        projected_gradient_norm=float(statistics.gradient.norm().item()),
        fisher_condition=fisher_condition_number(statistics.fisher),
        line_search_trials=trials,
        policy_evaluations=policy_evaluations,
        forward_calls=forward_calls,
        backward_calls=0,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=policy_evaluations * rollout.environment_samples,
        scored_tokens=policy_evaluations * rollout.valid_response_tokens,
        derivative_variance=derivative_variance,
        full_prefix_calls=full_prefix_calls,
        suffix_calls=suffix_calls,
    )


__all__ = [
    "ForwardSequenceConfig",
    "ForwardSequenceMethod",
    "ForwardSequenceStepResult",
    "SequenceActiveSubspace",
    "directional_sequence_score_statistics",
    "forward_sequence_step",
    "sampled_sequence_kl",
]
