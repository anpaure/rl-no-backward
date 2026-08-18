"""Strict backward-pass-free policy optimizers.

All model evaluations in this module run under ``torch.inference_mode``.  The
implementation intentionally contains no reverse-mode/autograd gradient calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

import torch
from torch import Tensor

from .common import (
    RolloutBatch,
    categorical_kl,
    clipped_surrogate,
    fisher_condition_number,
    gather_action_values,
    make_search_basis,
    sample_with_uniforms,
)
from .model import (
    ModelBundle,
    candidate_log_probs,
    parameter_vector,
    set_adapter_grad_enabled,
    set_parameter_vector,
)

ForwardMethod = Literal["fo_pg", "fo_npg", "focus_npg", "es"]


@dataclass
class ForwardConfig:
    method: ForwardMethod = "focus_npg"
    directions: int = 8
    finite_difference_mu: float = 0.5
    fisher_damping: float = 0.1
    kl_budget: float = 0.02
    clip_epsilon: float = 0.2
    max_step_norm: float = 5.0
    line_search_steps: int = 6
    line_search_decay: float = 0.5
    active_rank: int = 4
    history_size: int = 16
    es_sigma: float = 0.5
    es_step_norm: float = 0.5


@dataclass
class ForwardStepResult:
    accepted: bool
    reward_mean: float
    zero_advantage_fraction: float
    empirical_kl: float
    surrogate_improvement: float
    step_norm: float
    projected_gradient_norm: float
    fisher_condition: float
    line_search_trials: int
    forward_calls: int
    backward_calls: int
    environment_samples: int
    teacher_forced_examples: int
    derivative_variance: float
    extra: dict[str, float] = field(default_factory=dict)


class ActiveSubspace:
    """Online low-rank basis from recent forward projected-gradient sketches."""

    def __init__(self, rank: int, history_size: int) -> None:
        self.rank = int(rank)
        self.history: deque[Tensor] = deque(maxlen=int(history_size))
        self.basis: Tensor | None = None

    @torch.inference_mode()
    def update(self, sketch: Tensor) -> None:
        norm = sketch.float().norm()
        if not torch.isfinite(norm) or norm <= 1e-12:
            return
        self.history.append((sketch.float() / norm).detach())
        matrix = torch.stack(list(self.history), dim=1)
        left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        self.basis = left[:, : min(self.rank, left.shape[1])].contiguous()


@torch.inference_mode()
def directional_log_probability_scores(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    center: Tensor,
    basis: Tensor,
    mu: float,
) -> tuple[Tensor, int]:
    """Central-difference every candidate action's log probability along V."""

    if mu <= 0:
        raise ValueError("finite-difference mu must be positive")
    scores: list[Tensor] = []
    try:
        for index in range(basis.shape[1]):
            direction = basis[:, index]
            set_parameter_vector(bundle, center + mu * direction)
            plus = candidate_log_probs(bundle, encoded)
            set_parameter_vector(bundle, center - mu * direction)
            minus = candidate_log_probs(bundle, encoded)
            scores.append((plus - minus) / (2.0 * mu))
    finally:
        set_parameter_vector(bundle, center)
    return torch.stack(scores, dim=-1), 2 * basis.shape[1]


def projected_policy_gradient(
    candidate_scores: Tensor,
    rollout: RolloutBatch,
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate projected score-function gradient and empirical Fisher."""

    action_scores = gather_action_values(candidate_scores, rollout.actions)
    gradient = (rollout.advantages.unsqueeze(-1) * action_scores).mean(dim=(0, 1))
    flat_scores = action_scores.reshape(-1, action_scores.shape[-1])
    fisher = flat_scores.T @ flat_scores / max(flat_scores.shape[0], 1)
    return gradient, fisher, action_scores


def _solve_coordinates(gradient: Tensor, fisher: Tensor, config: ForwardConfig) -> Tensor:
    if config.method == "fo_pg":
        return gradient
    scale = (torch.trace(fisher) / max(fisher.shape[0], 1)).clamp_min(1e-8)
    damping = config.fisher_damping * scale
    regularized = fisher + damping * torch.eye(
        fisher.shape[0], device=fisher.device, dtype=fisher.dtype
    )
    return torch.linalg.solve(regularized, gradient)


def _initial_trust_region_scale(
    coordinates: Tensor,
    fisher: Tensor,
    config: ForwardConfig,
) -> float:
    curvature = torch.dot(coordinates, fisher @ coordinates).clamp_min(1e-12)
    alpha = sqrt(2.0 * config.kl_budget / float(curvature.item()))
    proposed_norm = alpha * float(coordinates.norm().item())
    if proposed_norm > config.max_step_norm:
        alpha *= config.max_step_norm / proposed_norm
    return alpha


@torch.inference_mode()
def _forward_line_search(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    center: Tensor,
    direction: Tensor,
    initial_alpha: float,
    rollout: RolloutBatch,
    config: ForwardConfig,
) -> tuple[bool, float, float, float, int, int]:
    calls = 0
    last_kl = float("nan")
    last_surrogate = float("nan")
    for trial in range(config.line_search_steps):
        alpha = initial_alpha * config.line_search_decay**trial
        set_parameter_vector(bundle, center + alpha * direction)
        new_log_probs = candidate_log_probs(bundle, encoded)
        calls += 1
        kl = float(categorical_kl(rollout.old_log_probs, new_log_probs).item())
        surrogate = float(clipped_surrogate(new_log_probs, rollout, config.clip_epsilon).item())
        last_kl, last_surrogate = kl, surrogate
        if kl <= config.kl_budget * 1.05 and surrogate >= -1e-8:
            return True, alpha, kl, surrogate, trial + 1, calls
    set_parameter_vector(bundle, center)
    return False, 0.0, last_kl, last_surrogate, config.line_search_steps, calls


@torch.inference_mode()
def forward_policy_step(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    rollout: RolloutBatch,
    generator: torch.Generator,
    config: ForwardConfig,
    active_subspace: ActiveSubspace | None = None,
) -> ForwardStepResult:
    """Take one FO-PG/FO-NPG/FOCUS-NPG update from a stored rollout group."""

    if config.method not in {"fo_pg", "fo_npg", "focus_npg"}:
        raise ValueError(f"forward_policy_step does not implement {config.method!r}")
    set_adapter_grad_enabled(bundle, False)
    center = parameter_vector(bundle).float()
    active = None
    if config.method == "focus_npg" and active_subspace is not None:
        active = active_subspace.basis
    basis = make_search_basis(
        center.numel(), config.directions, generator, bundle.device, active_basis=active
    )
    scores, forward_calls = directional_log_probability_scores(
        bundle, encoded, center, basis, config.finite_difference_mu
    )
    gradient, fisher, action_scores = projected_policy_gradient(scores, rollout)
    coordinates = _solve_coordinates(gradient, fisher, config)
    parameter_direction = basis @ coordinates
    alpha = _initial_trust_region_scale(coordinates, fisher, config)
    accepted, accepted_alpha, kl, surrogate, trials, search_calls = _forward_line_search(
        bundle,
        encoded,
        center,
        parameter_direction,
        alpha,
        rollout,
        config,
    )
    forward_calls += search_calls
    if active_subspace is not None and config.method == "focus_npg":
        active_subspace.update(basis @ gradient)

    scalar_derivatives = (rollout.advantages.unsqueeze(-1) * action_scores).reshape(
        -1, action_scores.shape[-1]
    )
    derivative_variance = float(scalar_derivatives.var(dim=0, unbiased=False).mean().item())
    return ForwardStepResult(
        accepted=accepted,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=rollout.zero_advantage_fraction,
        empirical_kl=kl,
        surrogate_improvement=surrogate,
        step_norm=float(accepted_alpha * parameter_direction.norm().item()),
        projected_gradient_norm=float(gradient.norm().item()),
        fisher_condition=fisher_condition_number(fisher),
        line_search_trials=trials,
        forward_calls=forward_calls,
        backward_calls=0,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=forward_calls * rollout.batch_size,
        derivative_variance=derivative_variance,
    )


@torch.inference_mode()
def evolution_strategy_step(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    targets: Tensor,
    group_size: int,
    generator: torch.Generator,
    config: ForwardConfig,
) -> ForwardStepResult:
    """Antithetic adapter-space ES with common random numbers for +/- rollouts."""

    if config.method != "es":
        raise ValueError("evolution_strategy_step requires method='es'")
    set_adapter_grad_enabled(bundle, False)
    center = parameter_vector(bundle).float()
    basis = make_search_basis(center.numel(), config.directions, generator, bundle.device)
    uniforms = torch.rand(targets.shape[0], group_size, generator=generator, device=bundle.device)
    derivatives: list[Tensor] = []
    reward_means: list[float] = []
    old_log_probs = candidate_log_probs(bundle, encoded)
    forward_calls = 1
    try:
        for index in range(config.directions):
            direction = basis[:, index]
            set_parameter_vector(bundle, center + config.es_sigma * direction)
            plus_log_probs = candidate_log_probs(bundle, encoded)
            plus_actions = sample_with_uniforms(plus_log_probs, uniforms)
            plus_rewards = plus_actions.eq(targets[:, None]).float()
            set_parameter_vector(bundle, center - config.es_sigma * direction)
            minus_log_probs = candidate_log_probs(bundle, encoded)
            minus_actions = sample_with_uniforms(minus_log_probs, uniforms)
            minus_rewards = minus_actions.eq(targets[:, None]).float()
            derivatives.append(
                (plus_rewards.mean() - minus_rewards.mean()) / (2.0 * config.es_sigma)
            )
            reward_means.extend([float(plus_rewards.mean()), float(minus_rewards.mean())])
            forward_calls += 2
    finally:
        set_parameter_vector(bundle, center)

    gradient = torch.stack(derivatives)
    raw_direction = basis @ gradient
    raw_norm = float(raw_direction.norm().item())
    if raw_norm > 1e-12:
        direction = raw_direction / raw_norm
        initial_alpha = config.es_step_norm
    else:
        direction = raw_direction
        initial_alpha = 0.0

    # ES has no stored policy-gradient surrogate. Enforce the actual KL budget.
    accepted = False
    accepted_alpha = 0.0
    last_kl = 0.0
    trials = 0
    for trial in range(config.line_search_steps):
        trials = trial + 1
        alpha = initial_alpha * config.line_search_decay**trial
        set_parameter_vector(bundle, center + alpha * direction)
        new_log_probs = candidate_log_probs(bundle, encoded)
        forward_calls += 1
        last_kl = float(categorical_kl(old_log_probs, new_log_probs).item())
        if last_kl <= config.kl_budget * 1.05:
            accepted = True
            accepted_alpha = alpha
            break
    if not accepted:
        set_parameter_vector(bundle, center)

    environment_samples = 2 * config.directions * targets.shape[0] * group_size
    return ForwardStepResult(
        accepted=accepted,
        reward_mean=float(sum(reward_means) / max(len(reward_means), 1)),
        zero_advantage_fraction=float("nan"),
        empirical_kl=last_kl,
        surrogate_improvement=float("nan"),
        step_norm=accepted_alpha,
        projected_gradient_norm=float(gradient.norm().item()),
        fisher_condition=float("nan"),
        line_search_trials=trials,
        forward_calls=forward_calls,
        backward_calls=0,
        environment_samples=environment_samples,
        teacher_forced_examples=forward_calls * targets.shape[0],
        derivative_variance=float(gradient.var(unbiased=False).item()),
    )
