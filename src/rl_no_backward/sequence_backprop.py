"""Conventional reverse-mode GRPO for fixed autoregressive rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled
from .sequence_policy import (
    SequenceRolloutBatch,
    clipped_grpo_surrogate,
    teacher_forced_token_log_probs,
)


@dataclass(frozen=True, slots=True)
class BackpropSequenceConfig:
    """AdamW settings for the reverse-mode sequence GRPO baseline."""

    learning_rate: float = 0.01
    weight_decay: float = 0.0
    clip_epsilon: float = 0.2
    epochs_per_rollout: int = 2
    max_grad_norm: float = 1.0
    scoring_micro_batch_size: int | None = None

    def __post_init__(self) -> None:
        for name, value, allow_zero in (
            ("learning_rate", self.learning_rate, False),
            ("weight_decay", self.weight_decay, True),
            ("clip_epsilon", self.clip_epsilon, False),
            ("max_grad_norm", self.max_grad_norm, False),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            lower_bound_ok = value >= 0 if allow_zero else value > 0
            if not lower_bound_ok or not math.isfinite(float(value)):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier} and finite")
        if (
            isinstance(self.epochs_per_rollout, bool)
            or not isinstance(self.epochs_per_rollout, int)
            or self.epochs_per_rollout < 1
        ):
            raise ValueError("epochs_per_rollout must be a positive integer")
        if self.scoring_micro_batch_size is not None and (
            isinstance(self.scoring_micro_batch_size, bool)
            or not isinstance(self.scoring_micro_batch_size, int)
            or self.scoring_micro_batch_size < 1
        ):
            raise ValueError("scoring_micro_batch_size must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class BackpropSequenceStepResult:
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


def _micro_batches_per_evaluation(
    rollout: SequenceRolloutBatch, micro_batch_size: int | None
) -> int:
    if micro_batch_size is None:
        return 1
    return math.ceil(rollout.environment_samples / micro_batch_size)


def _sampled_sequence_kl(new_token_log_probs: Tensor, rollout: SequenceRolloutBatch) -> Tensor:
    """Non-negative k3 estimator of KL(old policy || new policy)."""

    log_ratio = new_token_log_probs.float() - rollout.old_token_log_probs.float()
    token_kl = torch.expm1(log_ratio) - log_ratio
    token_kl = token_kl.masked_fill(~rollout.response_mask, 0.0)
    lengths = rollout.response_lengths.clamp_min(1).to(token_kl.dtype)
    return (token_kl.sum(dim=-1) / lengths).mean()


def make_sequence_grpo_optimizer(
    bundle: ModelBundle, config: BackpropSequenceConfig
) -> torch.optim.Optimizer:
    """Create AdamW over adapter coefficients only."""

    set_adapter_grad_enabled(bundle, True)
    if not bundle.trainable_parameters:
        raise ValueError("sequence GRPO requires at least one adapter parameter")
    return torch.optim.AdamW(
        bundle.trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def sequence_grpo_step(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    optimizer: torch.optim.Optimizer,
    config: BackpropSequenceConfig,
) -> BackpropSequenceStepResult:
    """Run one ordinary reverse-mode clipped-GRPO update on fixed candidates."""

    bundle.model.eval()
    set_adapter_grad_enabled(bundle, True)
    before = parameter_vector(bundle).float().clone()
    with torch.no_grad():
        initial_surrogate = float(
            clipped_grpo_surrogate(rollout.old_token_log_probs, rollout, config.clip_epsilon).item()
        )

    gradient_norm = 0.0
    for _ in range(config.epochs_per_rollout):
        optimizer.zero_grad(set_to_none=True)
        token_log_probs = teacher_forced_token_log_probs(
            bundle,
            rollout,
            micro_batch_size=config.scoring_micro_batch_size,
        )
        surrogate = clipped_grpo_surrogate(token_log_probs, rollout, config.clip_epsilon)
        (-surrogate).backward()
        norm = torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters, config.max_grad_norm)
        gradient_norm = float(norm.item())
        optimizer.step()

    with torch.inference_mode():
        final_token_log_probs = teacher_forced_token_log_probs(
            bundle,
            rollout,
            micro_batch_size=config.scoring_micro_batch_size,
        )
        empirical_kl = float(_sampled_sequence_kl(final_token_log_probs, rollout).item())
        final_surrogate = float(
            clipped_grpo_surrogate(final_token_log_probs, rollout, config.clip_epsilon).item()
        )
        after = parameter_vector(bundle).float()

    policy_evaluations = config.epochs_per_rollout + 1
    forward_calls = policy_evaluations * _micro_batches_per_evaluation(
        rollout, config.scoring_micro_batch_size
    )
    return BackpropSequenceStepResult(
        accepted=True,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=rollout.zero_advantage_fraction,
        empirical_kl=empirical_kl,
        surrogate_improvement=final_surrogate - initial_surrogate,
        step_norm=float((after - before).norm().item()),
        projected_gradient_norm=gradient_norm,
        fisher_condition=float("nan"),
        line_search_trials=0,
        policy_evaluations=policy_evaluations,
        forward_calls=forward_calls,
        backward_calls=config.epochs_per_rollout,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=policy_evaluations * rollout.environment_samples,
        scored_tokens=policy_evaluations * rollout.valid_response_tokens,
        derivative_variance=float("nan"),
    )


__all__ = [
    "BackpropSequenceConfig",
    "BackpropSequenceStepResult",
    "make_sequence_grpo_optimizer",
    "sequence_grpo_step",
]
