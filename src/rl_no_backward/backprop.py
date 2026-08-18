"""Conventional reverse-mode GRPO baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .common import RolloutBatch, categorical_kl, clipped_surrogate, entropy
from .model import ModelBundle, candidate_log_probs, set_adapter_grad_enabled


@dataclass
class BackpropConfig:
    learning_rate: float = 0.03
    weight_decay: float = 0.0
    clip_epsilon: float = 0.2
    epochs_per_rollout: int = 2
    max_grad_norm: float = 1.0
    reference_kl_beta: float = 0.0


@dataclass
class BackpropStepResult:
    accepted: bool
    reward_mean: float
    zero_advantage_fraction: float
    empirical_kl: float
    reference_kl: float
    entropy: float
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


def make_grpo_optimizer(bundle: ModelBundle, config: BackpropConfig) -> torch.optim.Optimizer:
    set_adapter_grad_enabled(bundle, True)
    return torch.optim.AdamW(
        bundle.trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def grpo_step(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    rollout: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    config: BackpropConfig,
    reference_log_probs: Tensor | None = None,
) -> BackpropStepResult:
    """Optimize the clipped group-relative surrogate with ordinary backprop."""

    set_adapter_grad_enabled(bundle, True)
    before = torch.cat([p.detach().reshape(-1) for p in bundle.trainable_parameters])
    gradient_norm = 0.0
    for _ in range(config.epochs_per_rollout):
        optimizer.zero_grad(set_to_none=True)
        log_probs = candidate_log_probs(bundle, encoded)
        surrogate = clipped_surrogate(log_probs, rollout, config.clip_epsilon)
        loss = -surrogate
        if config.reference_kl_beta:
            if reference_log_probs is None:
                raise ValueError("reference log probabilities required for KL regularization")
            loss = loss + config.reference_kl_beta * categorical_kl(log_probs, reference_log_probs)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters, config.max_grad_norm)
        gradient_norm = float(norm.item())
        optimizer.step()
    with torch.inference_mode():
        final_log_probs = candidate_log_probs(bundle, encoded)
        empirical_kl = float(categorical_kl(rollout.old_log_probs, final_log_probs).item())
        reference_kl = (
            float(categorical_kl(final_log_probs, reference_log_probs).item())
            if reference_log_probs is not None
            else float("nan")
        )
        policy_entropy = float(entropy(final_log_probs).item())
        final_surrogate = float(
            clipped_surrogate(final_log_probs, rollout, config.clip_epsilon).item()
        )
        after = torch.cat([p.detach().reshape(-1) for p in bundle.trainable_parameters])

    forward_calls = config.epochs_per_rollout + 1
    return BackpropStepResult(
        accepted=True,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=rollout.zero_advantage_fraction,
        empirical_kl=empirical_kl,
        reference_kl=reference_kl,
        entropy=policy_entropy,
        surrogate_improvement=final_surrogate,
        step_norm=float((after - before).norm().item()),
        projected_gradient_norm=gradient_norm,
        fisher_condition=float("nan"),
        line_search_trials=0,
        forward_calls=forward_calls,
        backward_calls=config.epochs_per_rollout,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=forward_calls * rollout.batch_size,
        derivative_variance=float("nan"),
    )
