"""Shared rollout and numerical utilities for all optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

BaselineKind = Literal["loo", "group_zscore"]


@dataclass
class RolloutBatch:
    old_log_probs: Tensor
    actions: Tensor
    rewards: Tensor
    returns: Tensor
    advantages: Tensor
    targets: Tensor
    zero_advantage_fraction: float

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def group_size(self) -> int:
        return self.actions.shape[1]

    @property
    def environment_samples(self) -> int:
        return self.actions.numel()


def leave_one_out_advantages(returns: Tensor) -> Tensor:
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("leave-one-out advantages require [batch, group>=2]")
    group_size = returns.shape[1]
    return (group_size * returns - returns.sum(dim=1, keepdim=True)) / (group_size - 1)


def group_zscore_advantages(returns: Tensor, epsilon: float = 1e-6) -> Tensor:
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("group z-score advantages require [batch, group>=2]")
    centered = returns - returns.mean(dim=1, keepdim=True)
    scale = returns.std(dim=1, unbiased=False, keepdim=True)
    return centered / scale.clamp_min(epsilon)


def advantages_from_returns(returns: Tensor, baseline: BaselineKind) -> Tensor:
    if baseline == "loo":
        return leave_one_out_advantages(returns)
    if baseline == "group_zscore":
        return group_zscore_advantages(returns)
    raise ValueError(f"unknown baseline {baseline!r}")


def _sample_categorical(log_probs: Tensor, group_size: int, generator: torch.Generator) -> Tensor:
    return torch.multinomial(
        log_probs.exp(),
        num_samples=group_size,
        replacement=True,
        generator=generator,
    )


def sample_with_uniforms(log_probs: Tensor, uniforms: Tensor) -> Tensor:
    """Categorical inverse-CDF sampler, useful for common-random-number ES."""

    cumulative = log_probs.exp().cumsum(dim=-1)
    actions = (uniforms.unsqueeze(-1) > cumulative.unsqueeze(1)).sum(dim=-1)
    return actions.clamp_max(log_probs.shape[-1] - 1)


def gather_action_values(values: Tensor, actions: Tensor) -> Tensor:
    """Gather [batch, actions, ...] values at [batch, group] action indices."""

    if values.ndim == 2:
        return values.gather(1, actions)
    if values.ndim == 3:
        expanded = actions.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return values.gather(1, expanded)
    raise ValueError(f"expected rank-2 or rank-3 values, received shape {tuple(values.shape)}")


def build_rollout_batch(
    log_probs: Tensor,
    targets: Tensor,
    group_size: int,
    generator: torch.Generator,
    baseline: BaselineKind = "loo",
    reference_log_probs: Tensor | None = None,
    kl_beta: float = 0.0,
) -> RolloutBatch:
    actions = _sample_categorical(log_probs.detach(), group_size, generator)
    rewards = actions.eq(targets[:, None]).float()
    selected_log_probs = gather_action_values(log_probs.detach(), actions)
    returns = rewards
    if kl_beta:
        if reference_log_probs is None:
            raise ValueError("reference log probabilities are required when kl_beta is nonzero")
        reference_selected = gather_action_values(reference_log_probs, actions)
        returns = returns - kl_beta * (selected_log_probs - reference_selected)
    advantages = advantages_from_returns(returns, baseline)
    zero_groups = advantages.abs().amax(dim=1).eq(0)
    return RolloutBatch(
        old_log_probs=log_probs.detach(),
        actions=actions,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        targets=targets,
        zero_advantage_fraction=float(zero_groups.float().mean().item()),
    )


def categorical_kl(old_log_probs: Tensor, new_log_probs: Tensor) -> Tensor:
    return (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(dim=-1).mean()


def entropy(log_probs: Tensor) -> Tensor:
    return -(log_probs.exp() * log_probs).sum(dim=-1).mean()


def clipped_surrogate(
    new_log_probs: Tensor,
    rollout: RolloutBatch,
    clip_epsilon: float,
) -> Tensor:
    new_selected = gather_action_values(new_log_probs, rollout.actions)
    old_selected = gather_action_values(rollout.old_log_probs, rollout.actions)
    ratio = (new_selected - old_selected).exp()
    unclipped = ratio * rollout.advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * rollout.advantages
    return torch.minimum(unclipped, clipped).mean()


def make_search_basis(
    dimension: int,
    directions: int,
    generator: torch.Generator,
    device: torch.device,
    active_basis: Tensor | None = None,
) -> Tensor:
    """Combine learned columns with fresh Gaussian scouts and Euclidean-QR them."""

    prefix = torch.empty(dimension, 0, device=device)
    if active_basis is not None and active_basis.numel():
        prefix = active_basis[:, :directions].to(device=device, dtype=torch.float32)
    fresh_count = directions - prefix.shape[1]
    fresh = torch.randn(
        dimension,
        fresh_count,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    matrix = torch.cat([prefix, fresh], dim=1)
    return torch.linalg.qr(matrix, mode="reduced").Q


def fisher_condition_number(fisher: Tensor, epsilon: float = 1e-12) -> float:
    eigenvalues = torch.linalg.eigvalsh(fisher.float()).clamp_min(epsilon)
    return float((eigenvalues[-1] / eigenvalues[0]).item())
