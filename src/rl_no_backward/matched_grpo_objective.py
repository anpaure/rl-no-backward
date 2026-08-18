"""TRL-equivalent fixed-rollout objective shared by BP and forward-only LoRA.

The denominator of the PPO ratio is always the Hugging Face policy score at
rollout time.  vLLM's sampling log probability appears only in a detached
truncated-importance correction.  Keeping those quantities separate is
essential: backend numerical mismatch must not masquerade as policy drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

InferenceCorrectionMode = Literal["token_truncate", "token_mask", "sequence_mask"]


@dataclass(frozen=True, slots=True)
class MatchedGRPOObjectiveConfig:
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1.0e-4
    inference_correction_mode: InferenceCorrectionMode = "sequence_mask"
    inference_ratio_min: float | None = None
    inference_ratio_max: float | None = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.clip_epsilon) or self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive and finite")
        if not math.isfinite(self.advantage_epsilon) or self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive and finite")
        if self.inference_correction_mode not in {
            "token_truncate",
            "token_mask",
            "sequence_mask",
        }:
            raise ValueError("unsupported inference correction mode")
        for name, value in (
            ("inference_ratio_min", self.inference_ratio_min),
            ("inference_ratio_max", self.inference_ratio_max),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite or None")
        if (
            self.inference_ratio_min is not None
            and self.inference_ratio_max is not None
            and self.inference_ratio_min >= self.inference_ratio_max
        ):
            raise ValueError("inference_ratio_min must be smaller than inference_ratio_max")


def trl_group_standardized_advantages(rewards: Tensor, epsilon: float = 1.0e-4) -> Tensor:
    """Match TRL's group centering and sample-standard-deviation scaling."""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [batch, group>=2]")
    numeric = rewards.float()
    if not torch.isfinite(numeric).all():
        raise ValueError("rewards must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    centered = numeric - numeric.mean(dim=1, keepdim=True)
    # TRL's nanstd applies Bessel's correction, i.e. torch.std(correction=1).
    scale = numeric.std(dim=1, correction=1, keepdim=True)
    return centered / (scale + epsilon)


def detached_inference_correction(
    old_hf_token_log_probs: Tensor,
    sampler_token_log_probs: Tensor,
    response_mask: Tensor,
    config: MatchedGRPOObjectiveConfig,
) -> Tensor:
    """Return detached pi_old_HF / q_sampler weights with explicit bounds."""

    if old_hf_token_log_probs.shape != sampler_token_log_probs.shape:
        raise ValueError("HF and sampler log probabilities must have identical shape")
    if response_mask.shape != old_hf_token_log_probs.shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean with the log-probability shape")
    log_difference = (old_hf_token_log_probs.detach() - sampler_token_log_probs.detach()).float()
    log_difference = log_difference.masked_fill(~response_mask, 0.0)
    if config.inference_correction_mode == "sequence_mask":
        ratio = log_difference.sum(dim=-1, keepdim=True).exp()
    else:
        ratio = log_difference.exp()

    lower = config.inference_ratio_min
    upper = config.inference_ratio_max
    if config.inference_correction_mode == "token_truncate":
        minimum = lower if lower is not None else 0.0
        maximum = upper if upper is not None else torch.finfo(ratio.dtype).max
        ratio = ratio.clamp(min=minimum, max=maximum)
    else:
        valid = torch.ones_like(ratio, dtype=torch.bool)
        if lower is not None:
            valid &= ratio >= lower
        if upper is not None:
            valid &= ratio <= upper
        ratio = ratio.masked_fill(~valid, 0.0)
    return ratio.detach()


def matched_token_grpo_surrogate(
    new_hf_token_log_probs: Tensor,
    old_hf_token_log_probs: Tensor,
    sampler_token_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    config: MatchedGRPOObjectiveConfig,
) -> Tensor:
    """Return the equal-completion-weight token-local clipped GRPO objective."""

    expected = old_hf_token_log_probs.shape
    if new_hf_token_log_probs.shape != expected or sampler_token_log_probs.shape != expected:
        raise ValueError("all token log-probability tensors must have the same shape")
    if response_mask.shape != expected or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean with the log-probability shape")
    if advantages.shape != expected[:-1]:
        raise ValueError("advantages must match the leading completion dimensions")
    # This is the PPO ratio.  q_sampler is deliberately absent.
    policy_ratio = (new_hf_token_log_probs - old_hf_token_log_probs.detach()).exp()
    expanded_advantages = advantages.unsqueeze(-1).to(policy_ratio.dtype)
    unclipped = policy_ratio * expanded_advantages
    clipped = policy_ratio.clamp(
        1.0 - config.clip_epsilon,
        1.0 + config.clip_epsilon,
    ) * expanded_advantages
    correction = detached_inference_correction(
        old_hf_token_log_probs,
        sampler_token_log_probs,
        response_mask,
        config,
    )
    token_objective = torch.minimum(unclipped, clipped) * correction
    token_objective = token_objective.masked_fill(~response_mask, 0.0)
    lengths = response_mask.sum(dim=-1).clamp_min(1).to(token_objective.dtype)
    return (token_objective.sum(dim=-1) / lengths).mean()


def sampled_hf_policy_kl(
    new_hf_token_log_probs: Tensor,
    old_hf_token_log_probs: Tensor,
    response_mask: Tensor,
    sampling_weights: Tensor | None = None,
) -> Tensor:
    """Non-negative k3 estimate of KL(pi_old_HF || pi_new_HF)."""

    if new_hf_token_log_probs.shape != old_hf_token_log_probs.shape:
        raise ValueError("new and old HF log probabilities must have identical shape")
    if response_mask.shape != new_hf_token_log_probs.shape:
        raise ValueError("response_mask shape mismatch")
    log_ratio = new_hf_token_log_probs.float() - old_hf_token_log_probs.float()
    token_kl = (torch.expm1(log_ratio) - log_ratio).masked_fill(~response_mask, 0.0)
    if sampling_weights is not None:
        if sampling_weights.shape == response_mask.shape[:-1] + (1,):
            sampling_weights = sampling_weights.expand_as(response_mask)
        elif sampling_weights.shape != response_mask.shape:
            raise ValueError("sampling_weights must be per-completion or per-token")
        weights = sampling_weights.to(token_kl.dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sampling_weights must be finite and non-negative")
        token_kl = token_kl * weights
    lengths = response_mask.sum(dim=-1).clamp_min(1).to(token_kl.dtype)
    return (token_kl.sum(dim=-1) / lengths).mean()


__all__ = [
    "InferenceCorrectionMode",
    "MatchedGRPOObjectiveConfig",
    "detached_inference_correction",
    "matched_token_grpo_surrogate",
    "sampled_hf_policy_kl",
    "trl_group_standardized_advantages",
]
