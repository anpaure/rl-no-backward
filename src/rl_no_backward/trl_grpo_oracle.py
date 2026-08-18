"""Thin TRL 1.10 fixed-rollout loss oracle for matched-GRPO certification.

This module deliberately imports TRL lazily.  Production training does not
depend on the oracle at runtime; the H100 release gate uses it to compare the
custom streaming loss, its parameter gradient, and a resulting update against
the installed upstream ``GRPOTrainer._compute_loss`` implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from importlib import metadata
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    matched_token_grpo_surrogate,
    trl_group_standardized_advantages,
)


class _SingleProcessAccelerator:
    num_processes = 1
    sync_gradients = True

    @staticmethod
    def gather(value: Tensor) -> Tensor:
        return value

    @staticmethod
    def reduce(value: Tensor, *, reduction: str) -> Tensor:
        if reduction != "sum":
            raise ValueError("the fixed-rollout oracle only supports sum reduction")
        return value


def _trl_inference_ratio(
    old_hf: Tensor,
    sampler: Tensor,
    mask: Tensor,
    config: MatchedGRPOObjectiveConfig,
) -> Tensor:
    """Reproduce TRL 1.10's generation-time vLLM correction independently."""

    difference = (old_hf.detach() - sampler.detach()).float() * mask
    if config.inference_correction_mode == "sequence_mask":
        ratio = difference.sum(dim=-1, keepdim=True).exp()
    else:
        ratio = difference.exp()
    if config.inference_correction_mode == "token_truncate":
        lower = config.inference_ratio_min if config.inference_ratio_min is not None else 0.0
        upper = (
            config.inference_ratio_max
            if config.inference_ratio_max is not None
            else torch.finfo(ratio.dtype).max
        )
        return ratio.clamp(min=lower, max=upper).detach()
    lower = config.inference_ratio_min if config.inference_ratio_min is not None else -math.inf
    upper = config.inference_ratio_max if config.inference_ratio_max is not None else math.inf
    return ratio.masked_fill((ratio < lower) | (ratio > upper), 0.0).detach()


def trl_110_fixed_rollout_loss(
    new_hf_token_log_probs: Tensor,
    old_hf_token_log_probs: Tensor,
    sampler_token_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    config: MatchedGRPOObjectiveConfig,
) -> Tensor:
    """Evaluate the exact tensors through upstream TRL 1.10 ``_compute_loss``.

    The adapter fixes model scoring to the caller-supplied differentiable
    ``new_hf_token_log_probs`` while leaving PPO clipping, vLLM inference
    correction, completion normalization, and reduction inside official TRL.
    Entropy, reference-KL, auxiliary loss, and gradient-accumulation scaling
    are deliberately disabled because the matched headline objective does not
    include them.  TRL ``loss_type='grpo'`` gives equal weight to each
    completion after completion-length normalization, matching this project.
    """

    try:
        from trl import GRPOTrainer
    except ImportError as error:  # pragma: no cover - optional local dependency
        raise RuntimeError("the certification gate requires trl==1.10.*") from error
    installed = metadata.version("trl")
    if not installed.startswith("1.10."):
        raise RuntimeError(f"the fixed-rollout oracle is pinned to TRL 1.10.*, found {installed}")
    expected = old_hf_token_log_probs.shape
    if (
        new_hf_token_log_probs.shape != expected
        or sampler_token_log_probs.shape != expected
        or response_mask.shape != expected
        or advantages.shape != expected[:-1]
    ):
        raise ValueError("fixed-rollout oracle tensor shapes do not match")
    if response_mask.dtype != torch.bool:
        raise TypeError("response_mask must be boolean")

    token_count = expected[-1]
    completion_count = math.prod(expected[:-1])
    new_flat = new_hf_token_log_probs.reshape(completion_count, token_count)
    old_flat = old_hf_token_log_probs.reshape(completion_count, token_count).detach()
    sampler_flat = sampler_token_log_probs.reshape(completion_count, token_count).detach()
    mask_flat = response_mask.reshape(completion_count, token_count)
    advantage_flat = advantages.reshape(completion_count)
    inference_ratio = _trl_inference_ratio(
        old_flat,
        sampler_flat,
        mask_flat,
        config,
    )

    trainer = object.__new__(GRPOTrainer)
    trainer.model = nn.Identity().train()
    trainer.top_entropy_quantile = 1.0
    trainer.aux_loss_enabled = False
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.beta = 0.0
    trainer.loss_type = "grpo"
    trainer.epsilon_low = config.clip_epsilon
    trainer.epsilon_high = config.clip_epsilon
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_correction = True
    trainer._entropy_bonus_enabled = False
    trainer.current_gradient_accumulation_steps = 1
    trainer.args = SimpleNamespace(delta=None, use_bias_correction_kl=False)
    trainer.accelerator = _SingleProcessAccelerator()
    trainer._metrics = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }

    def fixed_scores(
        _self: Any,
        _model: nn.Module,
        _input_ids: Tensor,
        _attention_mask: Tensor,
        _logits_to_keep: int,
        **_kwargs: Any,
    ) -> tuple[Tensor, Tensor, None]:
        return new_flat, torch.zeros_like(new_flat), None

    trainer._get_per_token_logps_and_entropies = MethodType(fixed_scores, trainer)
    inputs = {
        "prompt_ids": torch.zeros(
            completion_count,
            1,
            dtype=torch.long,
            device=new_flat.device,
        ),
        "prompt_mask": torch.ones(
            completion_count,
            1,
            dtype=torch.bool,
            device=new_flat.device,
        ),
        "completion_ids": torch.zeros(
            completion_count,
            token_count,
            dtype=torch.long,
            device=new_flat.device,
        ),
        "completion_mask": mask_flat,
        "advantages": advantage_flat,
        "old_per_token_logps": old_flat,
        "sampling_per_token_logps": sampler_flat,
        "importance_sampling_ratio": inference_ratio,
    }
    return trainer._compute_loss(trainer.model, inputs)


def run_trl_110_differential_certification() -> dict[str, Any]:
    """Certify loss, gradient, and one scalar update against installed TRL."""

    config = MatchedGRPOObjectiveConfig(
        clip_epsilon=0.2,
        advantage_epsilon=1.0e-4,
        inference_correction_mode="sequence_mask",
        inference_ratio_min=None,
        inference_ratio_max=3.0,
    )
    rewards = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5]], dtype=torch.float64)
    advantages = trl_group_standardized_advantages(rewards, config.advantage_epsilon).double()
    old = -torch.linspace(0.4, 1.5, 24, dtype=torch.float64).reshape(2, 3, 4)
    sampler = old - torch.linspace(-0.02, 0.04, 24, dtype=torch.float64).reshape_as(old)
    mask = torch.tensor(
        [
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
        ],
        dtype=torch.bool,
    )
    feature = torch.linspace(-0.7, 0.9, old.numel(), dtype=torch.float64).reshape_as(old)
    native_parameter = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    trl_parameter = native_parameter.detach().clone().requires_grad_(True)
    native_loss = -matched_token_grpo_surrogate(
        old + native_parameter * feature,
        old,
        sampler,
        advantages,
        mask,
        config,
    )
    trl_loss = trl_110_fixed_rollout_loss(
        old + trl_parameter * feature,
        old,
        sampler,
        advantages,
        mask,
        config,
    )
    native_gradient = torch.autograd.grad(native_loss, native_parameter)[0]
    trl_gradient = torch.autograd.grad(trl_loss, trl_parameter)[0]
    learning_rate = 3.0e-4
    native_update = native_parameter.detach() - learning_rate * native_gradient
    trl_update = trl_parameter.detach() - learning_rate * trl_gradient
    loss_delta = float((native_loss - trl_loss).abs().item())
    gradient_delta = float((native_gradient - trl_gradient).abs().item())
    update_delta = float((native_update - trl_update).abs().item())
    tolerance = 1.0e-12
    return {
        "schema": "rl-no-backward-trl-1.10-differential-v1",
        "passed": max(loss_delta, gradient_delta, update_delta) <= tolerance,
        "trl_version": metadata.version("trl"),
        "objective_settings": {
            "loss_type": "grpo",
            "importance_sampling_level": "token",
            "vllm_importance_sampling_mode": "sequence_mask",
            "clip_epsilon": config.clip_epsilon,
            "inference_ratio_max": config.inference_ratio_max,
            "completion_aggregation": "equal completion-length-normalized mean",
            "gradient_accumulation_normalizer": 1,
        },
        "native_loss": float(native_loss.item()),
        "trl_loss": float(trl_loss.item()),
        "native_gradient": float(native_gradient.item()),
        "trl_gradient": float(trl_gradient.item()),
        "loss_abs_difference": loss_delta,
        "gradient_abs_difference": gradient_delta,
        "update_abs_difference": update_delta,
        "absolute_tolerance": tolerance,
    }


__all__ = ["run_trl_110_differential_certification", "trl_110_fixed_rollout_loss"]
