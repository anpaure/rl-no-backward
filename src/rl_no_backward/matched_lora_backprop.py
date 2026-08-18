"""Streaming reverse-mode update for the shared matched-LoRA GRPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
    matched_token_grpo_surrogate,
    sampled_hf_policy_kl,
)
from .matched_lora_configs import MatchedBackpropConfig
from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled
from .sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs


@dataclass(frozen=True, slots=True)
class MatchedBackpropStepResult:
    accepted: bool
    reward_mean: float
    zero_advantage_fraction: float
    empirical_kl: float
    surrogate_improvement: float
    step_norm: float
    projected_gradient_norm: float
    policy_evaluations: int
    forward_calls: int
    backward_calls: int
    environment_samples: int
    teacher_forced_examples: int
    scored_tokens: int


def _slice_prompt_groups(
    rollout: SequenceRolloutBatch,
    start: int,
    end: int,
) -> SequenceRolloutBatch:
    return replace(
        rollout,
        prompts=rollout.prompts[start:end],
        completions=rollout.completions[start:end],
        prompt_input_ids=rollout.prompt_input_ids[start:end],
        prompt_attention_mask=rollout.prompt_attention_mask[start:end],
        response_input_ids=rollout.response_input_ids[start:end],
        response_mask=rollout.response_mask[start:end],
        old_token_log_probs=rollout.old_token_log_probs[start:end],
        rewards=rollout.rewards[start:end],
        advantages=rollout.advantages[start:end],
        frozen_prefix_cache=None,
        frozen_prefix_fallback_reason=None,
    )


def matched_grpo_streaming_backward(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    objective_config: MatchedGRPOObjectiveConfig,
    *,
    prompt_groups_per_micro_batch: int,
) -> tuple[float, int]:
    """Accumulate the exact global objective gradient one prompt group at a time."""

    if rollout.frozen_prefix_cache is not None:
        raise ValueError("all-layer LoRA is incompatible with frozen-prefix scoring")
    if sampler_token_log_probs.shape != rollout.old_token_log_probs.shape:
        raise ValueError("sampler log-probability shape mismatch")
    if prompt_groups_per_micro_batch < 1:
        raise ValueError("prompt_groups_per_micro_batch must be positive")
    batch_size = rollout.batch_size
    objective_sum = 0.0
    calls = 0
    for start in range(0, batch_size, prompt_groups_per_micro_batch):
        end = min(start + prompt_groups_per_micro_batch, batch_size)
        chunk = _slice_prompt_groups(rollout, start, end)
        new_logps = teacher_forced_token_log_probs(bundle, chunk)
        chunk_objective = matched_token_grpo_surrogate(
            new_logps,
            chunk.old_token_log_probs,
            sampler_token_log_probs[start:end],
            chunk.advantages,
            chunk.response_mask,
            objective_config,
        )
        weight = (end - start) / batch_size
        (-chunk_objective * weight).backward()
        objective_sum += float(chunk_objective.detach().item()) * weight
        calls += 1
    return objective_sum, calls


def make_matched_lora_optimizer(
    bundle: ModelBundle,
    config: MatchedBackpropConfig,
) -> torch.optim.Optimizer:
    set_adapter_grad_enabled(bundle, True)
    actual_trainable = {name for name, parameter in bundle.model.named_parameters() if parameter.requires_grad}
    if actual_trainable != set(bundle.adapter_names):
        raise RuntimeError(
            "reverse-mode trainable parameter set does not exactly match the locked LoRA layout"
        )
    return torch.optim.AdamW(
        bundle.trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=config.fused_adamw and bundle.device.type == "cuda",
    )


def make_matched_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: MatchedBackpropConfig,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup_steps = round(total_steps * config.warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if config.scheduler_type == "constant":
            return 1.0
        if config.scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def matched_backprop_grpo_step(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    optimizer: torch.optim.Optimizer,
    objective_config: MatchedGRPOObjectiveConfig,
    config: MatchedBackpropConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> MatchedBackpropStepResult:
    """Apply one AdamW update to the standard PEFT LoRA tensors only."""

    bundle.model.eval()
    set_adapter_grad_enabled(bundle, True)
    before = parameter_vector(bundle).float().clone()
    optimizer.zero_grad(set_to_none=True)
    initial_objective, backward_calls = matched_grpo_streaming_backward(
        bundle,
        rollout,
        sampler_token_log_probs,
        objective_config,
        prompt_groups_per_micro_batch=config.prompt_groups_per_micro_batch,
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters, config.max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    after = parameter_vector(bundle).float()

    final_objective_sum = 0.0
    final_kl_sum = 0.0
    final_calls = 0
    with torch.inference_mode():
        for start in range(0, rollout.batch_size, config.prompt_groups_per_micro_batch):
            end = min(start + config.prompt_groups_per_micro_batch, rollout.batch_size)
            chunk = _slice_prompt_groups(rollout, start, end)
            final_logps = teacher_forced_token_log_probs(bundle, chunk)
            weight = (end - start) / rollout.batch_size
            final_objective_sum += float(
                matched_token_grpo_surrogate(
                    final_logps,
                    chunk.old_token_log_probs,
                    sampler_token_log_probs[start:end],
                    chunk.advantages,
                    chunk.response_mask,
                    objective_config,
                ).item()
            ) * weight
            final_kl_sum += float(
                sampled_hf_policy_kl(
                    final_logps,
                    chunk.old_token_log_probs,
                    chunk.response_mask,
                    sampling_weights=detached_inference_correction(
                        chunk.old_token_log_probs,
                        sampler_token_log_probs[start:end],
                        chunk.response_mask,
                        objective_config,
                    ),
                ).item()
            ) * weight
            final_calls += 1
    zero_groups = rollout.advantages.abs().amax(dim=1).eq(0)
    forward_calls = backward_calls + final_calls
    return MatchedBackpropStepResult(
        accepted=True,
        reward_mean=float(rollout.rewards.mean().item()),
        zero_advantage_fraction=float(zero_groups.float().mean().item()),
        empirical_kl=final_kl_sum,
        surrogate_improvement=final_objective_sum - initial_objective,
        step_norm=float((after - before).norm().item()),
        projected_gradient_norm=float(grad_norm.item()),
        policy_evaluations=2,
        forward_calls=forward_calls,
        backward_calls=backward_calls,
        environment_samples=rollout.environment_samples,
        teacher_forced_examples=2 * rollout.environment_samples,
        scored_tokens=2 * rollout.valid_response_tokens,
    )


__all__ = [
    "MatchedBackpropConfig",
    "MatchedBackpropStepResult",
    "make_matched_lora_optimizer",
    "make_matched_lr_scheduler",
    "matched_backprop_grpo_step",
    "matched_grpo_streaming_backward",
]
