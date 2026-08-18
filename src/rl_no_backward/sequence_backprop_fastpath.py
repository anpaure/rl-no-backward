"""Memory-bounded GRPO backward pass for larger sequence batches.

The existing generic scorer micro-batches model forwards but concatenates all
differentiable outputs before one backward call.  That preserves the whole
autograd graph and therefore does not bound training activation memory.  This
opt-in helper backpropagates each separable chunk of the exact GRPO objective
immediately, accumulating the same adapter gradient while releasing each
chunk's graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .frozen_prefix import qwen_teacher_forcing_logits_to_keep, replay_frozen_suffix_logits
from .model import ModelBundle
from .sequence_fastpath import selected_token_log_probs
from .sequence_policy import SequenceRolloutBatch


@dataclass(frozen=True, slots=True)
class StreamingBackwardResult:
    surrogate: float
    model_calls: int
    full_prefix_calls: int
    suffix_calls: int


def streaming_grpo_backward(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    *,
    clip_epsilon: float,
    micro_batch_size: int,
    temperature: float | None = None,
) -> StreamingBackwardResult:
    """Accumulate the exact clipped-GRPO gradient one example chunk at a time.

    The caller owns ``optimizer.zero_grad`` and ``optimizer.step``.  Every
    completion still has weight ``1 / (B*G)`` and every token has weight
    ``1 / response_length``, so changing ``micro_batch_size`` changes memory
    and floating-point summation order only—not the objective.
    """

    if clip_epsilon <= 0 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be positive and finite")
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size < 1
    ):
        raise ValueError("micro_batch_size must be a positive integer")
    target_temperature = (
        rollout.sampling_temperature if temperature is None else float(temperature)
    )
    if target_temperature <= 0 or not math.isfinite(target_temperature):
        raise ValueError("temperature must be positive and finite")

    flat_input_ids = rollout.flat_input_ids
    flat_attention_mask = rollout.flat_attention_mask
    flat_targets = rollout.flat_response_input_ids
    flat_mask = rollout.flat_response_mask
    flat_old = rollout.old_token_log_probs.reshape(
        rollout.environment_samples, rollout.max_response_length
    )
    flat_advantages = rollout.advantages.reshape(rollout.environment_samples)
    flat_lengths = rollout.response_lengths.reshape(rollout.environment_samples).clamp_min(1)
    prompt_width = rollout.prompt_input_ids.shape[1]
    response_width = rollout.max_response_length
    example_count = rollout.environment_samples

    surrogate_sum = torch.zeros((), device=bundle.device, dtype=torch.float32)
    model_calls = 0
    logits_to_keep = qwen_teacher_forcing_logits_to_keep(
        bundle,
        prompt_width,
        response_width,
    )
    for start in range(0, example_count, micro_batch_size):
        stop = min(start + micro_batch_size, example_count)
        if rollout.frozen_prefix_cache is None:
            model_kwargs: dict[str, object] = {
                "input_ids": flat_input_ids[start:stop],
                "attention_mask": flat_attention_mask[start:stop],
                "use_cache": False,
            }
            if logits_to_keep is not None:
                model_kwargs["logits_to_keep"] = logits_to_keep
            outputs = bundle.model(
                **model_kwargs,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        else:
            logits = replay_frozen_suffix_logits(
                bundle,
                rollout.frozen_prefix_cache,
                start=start,
                stop=stop,
                logits_to_keep=logits_to_keep,
            )
        response_logits = (
            logits
            if logits_to_keep is not None
            else logits[:, prompt_width - 1 : prompt_width + response_width - 1, :]
        )
        token_log_probs = selected_token_log_probs(
            response_logits,
            flat_targets[start:stop],
            target_temperature,
        )

        log_ratios = token_log_probs - flat_old[start:stop]
        ratios = log_ratios.exp()
        advantages = flat_advantages[start:stop, None]
        unclipped = ratios * advantages
        clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        token_surrogate = torch.minimum(unclipped, clipped).masked_fill(
            ~flat_mask[start:stop], 0.0
        )
        completion_surrogate = token_surrogate.sum(dim=-1) / flat_lengths[
            start:stop
        ].to(token_surrogate.dtype)
        chunk_sum = completion_surrogate.sum()
        (-chunk_sum / example_count).backward()
        surrogate_sum += chunk_sum.detach()
        model_calls += 1

    return StreamingBackwardResult(
        surrogate=float((surrogate_sum / example_count).item()),
        model_calls=model_calls,
        full_prefix_calls=model_calls if rollout.frozen_prefix_cache is None else 0,
        # A full-model call also executes the adapted suffix.  These counters
        # measure stage executions, not mutually exclusive call categories.
        suffix_calls=model_calls,
    )


__all__ = ["StreamingBackwardResult", "streaming_grpo_backward"]
