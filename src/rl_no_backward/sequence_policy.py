"""Autoregressive rollout utilities for exact-reward sequence RL.

The module deliberately separates stochastic on-policy generation from
teacher-forced evaluation.  A :class:`SequenceRolloutBatch` therefore remains
valid while adapter parameters are perturbed: its candidate token sequences,
old-policy token log probabilities, rewards, and group-relative advantages are
all fixed, while :func:`teacher_forced_token_log_probs` evaluates those same
tokens under the model's current parameters.

No function in this module changes ``requires_grad`` or invokes backward.
Teacher-forced rescoring is differentiable when its caller enables autograd and
is an ordinary inference-only forward pass under ``torch.inference_mode()``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .model import ModelBundle

if TYPE_CHECKING:
    from .frozen_prefix import FrozenPrefixCache


@dataclass(frozen=True, slots=True)
class CompletionSample:
    """One generated completion passed to an exact reward callback."""

    prompt_index: int
    group_index: int
    prompt: str
    completion: str
    response_token_ids: tuple[int, ...]


RewardCallback = Callable[[CompletionSample], float]


@dataclass(frozen=True, slots=True)
class ProjectedScoreStatistics:
    """Projected sequence-score quantities for a forward-only policy update.

    ``directional_token_scores`` has shape ``[B, G, T, D]`` and
    ``completion_scores`` has shape ``[B, G, D]``.  ``gradient`` and ``fisher``
    are the empirical REINFORCE gradient and empirical Fisher in the ``D``
    probed adapter directions.
    """

    directional_token_scores: Tensor
    completion_scores: Tensor
    gradient: Tensor
    fisher: Tensor


@dataclass(frozen=True, slots=True)
class SequenceRolloutBatch:
    """A prompt-major group of fixed autoregressive candidate sequences.

    Tensor shapes are:

    * prompt tensors: ``[B, P]``
    * response tensors and old token log probabilities: ``[B, G, T]``
    * rewards and leave-one-out advantages: ``[B, G]``

    Prompt padding is on the left.  Response masks include the first EOS token
    and exclude every position after it.  Invalid response token log
    probabilities are stored as zero so masked sums are safe.
    """

    prompts: tuple[str, ...]
    completions: tuple[tuple[str, ...], ...]
    prompt_input_ids: Tensor
    prompt_attention_mask: Tensor
    response_input_ids: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor
    rewards: Tensor
    advantages: Tensor
    sampling_temperature: float
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    frozen_prefix_cache: FrozenPrefixCache | None = None
    frozen_prefix_fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.prompt_input_ids.ndim != 2:
            raise ValueError("prompt_input_ids must have shape [B, P]")
        if self.response_input_ids.ndim != 3:
            raise ValueError("response_input_ids must have shape [B, G, T]")
        batch_size, group_size, response_width = self.response_input_ids.shape
        prompt_shape = self.prompt_input_ids.shape
        if batch_size == 0 or prompt_shape[1] == 0 or response_width == 0:
            raise ValueError("rollouts require non-empty prompts and responses")
        if group_size < 2:
            raise ValueError("leave-one-out rollouts require group_size >= 2")
        if prompt_shape[0] != batch_size or len(self.prompts) != batch_size:
            raise ValueError("prompt count does not match response batch size")
        if self.prompt_attention_mask.shape != prompt_shape:
            raise ValueError("prompt_attention_mask shape must match prompt_input_ids")
        response_shape = self.response_input_ids.shape
        for name, tensor in (
            ("response_mask", self.response_mask),
            ("old_token_log_probs", self.old_token_log_probs),
        ):
            if tensor.shape != response_shape:
                raise ValueError(f"{name} shape must match response_input_ids")
        for name, tensor in (("rewards", self.rewards), ("advantages", self.advantages)):
            if tensor.shape != (batch_size, group_size):
                raise ValueError(f"{name} must have shape [B, G]")
        if len(self.completions) != batch_size or any(
            len(group) != group_size for group in self.completions
        ):
            raise ValueError("completion text must have shape [B][G]")
        devices = {
            tensor.device
            for tensor in (
                self.prompt_input_ids,
                self.prompt_attention_mask,
                self.response_input_ids,
                self.response_mask,
                self.old_token_log_probs,
                self.rewards,
                self.advantages,
            )
        }
        if len(devices) != 1:
            raise ValueError("all rollout tensors must be on one device")
        if self.prompt_attention_mask.dtype != torch.bool or self.response_mask.dtype != torch.bool:
            raise TypeError("attention masks must have boolean dtype")
        if not self.old_token_log_probs.is_floating_point():
            raise TypeError("old_token_log_probs must be floating point")
        if self.old_token_log_probs.requires_grad:
            raise ValueError("old_token_log_probs must be detached from autograd")
        if not torch.isfinite(self.rewards).all() or not torch.isfinite(self.advantages).all():
            raise ValueError("rewards and advantages must be finite")
        if self.sampling_temperature <= 0 or not math.isfinite(self.sampling_temperature):
            raise ValueError("sampling_temperature must be positive and finite")
        if self.frozen_prefix_fallback_reason is not None and (
            not isinstance(self.frozen_prefix_fallback_reason, str)
            or not self.frozen_prefix_fallback_reason.strip()
        ):
            raise ValueError("frozen_prefix_fallback_reason must be a non-empty string or None")
        if self.frozen_prefix_cache is not None:
            if self.frozen_prefix_fallback_reason is not None:
                raise ValueError("an active frozen-prefix cache cannot also have a fallback reason")
            if self.frozen_prefix_cache.batch_size != batch_size * group_size:
                raise ValueError("frozen-prefix cache batch size does not match the rollout")
            if self.frozen_prefix_cache.hidden_states.device not in devices:
                raise ValueError("frozen-prefix cache and rollout must be on the same device")

    @property
    def batch_size(self) -> int:
        return self.response_input_ids.shape[0]

    @property
    def group_size(self) -> int:
        return self.response_input_ids.shape[1]

    @property
    def max_response_length(self) -> int:
        return self.response_input_ids.shape[2]

    @property
    def environment_samples(self) -> int:
        return self.batch_size * self.group_size

    @property
    def valid_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())

    @property
    def response_lengths(self) -> Tensor:
        return self.response_mask.sum(dim=-1)

    @property
    def flat_response_input_ids(self) -> Tensor:
        return self.response_input_ids.reshape(-1, self.max_response_length)

    @property
    def flat_response_mask(self) -> Tensor:
        return self.response_mask.reshape(-1, self.max_response_length)

    @property
    def flat_input_ids(self) -> Tensor:
        prompts = self.prompt_input_ids[:, None, :].expand(-1, self.group_size, -1)
        return torch.cat(
            [prompts.reshape(self.environment_samples, -1), self.flat_response_input_ids],
            dim=1,
        )

    @property
    def flat_attention_mask(self) -> Tensor:
        prompt_mask = self.prompt_attention_mask[:, None, :].expand(-1, self.group_size, -1)
        return torch.cat(
            [prompt_mask.reshape(self.environment_samples, -1), self.flat_response_mask],
            dim=1,
        )

    @property
    def old_completion_log_probs(self) -> Tensor:
        return completion_log_probs(self.old_token_log_probs, self.response_mask)

    @property
    def zero_advantage_fraction(self) -> float:
        zero_groups = self.advantages.abs().amax(dim=1).eq(0)
        return float(zero_groups.float().mean().item())

    def to(self, device: torch.device | str) -> SequenceRolloutBatch:
        """Return an equivalent rollout whose tensors live on ``device``."""

        return replace(
            self,
            prompt_input_ids=self.prompt_input_ids.to(device),
            prompt_attention_mask=self.prompt_attention_mask.to(device),
            response_input_ids=self.response_input_ids.to(device),
            response_mask=self.response_mask.to(device),
            old_token_log_probs=self.old_token_log_probs.to(device),
            rewards=self.rewards.to(device),
            advantages=self.advantages.to(device),
            frozen_prefix_cache=None,
            frozen_prefix_fallback_reason=(
                "frozen-prefix cache invalidated by rollout device transfer"
                if self.frozen_prefix_cache is not None
                else self.frozen_prefix_fallback_reason
            ),
        )


def group_leave_one_out_advantages(rewards: Tensor) -> Tensor:
    """Return ``reward - mean(other rewards)`` independently per prompt."""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("leave-one-out advantages require rewards with shape [B, G>=2]")
    numeric = rewards if rewards.is_floating_point() else rewards.float()
    if not torch.isfinite(numeric).all():
        raise ValueError("rewards must be finite")
    group_size = numeric.shape[1]
    return (group_size * numeric - numeric.sum(dim=1, keepdim=True)) / (group_size - 1)


def response_token_mask(response_input_ids: Tensor, eos_token_ids: Sequence[int]) -> Tensor:
    """Mask tokens through the first EOS, including that EOS token itself."""

    if response_input_ids.ndim < 1:
        raise ValueError("response_input_ids must have at least one dimension")
    eos_ids = tuple(dict.fromkeys(int(token_id) for token_id in eos_token_ids))
    if not eos_ids:
        return torch.ones_like(response_input_ids, dtype=torch.bool)
    is_eos = torch.zeros_like(response_input_ids, dtype=torch.bool)
    for token_id in eos_ids:
        is_eos |= response_input_ids.eq(token_id)
    eos_before = is_eos.to(torch.int64).cumsum(dim=-1) - is_eos.to(torch.int64)
    return eos_before.eq(0)


def completion_log_probs(
    token_log_probs: Tensor,
    response_mask: Tensor,
    *,
    length_normalize: bool = False,
) -> Tensor:
    """Reduce per-token values to fixed-candidate sequence log probabilities."""

    if token_log_probs.shape != response_mask.shape:
        raise ValueError("token_log_probs and response_mask must have the same shape")
    if response_mask.dtype != torch.bool:
        raise TypeError("response_mask must have boolean dtype")
    totals = token_log_probs.masked_fill(~response_mask, 0.0).sum(dim=-1)
    if not length_normalize:
        return totals
    lengths = response_mask.sum(dim=-1).clamp_min(1).to(totals.dtype)
    return totals / lengths


def clipped_grpo_surrogate(
    new_token_log_probs: Tensor,
    rollout: SequenceRolloutBatch,
    clip_epsilon: float,
) -> Tensor:
    """Return the response-length-normalized clipped GRPO surrogate.

    Ratios are token-local and each completion contributes equally regardless
    of its generated length.  Maximizing the returned scalar is the standard
    policy update; callers using an optimizer should minimize its negative.
    """

    if new_token_log_probs.shape != rollout.response_input_ids.shape:
        raise ValueError("new_token_log_probs must have shape [B, G, T]")
    if clip_epsilon <= 0 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be positive and finite")
    log_ratios = new_token_log_probs - rollout.old_token_log_probs
    ratios = log_ratios.exp()
    advantages = rollout.advantages.unsqueeze(-1)
    unclipped = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    token_surrogate = torch.minimum(unclipped, clipped).masked_fill(~rollout.response_mask, 0.0)
    lengths = rollout.response_lengths.clamp_min(1).to(token_surrogate.dtype)
    return (token_surrogate.sum(dim=-1) / lengths).mean()


def _teacher_forced_flat_token_log_probs(
    bundle: ModelBundle,
    input_ids: Tensor,
    attention_mask: Tensor,
    response_input_ids: Tensor,
    response_mask: Tensor,
    prompt_width: int,
    temperature: float,
    micro_batch_size: int | None,
) -> Tensor:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    total_examples, response_width = response_input_ids.shape
    if micro_batch_size is None:
        micro_batch_size = total_examples
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size < 1
    ):
        raise ValueError("micro_batch_size must be a positive integer")

    chunks: list[Tensor] = []
    from .frozen_prefix import qwen_teacher_forcing_logits_to_keep

    logits_to_keep = qwen_teacher_forcing_logits_to_keep(
        bundle,
        prompt_width,
        response_width,
    )
    for start in range(0, total_examples, micro_batch_size):
        end = min(start + micro_batch_size, total_examples)
        model_kwargs: dict[str, object] = {
            "input_ids": input_ids[start:end],
            "attention_mask": attention_mask[start:end],
            "use_cache": False,
        }
        if logits_to_keep is not None:
            model_kwargs["logits_to_keep"] = logits_to_keep
        outputs = bundle.model(
            **model_kwargs,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        expected_logit_width = (
            response_width if logits_to_keep is not None else prompt_width + response_width
        )
        if logits.ndim != 3 or logits.shape[1] < expected_logit_width:
            raise ValueError("causal LM returned logits with an incompatible shape")
        # Token at full-sequence position P + t is predicted by logits at P + t - 1.
        response_logits = (
            logits
            if logits_to_keep is not None
            else logits[:, prompt_width - 1 : prompt_width + response_width - 1, :]
        ).float()
        scaled_logits = response_logits / temperature
        targets = response_input_ids[start:end, :, None]
        selected_logits = scaled_logits.gather(-1, targets).squeeze(-1)
        token_log_probs = selected_logits - torch.logsumexp(scaled_logits, dim=-1)
        chunks.append(token_log_probs.masked_fill(~response_mask[start:end], 0.0))
    return torch.cat(chunks, dim=0)


def teacher_forced_token_log_probs(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    *,
    temperature: float | None = None,
    micro_batch_size: int | None = None,
) -> Tensor:
    """Rescore every stored response token under the current adapter parameters.

    This function intentionally has no ``no_grad`` decorator.  Standard GRPO
    can differentiate its result, while a forward-only optimizer can call it
    inside ``torch.inference_mode()`` after each adapter perturbation.
    """

    target_temperature = rollout.sampling_temperature if temperature is None else float(temperature)
    if rollout.frozen_prefix_cache is None:
        flat = _teacher_forced_flat_token_log_probs(
            bundle=bundle,
            input_ids=rollout.flat_input_ids,
            attention_mask=rollout.flat_attention_mask,
            response_input_ids=rollout.flat_response_input_ids,
            response_mask=rollout.flat_response_mask,
            prompt_width=rollout.prompt_input_ids.shape[1],
            temperature=target_temperature,
            micro_batch_size=micro_batch_size,
        )
    else:
        from .frozen_prefix import cached_prefix_flat_token_log_probs

        flat = cached_prefix_flat_token_log_probs(
            bundle,
            rollout.frozen_prefix_cache,
            rollout.flat_response_input_ids,
            rollout.flat_response_mask,
            prompt_width=rollout.prompt_input_ids.shape[1],
            temperature=target_temperature,
            micro_batch_size=micro_batch_size,
        )
    return flat.reshape_as(rollout.response_input_ids)


def attach_frozen_prefix_cache(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
) -> SequenceRolloutBatch:
    """Return ``rollout`` with an exact Qwen prefix cache when supported."""

    from .frozen_prefix import maybe_build_frozen_prefix_cache

    result = maybe_build_frozen_prefix_cache(
        bundle,
        rollout.flat_input_ids,
        rollout.flat_attention_mask,
    )
    return replace(
        rollout,
        frozen_prefix_cache=result.cache,
        frozen_prefix_fallback_reason=result.fallback_reason,
    )


def central_difference_score_statistics(
    positive_token_log_probs: Tensor,
    negative_token_log_probs: Tensor,
    rollout: SequenceRolloutBatch,
    radius: float,
    *,
    length_normalize: bool = False,
) -> ProjectedScoreStatistics:
    """Build projected policy-gradient/Fisher statistics from paired probes.

    Probe tensors must have shape ``[B, G, T, D]``.  All directions rescore the
    same fixed candidates in ``rollout``; no generation is performed here.
    """

    expected_prefix = rollout.response_input_ids.shape
    if positive_token_log_probs.shape != negative_token_log_probs.shape:
        raise ValueError("positive and negative probe tensors must have the same shape")
    if positive_token_log_probs.ndim != 4 or positive_token_log_probs.shape[:3] != expected_prefix:
        raise ValueError("probe tensors must have shape [B, G, T, D]")
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")

    mask = rollout.response_mask.unsqueeze(-1)
    token_scores = (
        (positive_token_log_probs - negative_token_log_probs) / (2.0 * radius)
    ).masked_fill(~mask, 0.0)
    sequence_scores = token_scores.sum(dim=2)
    lengths = rollout.response_lengths.clamp_min(1).to(sequence_scores.dtype)
    if length_normalize:
        sequence_scores = sequence_scores / lengths.unsqueeze(-1)
    gradient = (rollout.advantages.unsqueeze(-1) * sequence_scores).mean(dim=(0, 1))
    # The trust-region KL is an equal-weighted per-completion mean of
    # token-local KLs. Its local curvature is therefore the mean of token score
    # outer products—not an outer product of the summed completion score,
    # which would introduce spurious cross-token terms.
    per_completion_fisher = (
        torch.einsum("bgtd,bgte->bgde", token_scores, token_scores) / lengths[..., None, None]
    )
    fisher = per_completion_fisher.mean(dim=(0, 1))
    return ProjectedScoreStatistics(
        directional_token_scores=token_scores,
        completion_scores=sequence_scores,
        gradient=gradient,
        fisher=fisher,
    )


def _resolve_eos_token_ids(bundle: ModelBundle) -> tuple[int, ...]:
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    if eos is None:
        generation_config = getattr(bundle.model, "generation_config", None)
        eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        return ()
    if isinstance(eos, int):
        return (eos,)
    return tuple(dict.fromkeys(int(token_id) for token_id in eos))


def _resolve_pad_token_id(bundle: ModelBundle, eos_token_ids: tuple[int, ...]) -> int:
    pad_token_id = getattr(bundle.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        generation_config = getattr(bundle.model, "generation_config", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None and eos_token_ids:
        pad_token_id = eos_token_ids[0]
    if pad_token_id is None:
        raise ValueError("causal generation requires a tokenizer pad_token_id or eos_token_id")
    return int(pad_token_id)


def _left_padded_prompts(
    bundle: ModelBundle,
    prompts: Sequence[str],
    max_prompt_tokens: int | None,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    tokenizer_kwargs: dict[str, object] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_prompt_tokens is not None,
    }
    if max_prompt_tokens is not None:
        tokenizer_kwargs["max_length"] = max_prompt_tokens
    encoded = bundle.tokenizer(list(prompts), **tokenizer_kwargs)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        raise ValueError("tokenizer must return attention_mask for padded causal generation")
    if input_ids.ndim != 2 or input_ids.shape[0] != len(prompts):
        raise ValueError("tokenizer returned input_ids with an incompatible shape")

    # Normalize to left padding even when a caller supplies a right-padding tokenizer.
    rows = [row[mask.bool()] for row, mask in zip(input_ids, attention_mask, strict=True)]
    if any(row.numel() == 0 for row in rows):
        raise ValueError("every prompt must encode to at least one token")
    width = max(row.numel() for row in rows)
    left_ids = torch.full((len(rows), width), pad_token_id, dtype=input_ids.dtype)
    left_mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        left_ids[index, -row.numel() :] = row
        left_mask[index, -row.numel() :] = True
    return left_ids.to(bundle.device), left_mask.to(bundle.device)


@contextmanager
def _forked_rng(device: torch.device, seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer or None")
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        yield


def _decode_response(tokenizer: object, token_ids: Tensor) -> str:
    ids = token_ids.detach().cpu().tolist()
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=True)


def generate_sequence_rollouts(
    bundle: ModelBundle,
    prompts: Sequence[str],
    reward_callback: RewardCallback,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    max_prompt_tokens: int | None = 512,
    seed: int | None = None,
    scoring_micro_batch_size: int | None = None,
    use_frozen_prefix_scoring: bool = False,
) -> SequenceRolloutBatch:
    """Sample ``group_size`` on-policy completions per prompt and score them.

    Sampling is unrestricted temperature-scaled multinomial sampling (top-k is
    disabled and top-p is one), making teacher-forced old-policy rescoring
    exactly match the behavior policy before EOS.  A fresh generation config is
    used so model-specific decoding defaults cannot silently change that policy.
    """

    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a non-empty sequence of strings")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be a real number")
    temperature = float(temperature)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if max_prompt_tokens is not None and (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError("max_prompt_tokens must be a positive integer or None")
    if not isinstance(use_frozen_prefix_scoring, bool):
        raise TypeError("use_frozen_prefix_scoring must be boolean")

    eos_token_ids = _resolve_eos_token_ids(bundle)
    pad_token_id = _resolve_pad_token_id(bundle, eos_token_ids)
    prompt_input_ids, prompt_attention_mask = _left_padded_prompts(
        bundle, prompts, max_prompt_tokens, pad_token_id
    )

    # Import lazily so toy tests and callers that only rescore stored rollouts do
    # not initialize the Transformers generation stack.
    from transformers import GenerationConfig

    eos_for_generation: int | list[int] | None
    if not eos_token_ids:
        eos_for_generation = None
    elif len(eos_token_ids) == 1:
        eos_for_generation = eos_token_ids[0]
    else:
        eos_for_generation = list(eos_token_ids)
    generation_config = GenerationConfig(
        do_sample=True,
        num_return_sequences=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_for_generation,
        use_cache=True,
        return_dict_in_generate=True,
    )

    bundle.model.eval()
    with _forked_rng(bundle.device, seed), torch.no_grad():
        generated = bundle.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            generation_config=generation_config,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    expected_examples = len(prompts) * group_size
    prompt_width = prompt_input_ids.shape[1]
    if sequences.ndim != 2 or sequences.shape[0] != expected_examples:
        raise ValueError("generate returned an unexpected number or shape of sequences")
    if not torch.equal(
        sequences[:, :prompt_width], prompt_input_ids.repeat_interleave(group_size, dim=0)
    ):
        raise ValueError("only decoder-only models that preserve the input prefix are supported")
    response_flat = sequences[:, prompt_width:]
    if response_flat.shape[1] == 0 or response_flat.shape[1] > max_new_tokens:
        raise ValueError("generate returned an invalid number of response tokens")
    response_width = response_flat.shape[1]
    response_input_ids = response_flat.reshape(len(prompts), group_size, response_width)
    response_mask = response_token_mask(response_input_ids, eos_token_ids)
    response_input_ids = response_input_ids.masked_fill(~response_mask, pad_token_id)

    # Freeze behavior-policy statistics before invoking arbitrary user reward code.
    repeated_prompts = prompt_input_ids[:, None, :].expand(-1, group_size, -1)
    repeated_prompt_mask = prompt_attention_mask[:, None, :].expand(-1, group_size, -1)
    flat_input_ids = torch.cat(
        [
            repeated_prompts.reshape(expected_examples, -1),
            response_input_ids.reshape(expected_examples, -1),
        ],
        dim=1,
    )
    flat_attention_mask = torch.cat(
        [
            repeated_prompt_mask.reshape(expected_examples, -1),
            response_mask.reshape(expected_examples, -1),
        ],
        dim=1,
    )
    frozen_prefix_cache = None
    frozen_prefix_fallback_reason = None
    with torch.no_grad():
        if use_frozen_prefix_scoring:
            from .frozen_prefix import (
                cached_prefix_flat_token_log_probs,
                maybe_build_frozen_prefix_cache,
            )

            cache_result = maybe_build_frozen_prefix_cache(
                bundle,
                flat_input_ids,
                flat_attention_mask,
            )
            frozen_prefix_cache = cache_result.cache
            frozen_prefix_fallback_reason = cache_result.fallback_reason
        if frozen_prefix_cache is None:
            old_flat = _teacher_forced_flat_token_log_probs(
                bundle=bundle,
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                response_input_ids=response_input_ids.reshape(expected_examples, -1),
                response_mask=response_mask.reshape(expected_examples, -1),
                prompt_width=prompt_width,
                temperature=temperature,
                micro_batch_size=scoring_micro_batch_size,
            )
        else:
            old_flat = cached_prefix_flat_token_log_probs(
                bundle,
                frozen_prefix_cache,
                response_input_ids.reshape(expected_examples, -1),
                response_mask.reshape(expected_examples, -1),
                prompt_width=prompt_width,
                temperature=temperature,
                micro_batch_size=scoring_micro_batch_size,
            )
    old_token_log_probs = old_flat.reshape_as(response_input_ids).detach()

    completion_groups: list[tuple[str, ...]] = []
    reward_groups: list[list[float]] = []
    for prompt_index, prompt in enumerate(prompts):
        completion_group: list[str] = []
        reward_group: list[float] = []
        for group_index in range(group_size):
            valid_ids = response_input_ids[prompt_index, group_index][
                response_mask[prompt_index, group_index]
            ]
            completion = _decode_response(bundle.tokenizer, valid_ids)
            sample = CompletionSample(
                prompt_index=prompt_index,
                group_index=group_index,
                prompt=prompt,
                completion=completion,
                response_token_ids=tuple(int(token_id) for token_id in valid_ids.tolist()),
            )
            reward = reward_callback(sample)
            if not isinstance(reward, Real):
                raise TypeError("reward callback must return a real number")
            numeric_reward = float(reward)
            if not math.isfinite(numeric_reward):
                raise ValueError("reward callback must return a finite value")
            completion_group.append(completion)
            reward_group.append(numeric_reward)
        completion_groups.append(tuple(completion_group))
        reward_groups.append(reward_group)

    rewards = torch.tensor(reward_groups, dtype=torch.float32, device=bundle.device)
    advantages = group_leave_one_out_advantages(rewards)
    return SequenceRolloutBatch(
        prompts=tuple(prompts),
        completions=tuple(completion_groups),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        response_input_ids=response_input_ids,
        response_mask=response_mask,
        old_token_log_probs=old_token_log_probs,
        rewards=rewards,
        advantages=advantages,
        sampling_temperature=temperature,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
        frozen_prefix_cache=frozen_prefix_cache,
        frozen_prefix_fallback_reason=frozen_prefix_fallback_reason,
    )


__all__ = [
    "CompletionSample",
    "ProjectedScoreStatistics",
    "RewardCallback",
    "SequenceRolloutBatch",
    "attach_frozen_prefix_cache",
    "central_difference_score_statistics",
    "clipped_grpo_surrogate",
    "completion_log_probs",
    "generate_sequence_rollouts",
    "group_leave_one_out_advantages",
    "response_token_mask",
    "teacher_forced_token_log_probs",
]
