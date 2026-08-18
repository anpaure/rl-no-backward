"""Small-model GSM8K RLVR benchmark for backprop and forward-only updates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    DifficultyFilter,
    GSM8KExample,
    exact_match_reward,
    extract_model_answer,
    filter_by_difficulty,
    format_prompt,
    load_gsm8k_split,
    select_seeded_subset,
)
from .model import (
    ATTENTION_IMPLEMENTATIONS,
    TORCH_COMPILE_MODES,
    ModelBundle,
    installed_flash_attn_version,
    load_model_bundle,
    model_forward_is_compiled,
    parameter_vector,
    resolved_attention_implementation,
    set_adapter_grad_enabled,
    set_parameter_vector,
)
from .rollout_provenance import build_rollout_provenance
from .sequence_optimizers import (
    BackpropSequenceConfig,
    ForwardSequenceConfig,
    SequenceActiveSubspace,
    enable_batched_probe_adapters,
    forward_sequence_step,
    make_sequence_grpo_optimizer,
    sequence_grpo_step,
)
from .sequence_policy import (
    CompletionSample,
    SequenceRolloutBatch,
    _decode_response,
    _left_padded_prompts,
    _resolve_eos_token_ids,
    _resolve_pad_token_id,
    attach_frozen_prefix_cache,
    generate_sequence_rollouts,
    group_leave_one_out_advantages,
    teacher_forced_token_log_probs,
)
from .task import CANDIDATE_ACTIONS
from .vllm_plugin import VLLM_V1_MULTIPROCESSING_ENV
from .vllm_rollout import (
    OnPolicyVLLMGenerator,
    capture_residual_adapter_snapshot,
    create_vllm_engine,
)

GSM8K_METHODS = ("base", "bp_grpo", "fo_pg", "fo_npg", "focus_npg")


@dataclass
class GSM8KExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str | None = None
    dataset_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    attention_implementation: str = "eager"
    compile_model_forward: bool = False
    compile_model_forward_mode: str = "default"
    adapter_rank: int = 8
    adapter_layers: int = 4
    adapter_scale: float = 1.0
    train_size: int = 128
    val_size: int = 64
    test_size: int = 128
    run_test_evaluation: bool = False
    test_exclusion_metadata: str | None = None
    subset_seed: int = 0
    min_reasoning_lines: int = 2
    max_reasoning_lines: int = 3
    max_answer_magnitude: float = 100_000.0
    methods: list[str] = field(
        default_factory=lambda: ["base", "bp_grpo", "fo_pg", "fo_npg", "focus_npg"]
    )
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    steps: int = 40
    batch_size: int = 4
    group_size: int = 4
    max_prompt_tokens: int = 256
    max_new_tokens: int = 256
    sampling_temperature: float = 0.8
    scoring_micro_batch_size: int = 4
    use_frozen_prefix_scoring: bool = False
    record_rollout_provenance: bool = False
    eval_interval: int = 10
    eval_batch_size: int = 4
    numeric_shaping_weight: float = 0.1
    calibration_examples: int = 32
    wandb_project: str = "rl-no-backward"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
    rollout_backend: str = "hf"
    vllm_kv_cache_memory_bytes: int = 2 * 1024**3
    vllm_enforce_eager: bool = True
    vllm_batch_invariant: bool = False
    vllm_enable_v1_multiprocessing: bool = True
    vllm_flash_attn_version: int = 2
    vllm_allow_insecure_serialization: bool = False
    vllm_logprob_mean_abs_tolerance: float = 0.02
    vllm_logprob_p99_abs_tolerance: float = 0.2
    vllm_logprob_max_abs_tolerance: float = 0.5
    backprop: dict[str, Any] = field(default_factory=dict)
    forward: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GSM8KExperimentConfig:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown GSM8K config keys: {unknown}")
        config = cls(**dict(mapping))
        config.validate()
        return config

    def validate(self) -> None:
        unknown = sorted(set(self.methods) - set(GSM8K_METHODS))
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        for name in (
            "train_size",
            "val_size",
            "steps",
            "batch_size",
            "group_size",
            "max_prompt_tokens",
            "max_new_tokens",
            "eval_interval",
            "eval_batch_size",
            "scoring_micro_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.test_size < 0:
            raise ValueError("test_size must be non-negative")
        if self.run_test_evaluation and self.test_size < 1:
            raise ValueError("test_size must be positive when test evaluation is enabled")
        for name in ("model_revision", "dataset_revision", "test_exclusion_metadata"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or null")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not 0.0 <= self.numeric_shaping_weight < 1.0:
            raise ValueError("numeric_shaping_weight must lie in [0, 1)")
        if self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if not isinstance(self.use_frozen_prefix_scoring, bool):
            raise TypeError("use_frozen_prefix_scoring must be boolean")
        if not isinstance(self.record_rollout_provenance, bool):
            raise TypeError("record_rollout_provenance must be boolean")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.rollout_backend not in {"hf", "vllm"}:
            raise ValueError("rollout_backend must be hf or vllm")
        if self.rollout_backend == "vllm" and not str(self.device).startswith("cuda"):
            raise ValueError("the vLLM rollout backend requires a CUDA device")
        if (
            isinstance(self.vllm_kv_cache_memory_bytes, bool)
            or not isinstance(self.vllm_kv_cache_memory_bytes, int)
            or self.vllm_kv_cache_memory_bytes < 1
        ):
            raise ValueError("vllm_kv_cache_memory_bytes must be a positive integer")
        if not isinstance(self.vllm_enforce_eager, bool):
            raise TypeError("vllm_enforce_eager must be boolean")
        if not isinstance(self.vllm_batch_invariant, bool):
            raise TypeError("vllm_batch_invariant must be boolean")
        if self.vllm_batch_invariant and self.rollout_backend != "vllm":
            raise ValueError("vllm_batch_invariant=true requires rollout_backend='vllm'")
        if not isinstance(self.vllm_enable_v1_multiprocessing, bool):
            raise TypeError("vllm_enable_v1_multiprocessing must be boolean")
        if (
            isinstance(self.vllm_flash_attn_version, bool)
            or not isinstance(self.vllm_flash_attn_version, int)
            or self.vllm_flash_attn_version not in {2, 3}
        ):
            raise ValueError("vllm_flash_attn_version must be 2 or 3")
        if not isinstance(self.vllm_allow_insecure_serialization, bool):
            raise TypeError("vllm_allow_insecure_serialization must be boolean")
        if (
            self.rollout_backend == "vllm"
            and self.vllm_enable_v1_multiprocessing
            and not self.vllm_allow_insecure_serialization
        ):
            raise ValueError(
                "the multiprocess mutable vLLM rollout backend requires explicit "
                "vllm_allow_insecure_serialization=true for trusted local callable IPC"
            )
        for name in (
            "vllm_logprob_mean_abs_tolerance",
            "vllm_logprob_p99_abs_tolerance",
            "vllm_logprob_max_abs_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if not (
            self.vllm_logprob_mean_abs_tolerance
            <= self.vllm_logprob_p99_abs_tolerance
            <= self.vllm_logprob_max_abs_tolerance
        ):
            raise ValueError("vLLM log-prob tolerances must satisfy mean <= p99 <= maximum")
        if not isinstance(self.attention_implementation, str) or (
            self.attention_implementation not in ATTENTION_IMPLEMENTATIONS
        ):
            raise ValueError(
                f"attention_implementation must be one of {', '.join(ATTENTION_IMPLEMENTATIONS)}"
            )
        if not isinstance(self.compile_model_forward, bool):
            raise TypeError("compile_model_forward must be boolean")
        if not isinstance(self.compile_model_forward_mode, str) or (
            self.compile_model_forward_mode not in TORCH_COMPILE_MODES
        ):
            raise ValueError(
                f"compile_model_forward_mode must be one of {', '.join(TORCH_COMPILE_MODES)}"
            )
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not isinstance(self.backprop, Mapping):
            raise TypeError("backprop must be a mapping")
        if not isinstance(self.forward, Mapping):
            raise TypeError("forward must be a mapping")
        BackpropSequenceConfig(**dict(self.backprop))
        # Validate the shared forward settings independently of which
        # forward-only methods are selected for this run.
        ForwardSequenceConfig(method="fo_pg", **dict(self.forward))


def _model_runtime_metadata(
    bundle: ModelBundle,
    config: GSM8KExperimentConfig,
) -> dict[str, Any]:
    """Return requested and effective Hugging Face execution controls."""

    return {
        "requested_attention_implementation": config.attention_implementation,
        "resolved_attention_implementation": resolved_attention_implementation(bundle.model),
        "flash_attn_version": installed_flash_attn_version(),
        "compile_model_forward": config.compile_model_forward,
        "compile_model_forward_mode": config.compile_model_forward_mode,
        "model_forward_compiled": model_forward_is_compiled(bundle.model),
    }


def _vllm_process_metadata(
    config: GSM8KExperimentConfig,
    *,
    active: bool,
) -> dict[str, Any]:
    """Describe process-local memory scope and RNG behavior for vLLM."""

    if not active:
        return {
            "gpu_memory_metric_scope": "hugging_face_trainer_process_torch_allocator",
            "gpu_memory_metrics_exclude_vllm_worker": False,
            "vllm_enable_v1_multiprocessing": config.vllm_enable_v1_multiprocessing,
            "vllm_enable_v1_multiprocessing_env": os.environ.get(
                VLLM_V1_MULTIPROCESSING_ENV
            ),
            "vllm_engine_process_mode": None,
            "vllm_inprocess_global_rng_caveat": None,
        }
    if config.vllm_enable_v1_multiprocessing:
        return {
            "gpu_memory_metric_scope": "hugging_face_trainer_process_torch_allocator",
            "gpu_memory_metrics_exclude_vllm_worker": True,
            "vllm_enable_v1_multiprocessing": True,
            "vllm_enable_v1_multiprocessing_env": os.environ.get(
                VLLM_V1_MULTIPROCESSING_ENV
            ),
            "vllm_engine_process_mode": "multiprocess",
            "vllm_inprocess_global_rng_caveat": None,
        }
    return {
        "gpu_memory_metric_scope": (
            "single_process_hugging_face_and_vllm_torch_allocator"
        ),
        "gpu_memory_metrics_exclude_vllm_worker": False,
        "vllm_enable_v1_multiprocessing": False,
        "vllm_enable_v1_multiprocessing_env": os.environ.get(
            VLLM_V1_MULTIPROCESSING_ENV
        ),
        "vllm_engine_process_mode": "in_process",
        "vllm_inprocess_global_rng_caveat": (
            "vLLM 0.22 sets the process-global random seed when V1 multiprocessing "
            "is disabled; benchmark sampling, example selection, and probe directions "
            "use explicit seeds, but unrelated global RNG consumers may be affected"
        ),
    }


def _chat_formatter(
    tokenizer: object,
) -> Callable[[Sequence[Mapping[str, str]]], str]:
    def render(messages: Sequence[Mapping[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    return render


def _prompts(tokenizer: object, examples: Sequence[GSM8KExample]) -> list[str]:
    formatter = _chat_formatter(tokenizer)
    return [format_prompt(example.question, formatter) for example in examples]


def _fraction(value: str) -> Fraction:
    return Fraction(value)


def shaped_gsm8k_reward(
    completion: str,
    reference: GSM8KExample,
    shaping_weight: float,
) -> float:
    """Exact reward plus a small, bounded log-distance signal for wrong numerics."""

    if exact_match_reward(completion, reference):
        return 1.0
    predicted = extract_model_answer(completion)
    if predicted is None or shaping_weight <= 0:
        return 0.0
    predicted_value = abs(float(_fraction(predicted)))
    reference_value = abs(float(_fraction(reference.canonical_answer)))
    log_distance = abs(math.log1p(predicted_value) - math.log1p(reference_value))
    return float(shaping_weight * math.exp(-log_distance))


def _reward_callback(
    examples: Sequence[GSM8KExample], shaping_weight: float
) -> Callable[[CompletionSample], float]:
    def reward(sample: CompletionSample) -> float:
        return shaped_gsm8k_reward(
            sample.completion,
            examples[sample.prompt_index],
            shaping_weight,
        )

    return reward


def _exact_rollout_metrics(
    completions: Sequence[Sequence[str]], examples: Sequence[GSM8KExample]
) -> tuple[float, float]:
    groups = [
        [exact_match_reward(completion, examples[prompt_index]) for completion in group]
        for prompt_index, group in enumerate(completions)
    ]
    values = [value for group in groups for value in group]
    zero_advantage_groups = sum(len(set(group)) == 1 for group in groups)
    return (
        float(sum(values) / len(values)),
        float(zero_advantage_groups / len(groups)),
    )


def _unpadded_prompt_token_ids(
    prompt_input_ids: Tensor, prompt_attention_mask: Tensor
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token_id) for token_id in row[mask].detach().cpu().tolist())
        for row, mask in zip(prompt_input_ids, prompt_attention_mask.bool(), strict=True)
    )


def _vllm_policy_fields(
    rollout_policy: OnPolicyVLLMGenerator | None,
    *,
    field_prefix: str = "rollout_policy",
) -> dict[str, str | None]:
    if rollout_policy is None:
        return {
            f"{field_prefix}_version": None,
            f"{field_prefix}_state_digest": None,
        }
    return {
        f"{field_prefix}_version": rollout_policy.policy_version,
        f"{field_prefix}_state_digest": rollout_policy.state_digest,
    }


def _sync_vllm_policy(
    bundle: ModelBundle,
    rollout_policy: OnPolicyVLLMGenerator | None,
    *,
    version: str,
) -> None:
    if rollout_policy is None:
        return
    snapshot = capture_residual_adapter_snapshot(bundle)
    rollout_policy.sync(snapshot, version=version, include_bases=False)


@torch.no_grad()
def _generate_vllm_sequence_rollouts(
    bundle: ModelBundle,
    prompts: Sequence[str],
    examples: Sequence[GSM8KExample],
    config: GSM8KExperimentConfig,
    rollout_policy: OnPolicyVLLMGenerator,
    *,
    seed: int,
) -> tuple[SequenceRolloutBatch, dict[str, float]]:
    """Build a regular rollout from vLLM behavior tokens and probabilities."""

    eos_token_ids = _resolve_eos_token_ids(bundle)
    pad_token_id = _resolve_pad_token_id(bundle, eos_token_ids)
    prompt_input_ids, prompt_attention_mask = _left_padded_prompts(
        bundle,
        prompts,
        config.max_prompt_tokens,
        pad_token_id,
    )
    prompt_token_ids = _unpadded_prompt_token_ids(prompt_input_ids, prompt_attention_mask)
    behavior_version = rollout_policy.policy_version
    behavior_digest = rollout_policy.state_digest
    if behavior_version is None or behavior_digest is None:
        raise RuntimeError("vLLM generation requires a fully synchronized behavior policy")
    generated = rollout_policy.generate(
        prompt_token_ids,
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.sampling_temperature,
        seed=seed,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
        device=bundle.device,
    )
    if generated.policy_version != behavior_version:
        raise RuntimeError(
            "vLLM generation returned the wrong behavior-policy version: "
            f"expected {behavior_version!r}, got {generated.policy_version!r}"
        )
    if (
        rollout_policy.policy_version != behavior_version
        or rollout_policy.state_digest != behavior_digest
    ):
        raise RuntimeError("vLLM behavior policy changed during blocking generation")

    completion_groups: list[tuple[str, ...]] = []
    reward_groups: list[list[float]] = []
    reward_callback = _reward_callback(examples, config.numeric_shaping_weight)
    for prompt_index, prompt in enumerate(prompts):
        completion_group: list[str] = []
        reward_group: list[float] = []
        for group_index in range(config.group_size):
            valid_ids = generated.response_input_ids[prompt_index, group_index][
                generated.response_mask[prompt_index, group_index]
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
            completion_group.append(completion)
            reward_group.append(float(reward))
        completion_groups.append(tuple(completion_group))
        reward_groups.append(reward_group)

    rewards = torch.tensor(reward_groups, dtype=torch.float32, device=bundle.device)
    rollout = SequenceRolloutBatch(
        prompts=tuple(prompts),
        completions=tuple(completion_groups),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        response_input_ids=generated.response_input_ids,
        response_mask=generated.response_mask,
        old_token_log_probs=generated.old_token_log_probs.detach(),
        rewards=rewards,
        advantages=group_leave_one_out_advantages(rewards),
        sampling_temperature=config.sampling_temperature,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
    )
    if config.use_frozen_prefix_scoring:
        rollout = attach_frozen_prefix_cache(bundle, rollout)

    # q_vLLM is the actual behavior distribution, so it remains the GRPO
    # denominator.  A same-weight HF score is a mandatory equivalence gate,
    # not a replacement that would conceal an off-policy rollout.
    hf_center_log_probs = teacher_forced_token_log_probs(
        bundle,
        rollout,
        micro_batch_size=config.scoring_micro_batch_size,
    ).detach()
    signed_delta = hf_center_log_probs - rollout.old_token_log_probs
    valid_delta = signed_delta[rollout.response_mask].float()
    absolute_delta = valid_delta.abs()
    mean_absolute = float(absolute_delta.mean().item())
    maximum_absolute = float(absolute_delta.max().item())
    p99_absolute = float(torch.quantile(absolute_delta, 0.99).item())
    mean_signed = float(valid_delta.mean().item())
    maximum_ratio_deviation = float((valid_delta.exp() - 1.0).abs().max().item())
    diagnostics = {
        "behavior_hf_logprob_mean_abs_delta": mean_absolute,
        "behavior_hf_logprob_p99_abs_delta": p99_absolute,
        "behavior_hf_logprob_max_abs_delta": maximum_absolute,
        "behavior_hf_logprob_mean_signed_delta": mean_signed,
        "behavior_hf_max_importance_ratio_deviation": maximum_ratio_deviation,
    }
    if (
        mean_absolute > config.vllm_logprob_mean_abs_tolerance
        or p99_absolute > config.vllm_logprob_p99_abs_tolerance
        or maximum_absolute > config.vllm_logprob_max_abs_tolerance
    ):
        raise RuntimeError(
            "vLLM/HF behavior-policy equivalence gate failed: "
            f"mean_abs={mean_absolute:.6g} "
            f"(limit {config.vllm_logprob_mean_abs_tolerance:.6g}), "
            f"p99_abs={p99_absolute:.6g} "
            f"(limit {config.vllm_logprob_p99_abs_tolerance:.6g}), "
            f"max_abs={maximum_absolute:.6g} "
            f"(limit {config.vllm_logprob_max_abs_tolerance:.6g}), "
            f"max_ratio_deviation={maximum_ratio_deviation:.6g}; "
            f"HF attention={config.attention_implementation!r}, "
            f"vLLM flash_attn_version={config.vllm_flash_attn_version}"
        )
    return rollout, diagnostics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _start_wandb(
    config: GSM8KExperimentConfig,
    method: str,
    seed: int,
    output_dir: Path,
) -> Any | None:
    if config.wandb_mode == "disabled":
        return None
    import wandb

    optimizer_family = (
        "baseline" if method == "base" else "backprop" if method == "bp_grpo" else "forward-only"
    )
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=f"gsm8k-{output_dir.name}",
        name=f"gsm8k-{method}-seed-{seed}",
        job_type="evaluation" if method == "base" else "train",
        tags=["gsm8k", method, optimizer_family],
        config={**asdict(config), "method": method, "seed": seed},
        mode=config.wandb_mode,
        dir=str(output_dir),
        reinit="finish_previous",
    )
    run.define_metric("optimizer_step")
    run.define_metric("eval/*", step_metric="optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("progress/*", step_metric="optimizer_step")
    return run


def _log_wandb(run: Any | None, record: dict[str, Any]) -> None:
    if run is None:
        return
    namespace = "eval" if record["kind"] == "evaluation" else "train"
    progress_keys = {
        "wall_time_seconds",
        "environment_samples",
        "generated_tokens",
        "scored_tokens",
        "forward_calls",
        "full_prefix_calls",
        "suffix_calls",
        "backward_calls",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
    }
    payload: dict[str, Any] = {"optimizer_step": record["step"]}
    for key, value in _json_safe(record).items():
        if isinstance(value, (int, float)) and key not in {"step", "seed"}:
            prefix = "progress" if key in progress_keys else namespace
            payload[f"{prefix}/{key}"] = value
    run.log(payload)


def _eos_token_ids(tokenizer: object) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(token_id) for token_id in value)


def _rollout_truncation_fraction(rollout: Any, tokenizer: object) -> float:
    eos_ids = _eos_token_ids(tokenizer)
    if not eos_ids:
        return float("nan")
    lengths = rollout.response_lengths.clamp_min(1)
    final_ids = rollout.response_input_ids.gather(-1, (lengths - 1).unsqueeze(-1)).squeeze(-1)
    ended = torch.zeros_like(final_ids, dtype=torch.bool)
    for token_id in eos_ids:
        ended |= final_ids.eq(token_id)
    return float((~ended).float().mean().item())


def _peak_gpu_memory_metrics(device: torch.device) -> dict[str, int]:
    """Return cumulative CUDA allocator peaks with a compatibility alias."""

    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        allocated = 0
        reserved = 0
    return {
        # Historical artifacts used this key for allocated—not reserved—bytes.
        "peak_gpu_memory_bytes": allocated,
        "peak_gpu_memory_allocated_bytes": allocated,
        "peak_gpu_memory_reserved_bytes": reserved,
    }


@torch.inference_mode()
def evaluate_gsm8k(
    bundle: ModelBundle,
    examples: Sequence[GSM8KExample],
    config: GSM8KExperimentConfig,
    metric_prefix: str = "val",
    rollout_policy: OnPolicyVLLMGenerator | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Greedy exact-match evaluation on a fixed official-test subset."""

    prompts = _prompts(bundle.tokenizer, examples)
    exact_values: list[float] = []
    shaped_values: list[float] = []
    valid_values: list[float] = []
    response_lengths: list[int] = []
    truncated_values: list[float] = []
    rows: list[dict[str, Any]] = []
    eos_ids = _eos_token_ids(bundle.tokenizer)
    eos_id = bundle.tokenizer.eos_token_id
    pad_id = bundle.tokenizer.pad_token_id
    evaluation_policy_version: str | None = None
    evaluation_policy_digest: str | None = None
    if rollout_policy is not None:
        evaluation_policy_version = rollout_policy.policy_version
        evaluation_policy_digest = rollout_policy.state_digest
        if evaluation_policy_version is None or evaluation_policy_digest is None:
            raise RuntimeError("vLLM evaluation requires a fully synchronized policy")
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = examples[start : start + config.eval_batch_size]
        batch_prompts = prompts[start : start + config.eval_batch_size]
        if rollout_policy is None:
            encoded = bundle.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_prompt_tokens,
            )
            encoded = {name: value.to(bundle.device) for name, value in encoded.items()}
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            response_ids = generated[:, encoded["input_ids"].shape[1] :]
            token_sequences = [row for row in response_ids]
        else:
            prompt_ids, prompt_mask = _left_padded_prompts(
                bundle,
                batch_prompts,
                config.max_prompt_tokens,
                int(pad_id),
            )
            greedy = rollout_policy.generate_greedy(
                _unpadded_prompt_token_ids(prompt_ids, prompt_mask),
                max_new_tokens=config.max_new_tokens,
                eos_token_ids=eos_ids,
            )
            if greedy.policy_version != evaluation_policy_version:
                raise RuntimeError(
                    "vLLM greedy evaluation returned the wrong policy version: "
                    f"expected {evaluation_policy_version!r}, got {greedy.policy_version!r}"
                )
            if (
                rollout_policy.policy_version != evaluation_policy_version
                or rollout_policy.state_digest != evaluation_policy_digest
            ):
                raise RuntimeError("vLLM policy changed during greedy evaluation")
            token_sequences = [
                torch.tensor(token_ids, dtype=torch.long, device=bundle.device)
                for token_ids in greedy.response_token_ids
            ]
        completions = [
            _decode_response(bundle.tokenizer, token_ids) for token_ids in token_sequences
        ]
        for example, completion, token_ids in zip(
            batch_examples, completions, token_sequences, strict=True
        ):
            predicted = extract_model_answer(completion)
            exact = exact_match_reward(completion, example)
            shaped = shaped_gsm8k_reward(completion, example, config.numeric_shaping_weight)
            if eos_ids:
                eos_mask = torch.zeros_like(token_ids, dtype=torch.bool)
                for token_id in eos_ids:
                    eos_mask |= token_ids.eq(token_id)
                eos_positions = eos_mask.nonzero(as_tuple=False)
                length = (
                    int(eos_positions[0].item()) + 1 if eos_positions.numel() else token_ids.numel()
                )
                truncated = not bool(eos_positions.numel())
            else:
                length = token_ids.numel()
                truncated = False
            exact_values.append(exact)
            shaped_values.append(shaped)
            valid_values.append(float(predicted is not None))
            response_lengths.append(length)
            truncated_values.append(float(truncated))
            rows.append(
                {
                    "example_id": example.example_id,
                    "question": example.question,
                    "reference_answer": example.canonical_answer,
                    "predicted_answer": predicted,
                    "completion": completion,
                    "exact_reward": exact,
                    "shaped_reward": shaped,
                    "response_tokens": length,
                    "truncated": truncated,
                }
            )
    metrics = {
        f"{metric_prefix}_accuracy": float(sum(exact_values) / len(exact_values)),
        f"{metric_prefix}_exact_reward": float(sum(exact_values) / len(exact_values)),
        f"{metric_prefix}_shaped_reward": float(sum(shaped_values) / len(shaped_values)),
        f"{metric_prefix}_valid_answer_rate": float(sum(valid_values) / len(valid_values)),
        f"{metric_prefix}_mean_response_tokens": float(
            sum(response_lengths) / len(response_lengths)
        ),
        f"{metric_prefix}_truncation_fraction": float(
            sum(truncated_values) / len(truncated_values)
        ),
    }
    return metrics, rows


def _sample_training_batch(
    examples: Sequence[GSM8KExample], batch_size: int, rng: random.Random
) -> list[GSM8KExample]:
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def _example_id_fingerprint(examples: Sequence[GSM8KExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.example_id.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(output.strip())


def _excluded_test_ids(metadata_path: str | None) -> set[str]:
    if metadata_path is None:
        return set()
    path = Path(metadata_path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read test exclusion metadata {path}: {error}") from error
    values = metadata.get("test_example_ids") if isinstance(metadata, Mapping) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path} must contain a string list named test_example_ids")
    return set(values)


def _finish_run_artifact(
    wandb_run: Any | None,
    raw_path: Path,
    checkpoint_path: Path,
    samples_path: Path,
) -> None:
    if wandb_run is None:
        return
    import wandb

    artifact = wandb.Artifact(f"gsm8k-run-{wandb_run.id}", type="training-run")
    artifact.add_file(str(raw_path), name=raw_path.name)
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    artifact.add_file(str(samples_path), name=samples_path.name)
    wandb_run.log_artifact(artifact)
    wandb_run.finish()


def run_gsm8k_trial(
    bundle: ModelBundle,
    train_examples: Sequence[GSM8KExample],
    val_examples: Sequence[GSM8KExample],
    test_examples: Sequence[GSM8KExample],
    config: GSM8KExperimentConfig,
    method: str,
    seed: int,
    initial_parameters: Tensor,
    initial_val_metrics: Mapping[str, float],
    initial_val_samples: Sequence[Mapping[str, Any]],
    output_dir: Path,
    initial_evaluation_seconds: float = 0.0,
    rollout_policy: OnPolicyVLLMGenerator | None = None,
) -> Path:
    """Run one matched method/seed trial and persist all raw evidence."""

    raw_dir = output_dir / "raw"
    checkpoint_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"gsm8k_{method}_seed{seed}.jsonl"
    checkpoint_path = checkpoint_dir / f"gsm8k_{method}_seed{seed}.pt"
    samples_path = samples_dir / f"gsm8k_{method}_seed{seed}.json"
    if raw_path.exists():
        raw_path.unlink()

    set_parameter_vector(bundle, initial_parameters)
    bundle.model.eval()
    set_adapter_grad_enabled(bundle, method == "bp_grpo")
    if config.rollout_backend == "vllm" and rollout_policy is None:
        raise ValueError("rollout_backend='vllm' requires an initialized rollout policy")
    if config.rollout_backend == "hf" and rollout_policy is not None:
        raise ValueError("the HF rollout backend must not receive a vLLM policy")
    _sync_vllm_policy(
        bundle,
        rollout_policy,
        version=f"method={method}/seed={seed}/reset",
    )
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(bundle.device)

    wandb_run = _start_wandb(config, method, seed, output_dir)
    optimizer = None
    bp_config = BackpropSequenceConfig(**config.backprop)
    if method == "bp_grpo":
        optimizer = make_sequence_grpo_optimizer(bundle, bp_config)
    focus_state = None
    if method == "focus_npg":
        template = ForwardSequenceConfig(method="focus_npg", **config.forward)
        focus_state = SequenceActiveSubspace(template.active_rank, template.history_size)

    prompt_rng = random.Random(10_000 + seed)
    direction_generator = torch.Generator(device=bundle.device).manual_seed(30_000 + seed)
    cumulative_environment_samples = 0
    cumulative_generated_tokens = 0
    cumulative_scored_tokens = 0
    cumulative_forward_calls = 0
    cumulative_full_prefix_calls = 0
    cumulative_suffix_calls = 0
    cumulative_backward_calls = 0
    cumulative_teacher_forced_examples = 0
    final_samples = [dict(sample) for sample in initial_val_samples]
    initial_record = {
        "kind": "evaluation",
        "method": method,
        "seed": seed,
        "step": 0,
        "wall_time_seconds": 0.0,
        "evaluation_seconds": initial_evaluation_seconds,
        "environment_samples": 0,
        "generated_tokens": 0,
        "scored_tokens": 0,
        "forward_calls": 0,
        "full_prefix_calls": 0,
        "suffix_calls": 0,
        "backward_calls": 0,
        "teacher_forced_examples": 0,
        "rollout_backend": config.rollout_backend,
        **_vllm_policy_fields(rollout_policy),
        **initial_val_metrics,
    }
    _append_jsonl(raw_path, initial_record)
    _log_wandb(wandb_run, initial_record)
    best_val_accuracy = float(initial_val_metrics["val_accuracy"])
    best_step = 0
    best_parameters = initial_parameters.clone()
    _sync(bundle.device)
    start_time = time.perf_counter()

    if method == "base":
        if config.run_test_evaluation:
            _sync(bundle.device)
            evaluation_start = time.perf_counter()
            test_metrics, final_samples = evaluate_gsm8k(
                bundle,
                test_examples,
                config,
                metric_prefix="test",
                rollout_policy=rollout_policy,
            )
            _sync(bundle.device)
            evaluation_seconds = time.perf_counter() - evaluation_start
            test_record = {
                **initial_record,
                "split": "test",
                "selected_step": 0,
                "selection_val_accuracy": best_val_accuracy,
                "wall_time_seconds": time.perf_counter() - start_time,
                "evaluation_seconds": evaluation_seconds,
                **_vllm_policy_fields(rollout_policy),
                **_peak_gpu_memory_metrics(bundle.device),
                **test_metrics,
            }
            _append_jsonl(raw_path, test_record)
            _log_wandb(wandb_run, test_record)
        torch.save(
            {
                "selected_parameters": initial_parameters.cpu(),
                "final_parameters": initial_parameters.cpu(),
                "selected_step": 0,
                "selection_metric": best_val_accuracy,
            },
            checkpoint_path,
        )
        samples_path.write_text(
            json.dumps(final_samples, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _finish_run_artifact(wandb_run, raw_path, checkpoint_path, samples_path)
        return raw_path

    for step in range(1, config.steps + 1):
        batch = _sample_training_batch(train_examples, config.batch_size, prompt_rng)
        prompts = _prompts(bundle.tokenizer, batch)
        set_adapter_grad_enabled(bundle, False)
        _sync(bundle.device)
        rollout_start = time.perf_counter()
        rollout_seed = 20_000 + seed * 1_000 + step
        if rollout_policy is None:
            behavior_policy_fields = _vllm_policy_fields(
                None,
                field_prefix="behavior_policy",
            )
            rollout = generate_sequence_rollouts(
                bundle,
                prompts,
                _reward_callback(batch, config.numeric_shaping_weight),
                group_size=config.group_size,
                max_new_tokens=config.max_new_tokens,
                temperature=config.sampling_temperature,
                max_prompt_tokens=config.max_prompt_tokens,
                seed=rollout_seed,
                scoring_micro_batch_size=config.scoring_micro_batch_size,
                use_frozen_prefix_scoring=config.use_frozen_prefix_scoring,
            )
            behavior_diagnostics = {
                "behavior_hf_logprob_mean_abs_delta": 0.0,
                "behavior_hf_logprob_p99_abs_delta": 0.0,
                "behavior_hf_logprob_max_abs_delta": 0.0,
                "behavior_hf_logprob_mean_signed_delta": 0.0,
                "behavior_hf_max_importance_ratio_deviation": 0.0,
            }
        else:
            # Freeze provenance before generation.  The optimizer sync below
            # advances ``rollout_policy`` to the next policy version.
            behavior_policy_fields = _vllm_policy_fields(
                rollout_policy,
                field_prefix="behavior_policy",
            )
            rollout, behavior_diagnostics = _generate_vllm_sequence_rollouts(
                bundle,
                prompts,
                batch,
                config,
                rollout_policy,
                seed=rollout_seed,
            )
        rollout_provenance_fields = (
            build_rollout_provenance(rollout, seed=rollout_seed).as_record_fields()
            if config.record_rollout_provenance
            else {}
        )
        _sync(bundle.device)
        rollout_and_old_score_seconds = time.perf_counter() - rollout_start
        rollout_exact_reward, exact_zero_advantage_fraction = _exact_rollout_metrics(
            rollout.completions, batch
        )

        _sync(bundle.device)
        optimizer_start = time.perf_counter()
        if method == "bp_grpo":
            assert optimizer is not None
            result = sequence_grpo_step(bundle, rollout, optimizer, bp_config)
        else:
            fo_config = ForwardSequenceConfig(method=method, **config.forward)
            result = forward_sequence_step(
                bundle,
                rollout,
                direction_generator,
                fo_config,
                active_subspace=focus_state,
            )
        _sync_vllm_policy(
            bundle,
            rollout_policy,
            version=f"method={method}/seed={seed}/step={step}",
        )
        next_policy_fields = _vllm_policy_fields(
            rollout_policy,
            field_prefix="next_policy",
        )
        _sync(bundle.device)
        optimizer_seconds = time.perf_counter() - optimizer_start

        # HF generation records its old-policy probabilities with this pass.
        # vLLM supplies exact behavior probabilities and spends the same pass
        # on the mandatory same-weight backend-equivalence gate.
        old_score_calls = math.ceil(rollout.environment_samples / config.scoring_micro_batch_size)
        frozen_prefix_cache = getattr(rollout, "frozen_prefix_cache", None)
        if frozen_prefix_cache is None:
            old_score_full_prefix_calls = old_score_calls
            old_score_suffix_calls = old_score_calls
            old_score_forward_calls = old_score_calls
        else:
            old_score_full_prefix_calls = frozen_prefix_cache.full_prefix_calls
            old_score_suffix_calls = old_score_calls
            old_score_forward_calls = old_score_full_prefix_calls + old_score_suffix_calls
        cumulative_environment_samples += rollout.environment_samples
        cumulative_generated_tokens += rollout.valid_response_tokens
        cumulative_scored_tokens += result.scored_tokens + rollout.valid_response_tokens
        cumulative_forward_calls += result.forward_calls + old_score_forward_calls
        cumulative_full_prefix_calls += result.full_prefix_calls + old_score_full_prefix_calls
        cumulative_suffix_calls += result.suffix_calls + old_score_suffix_calls
        cumulative_backward_calls += result.backward_calls
        cumulative_teacher_forced_examples += (
            result.teacher_forced_examples + rollout.environment_samples
        )
        _sync(bundle.device)
        step_record = {
            "kind": "train_step",
            "method": method,
            "seed": seed,
            "step": step,
            "wall_time_seconds": time.perf_counter() - start_time,
            "rollout_and_old_score_seconds": rollout_and_old_score_seconds,
            "optimizer_seconds": optimizer_seconds,
            "environment_samples": cumulative_environment_samples,
            "generated_tokens": cumulative_generated_tokens,
            "scored_tokens": cumulative_scored_tokens,
            "forward_calls": cumulative_forward_calls,
            "full_prefix_calls": cumulative_full_prefix_calls,
            "suffix_calls": cumulative_suffix_calls,
            "backward_calls": cumulative_backward_calls,
            "teacher_forced_examples": cumulative_teacher_forced_examples,
            "rollout_backend": config.rollout_backend,
            "frozen_prefix_cache_active": frozen_prefix_cache is not None,
            "frozen_prefix_fallback_reason": getattr(
                rollout, "frozen_prefix_fallback_reason", None
            ),
            **rollout_provenance_fields,
            **behavior_policy_fields,
            **next_policy_fields,
            **_peak_gpu_memory_metrics(bundle.device),
            "rollout_exact_reward": rollout_exact_reward,
            "exact_zero_advantage_fraction": exact_zero_advantage_fraction,
            "rollout_shaped_reward": float(rollout.rewards.mean().item()),
            "mean_response_tokens": float(rollout.response_lengths.float().mean().item()),
            "rollout_truncation_fraction": _rollout_truncation_fraction(rollout, bundle.tokenizer),
            **behavior_diagnostics,
            **asdict(result),
        }
        # Result counters are per-step; the canonical top-level counters are cumulative.
        step_record.update(
            {
                "environment_samples": cumulative_environment_samples,
                "generated_tokens": cumulative_generated_tokens,
                "scored_tokens": cumulative_scored_tokens,
                "forward_calls": cumulative_forward_calls,
                "full_prefix_calls": cumulative_full_prefix_calls,
                "suffix_calls": cumulative_suffix_calls,
                "backward_calls": cumulative_backward_calls,
                "teacher_forced_examples": cumulative_teacher_forced_examples,
            }
        )
        _append_jsonl(raw_path, step_record)
        _log_wandb(wandb_run, step_record)

        if step % config.eval_interval == 0 or step == config.steps:
            set_adapter_grad_enabled(bundle, False)
            _sync(bundle.device)
            evaluation_start = time.perf_counter()
            metrics, final_samples = evaluate_gsm8k(
                bundle,
                val_examples,
                config,
                metric_prefix="val",
                rollout_policy=rollout_policy,
            )
            _sync(bundle.device)
            evaluation_seconds = time.perf_counter() - evaluation_start
            evaluation_record = {
                "kind": "evaluation",
                "method": method,
                "seed": seed,
                "step": step,
                "wall_time_seconds": time.perf_counter() - start_time,
                "evaluation_seconds": evaluation_seconds,
                "environment_samples": cumulative_environment_samples,
                "generated_tokens": cumulative_generated_tokens,
                "scored_tokens": cumulative_scored_tokens,
                "forward_calls": cumulative_forward_calls,
                "full_prefix_calls": cumulative_full_prefix_calls,
                "suffix_calls": cumulative_suffix_calls,
                "backward_calls": cumulative_backward_calls,
                "teacher_forced_examples": cumulative_teacher_forced_examples,
                "rollout_backend": config.rollout_backend,
                **_vllm_policy_fields(rollout_policy),
                **_peak_gpu_memory_metrics(bundle.device),
                **metrics,
            }
            current_val_accuracy = float(metrics["val_accuracy"])
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_step = step
                best_parameters = parameter_vector(bundle).clone()
            evaluation_record.update(
                {
                    "best_val_accuracy": best_val_accuracy,
                    "best_step": best_step,
                }
            )
            _append_jsonl(raw_path, evaluation_record)
            _log_wandb(wandb_run, evaluation_record)

    final_parameters = parameter_vector(bundle).clone()
    if config.run_test_evaluation:
        set_adapter_grad_enabled(bundle, False)
        set_parameter_vector(bundle, best_parameters)
        _sync_vllm_policy(
            bundle,
            rollout_policy,
            version=(f"method={method}/seed={seed}/test-selected-step={best_step}"),
        )
        _sync(bundle.device)
        evaluation_start = time.perf_counter()
        test_metrics, final_samples = evaluate_gsm8k(
            bundle,
            test_examples,
            config,
            metric_prefix="test",
            rollout_policy=rollout_policy,
        )
        _sync(bundle.device)
        evaluation_seconds = time.perf_counter() - evaluation_start
        test_record = {
            "kind": "evaluation",
            "split": "test",
            "method": method,
            "seed": seed,
            "step": config.steps,
            "selected_step": best_step,
            "selection_val_accuracy": best_val_accuracy,
            "wall_time_seconds": time.perf_counter() - start_time,
            "evaluation_seconds": evaluation_seconds,
            "environment_samples": cumulative_environment_samples,
            "generated_tokens": cumulative_generated_tokens,
            "scored_tokens": cumulative_scored_tokens,
            "forward_calls": cumulative_forward_calls,
            "full_prefix_calls": cumulative_full_prefix_calls,
            "suffix_calls": cumulative_suffix_calls,
            "backward_calls": cumulative_backward_calls,
            "teacher_forced_examples": cumulative_teacher_forced_examples,
            "rollout_backend": config.rollout_backend,
            **_vllm_policy_fields(rollout_policy),
            **_peak_gpu_memory_metrics(bundle.device),
            **test_metrics,
        }
        _append_jsonl(raw_path, test_record)
        _log_wandb(wandb_run, test_record)

    torch.save(
        {
            "selected_parameters": best_parameters.cpu(),
            "final_parameters": final_parameters.cpu(),
            "selected_step": best_step,
            "selection_metric": best_val_accuracy,
        },
        checkpoint_path,
    )
    samples_path.write_text(
        json.dumps(final_samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _finish_run_artifact(wandb_run, raw_path, checkpoint_path, samples_path)
    del optimizer
    return raw_path


def run_gsm8k_benchmark(
    config: GSM8KExperimentConfig,
    output_dir: str | Path,
) -> Path:
    """Load one calibrated model and run every configured matched trial."""

    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    difficulty = DifficultyFilter(
        min_reasoning_lines=config.min_reasoning_lines,
        max_reasoning_lines=config.max_reasoning_lines,
        max_answer_magnitude=config.max_answer_magnitude,
    )
    official_train = filter_by_difficulty(
        load_gsm8k_split("train", revision=config.dataset_revision), difficulty
    )
    train_and_val = select_seeded_subset(
        official_train,
        config.train_size + config.val_size,
        seed=config.subset_seed,
        namespace="train-and-validation",
    )
    train_examples = train_and_val[: config.train_size]
    val_examples = train_and_val[config.train_size :]
    test_examples: tuple[GSM8KExample, ...] = ()
    excluded_test_ids = _excluded_test_ids(config.test_exclusion_metadata)
    if config.run_test_evaluation:
        official_test = filter_by_difficulty(
            load_gsm8k_split("test", revision=config.dataset_revision), difficulty
        )
        official_test = tuple(
            example for example in official_test if example.example_id not in excluded_test_ids
        )
        test_examples = select_seeded_subset(
            official_test,
            config.test_size,
            seed=config.subset_seed,
            namespace="locked-final-test",
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    calibration_examples = train_examples[: config.calibration_examples]
    calibration_prompts = _prompts(tokenizer, calibration_examples)
    bundle = load_model_bundle(
        model_name=config.model_name,
        calibration_prompts=calibration_prompts,
        # Candidate IDs are unused by sequence RL, but ModelBundle also serves
        # the one-token controlled benchmark and keeps this field explicit.
        candidates=CANDIDATE_ACTIONS[:4],
        adapter_rank=config.adapter_rank,
        adapter_layers=config.adapter_layers,
        adapter_scale=config.adapter_scale,
        dtype=config.dtype,
        device=config.device,
        revision=config.model_revision,
        attention_implementation=config.attention_implementation,
        compile_model_forward=config.compile_model_forward,
        compile_model_forward_mode=config.compile_model_forward_mode,
    )
    forward_template = ForwardSequenceConfig(method="fo_pg", **config.forward)
    if forward_template.use_fused_probes:
        enable_batched_probe_adapters(bundle)
    rollout_policy: OnPolicyVLLMGenerator | None = None
    initial_vllm_receipts: list[dict[str, Any]] = []
    vllm_version: str | None = None
    if config.rollout_backend == "vllm":
        initial_snapshot = capture_residual_adapter_snapshot(bundle)
        vllm_engine = create_vllm_engine(
            config.model_name,
            initial_snapshot,
            revision=config.model_revision,
            dtype=config.dtype,
            max_model_len=config.max_prompt_tokens + config.max_new_tokens,
            kv_cache_memory_bytes=config.vllm_kv_cache_memory_bytes,
            enforce_eager=config.vllm_enforce_eager,
            batch_invariant=config.vllm_batch_invariant,
            enable_v1_multiprocessing=config.vllm_enable_v1_multiprocessing,
            flash_attn_version=config.vllm_flash_attn_version,
            allow_insecure_serialization=config.vllm_allow_insecure_serialization,
            seed=config.subset_seed,
            disable_log_stats=True,
        )
        rollout_policy = OnPolicyVLLMGenerator(vllm_engine)
        initial_vllm_receipts = rollout_policy.sync(
            initial_snapshot,
            version="benchmark-initial",
            include_bases=True,
        )
        import vllm

        vllm_version = str(vllm.__version__)
    initial_parameters = parameter_vector(bundle).clone()
    set_adapter_grad_enabled(bundle, False)
    _sync(bundle.device)
    initial_evaluation_start = time.perf_counter()
    initial_val_metrics, initial_val_samples = evaluate_gsm8k(
        bundle,
        val_examples,
        config,
        metric_prefix="val",
        rollout_policy=rollout_policy,
    )
    _sync(bundle.device)
    initial_evaluation_seconds = time.perf_counter() - initial_evaluation_start

    metadata = {
        "config": asdict(config),
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_split_fingerprints": {
            "train": _example_id_fingerprint(train_examples),
            "validation": _example_id_fingerprint(val_examples),
            "test": _example_id_fingerprint(test_examples),
        },
        "adapter_parameter_count": bundle.parameter_count,
        "adapter_names": bundle.adapter_names,
        "train_example_ids": [example.example_id for example in train_examples],
        "val_example_ids": [example.example_id for example in val_examples],
        "test_example_ids": [example.example_id for example in test_examples],
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "resolved_model_revision": getattr(bundle.model.config, "_commit_hash", None),
        "dataset_revision": config.dataset_revision,
        "rollout_backend": config.rollout_backend,
        "rollout_eos_token_ids": list(_resolve_eos_token_ids(bundle)),
        "behavior_logprob_source": (
            "vllm_processed_sampling_distribution"
            if rollout_policy is not None
            else "hf_teacher_forced_rescore"
        ),
        **_vllm_process_metadata(config, active=rollout_policy is not None),
        "vllm_version": vllm_version,
        "vllm_batch_invariant": config.vllm_batch_invariant,
        "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "vllm_batch_invariant_performance_caveat": (
            "beta deterministic kernels may reduce throughput; measure on the locked H100 run"
            if config.vllm_batch_invariant
            else None
        ),
        "vllm_flash_attn_version": config.vllm_flash_attn_version,
        "vllm_allow_insecure_serialization": config.vllm_allow_insecure_serialization,
        "vllm_insecure_serialization_scope": (
            (
                "trusted_local_enginecore_worker_callable_ipc"
                if config.vllm_enable_v1_multiprocessing
                else "trusted_local_inprocess_direct_call_no_serialization_required"
            )
            if config.vllm_allow_insecure_serialization
            else None
        ),
        "vllm_initial_sync_receipts": initial_vllm_receipts,
        "excluded_test_example_ids": sorted(excluded_test_ids),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(bundle.device),
        "hostname": platform.node(),
        "pid": os.getpid(),
        **_model_runtime_metadata(bundle, config),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for method in config.methods:
        trial_seeds = config.seeds[:1] if method == "base" else config.seeds
        for seed in trial_seeds:
            run_gsm8k_trial(
                bundle,
                train_examples,
                val_examples,
                test_examples,
                config,
                method,
                seed,
                initial_parameters,
                initial_val_metrics,
                initial_val_samples,
                output,
                initial_evaluation_seconds=initial_evaluation_seconds,
                rollout_policy=rollout_policy,
            )
    return output
