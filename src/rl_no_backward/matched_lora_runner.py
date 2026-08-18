"""Unified matched-LoRA GSM8K runner for BP-GRPO versus forward-only NPG.

Every trained method uses the same PEFT LoRA layout and initialization,
precomputed prompt schedule, vLLM-LoRA rollout backend, exact reward,
response budget, Hugging Face old-policy rescore, and vLLM evaluation backend.
The locked-test partition is deliberately unavailable to this gate runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from torch import Tensor

from .evaluation_manifest import (
    EvaluationSplitManifest,
    load_evaluation_split_manifest,
)
from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8KExample,
    exact_match_reward,
    format_prompt,
    load_gsm8k_split,
    select_seeded_subset,
)
from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
    trl_group_standardized_advantages,
)
from .matched_lora_configs import MatchedBackpropConfig
from .matched_lora_forward_only import MatchedForwardConfig, matched_forward_npg_step
from .model import ModelBundle, set_adapter_grad_enabled
from .rollout_provenance import build_rollout_provenance
from .sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs
from .standard_lora import (
    StandardLoRAConfig,
    assert_lora_frozen,
    attach_standard_lora,
    frozen_base_parameter_digest,
    load_lora_state_dict,
    load_shared_lora_initialization,
    lora_parameter_layout,
    lora_state_dict,
    lora_state_digest,
    named_lora_parameters,
    save_shared_lora_initialization,
)
from .vllm_lora_rollout import ReloadableLoRAGenerator, create_standard_lora_vllm_engine

MATCHED_METHODS = ("base", "bp_grpo", "fo_npg")


@dataclass(slots=True)
class MatchedLoRAExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    dataset_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    dtype: str = "bfloat16"
    device: str = "cuda"
    attention_implementation: str = "flash_attention_2"
    train_size: int = 8
    dev_size: int = 256
    evaluation_manifest: str = "configs/gsm8k_standard_lora_eval_manifest.json"
    dev_source_index_receipt: str = (
        "configs/gsm8k_standard_lora_dev_source_indices.json"
    )
    touched_test_exclusions: str = "configs/gsm8k_touched_test_exclusions.json"
    subset_seed: int = 314159
    methods: list[str] = field(default_factory=lambda: ["base", "bp_grpo", "fo_npg"])
    seeds: list[int] = field(default_factory=lambda: [0])
    steps: int = 25
    batch_size: int = 8
    group_size: int = 8
    schedule_mode: str = "fixed_overfit"
    max_prompt_tokens: int = 256
    max_new_tokens: int = 512
    sampling_temperature: float = 1.0
    scoring_micro_batch_size: int = 16
    eval_interval: int = 5
    eval_batch_size: int = 64
    run_test_evaluation: bool = False
    test_size: int = 0
    record_rollout_provenance: bool = True
    rollout_backend: str = "vllm_lora"
    vllm_kv_cache_memory_bytes: int = 2 * 1024**3
    vllm_enforce_eager: bool = False
    vllm_flash_attn_version: int = 2
    vllm_batch_invariant: bool = False
    vllm_enable_v1_multiprocessing: bool = False
    vllm_allow_insecure_serialization: bool = False
    vllm_logprob_mean_abs_tolerance: float = 0.02
    vllm_logprob_p99_abs_tolerance: float = 0.2
    vllm_logprob_max_abs_tolerance: float = 5.0
    minimum_effective_completion_fraction: float = 0.5
    run_projected_gradient_gate: bool = True
    projected_gradient_min_cosine: float = 0.95
    projected_gradient_max_relative_l2: float = 0.25
    minimum_bp_train_accuracy_gain: float = 0.10
    minimum_bp_dev_accuracy_gain: float = 0.0
    expected_lora_parameter_count: int = 1_089_536
    shared_initialization_path_template: str = (
        "artifacts/shared_initializations/qwen25_1p5b_qv_r8_all28_seed{seed}.pt"
    )
    wandb_project: str = "rl-no-backward"
    wandb_mode: str = "offline"
    lora: StandardLoRAConfig = field(default_factory=StandardLoRAConfig)
    objective: MatchedGRPOObjectiveConfig = field(default_factory=MatchedGRPOObjectiveConfig)
    backprop: MatchedBackpropConfig = field(default_factory=MatchedBackpropConfig)
    forward: MatchedForwardConfig = field(default_factory=MatchedForwardConfig)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> MatchedLoRAExperimentConfig:
        payload = dict(mapping)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown matched LoRA config keys: {unknown}")
        payload["lora"] = StandardLoRAConfig.from_mapping(payload.get("lora"))
        payload["objective"] = MatchedGRPOObjectiveConfig(**dict(payload.get("objective") or {}))
        payload["backprop"] = MatchedBackpropConfig(**dict(payload.get("backprop") or {}))
        payload["forward"] = MatchedForwardConfig(**dict(payload.get("forward") or {}))
        return cls(**payload)

    def validate(self) -> None:
        unknown = sorted(set(self.methods) - set(MATCHED_METHODS))
        if unknown or len(set(self.methods)) != len(self.methods):
            raise ValueError(f"methods must be unique members of {MATCHED_METHODS}: {unknown}")
        if set(self.methods) != set(MATCHED_METHODS):
            raise ValueError("the headline gate requires methods [base, bp_grpo, fo_npg]")
        for name in (
            "train_size",
            "dev_size",
            "steps",
            "batch_size",
            "group_size",
            "max_prompt_tokens",
            "max_new_tokens",
            "scoring_micro_batch_size",
            "eval_interval",
            "eval_batch_size",
            "expected_lora_parameter_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.schedule_mode not in {"fixed_overfit", "shuffled_cycles"}:
            raise ValueError("schedule_mode must be fixed_overfit or shuffled_cycles")
        if self.schedule_mode == "fixed_overfit" and self.train_size != self.batch_size:
            raise ValueError("fixed_overfit requires train_size == batch_size")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must contain integers")
        if self.run_test_evaluation or self.test_size != 0:
            raise ValueError("the learning gate must not access locked-test evaluation")
        if not self.record_rollout_provenance:
            raise ValueError("the matched run requires rollout provenance")
        if self.rollout_backend != "vllm_lora":
            raise ValueError("the matched run requires the common vLLM LoRA backend")
        if self.vllm_flash_attn_version != 2:
            raise ValueError("both rollout and evaluation are pinned to vLLM FlashAttention 2")
        if self.vllm_batch_invariant or self.vllm_enable_v1_multiprocessing:
            raise ValueError("the matched vLLM engine requires fast in-process non-invariant mode")
        if self.vllm_allow_insecure_serialization:
            raise ValueError("standard PEFT LoRA reloads do not require callable serialization")
        if self.attention_implementation != "flash_attention_2":
            raise ValueError("HF scoring is pinned to FlashAttention 2")
        if self.dtype != "bfloat16" or not self.device.startswith("cuda"):
            raise ValueError("the H100 matched gate requires CUDA BF16")
        if self.sampling_temperature != 1.0:
            raise ValueError(
                "matched HF-old/vLLM processed-logprob parity is locked to temperature 1.0"
            )
        if self.wandb_mode not in {"offline", "online", "disabled"}:
            raise ValueError("invalid W&B mode")
        if "{seed}" not in self.shared_initialization_path_template:
            raise ValueError("shared initialization path must contain {seed}")
        if not (
            0 <= self.vllm_logprob_mean_abs_tolerance
            <= self.vllm_logprob_p99_abs_tolerance
            <= self.vllm_logprob_max_abs_tolerance
        ):
            raise ValueError("vLLM parity tolerances must satisfy mean <= p99 <= max")
        if not 0.0 < self.minimum_effective_completion_fraction <= 1.0:
            raise ValueError("minimum_effective_completion_fraction must lie in (0, 1]")
        if not self.run_projected_gradient_gate:
            raise ValueError("the H100 learning gate requires the non-updating gradient oracle")
        if not -1.0 <= self.projected_gradient_min_cosine <= 1.0:
            raise ValueError("projected_gradient_min_cosine must lie in [-1, 1]")
        if self.projected_gradient_max_relative_l2 <= 0:
            raise ValueError("projected_gradient_max_relative_l2 must be positive")
        if not 0.0 <= self.minimum_bp_train_accuracy_gain <= 1.0:
            raise ValueError("minimum_bp_train_accuracy_gain must lie in [0, 1]")
        if not 0.0 <= self.minimum_bp_dev_accuracy_gain <= 1.0:
            raise ValueError("minimum_bp_dev_accuracy_gain must lie in [0, 1]")

    @property
    def responses_per_step(self) -> int:
        return self.batch_size * self.group_size

    @property
    def response_budget(self) -> int:
        return self.steps * self.responses_per_step

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora"] = self.lora.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class MatchedRollout:
    rollout: SequenceRolloutBatch
    sampler_token_log_probs: Tensor
    parity: dict[str, float]
    provenance: dict[str, Any]
    behavior_policy_version: str
    behavior_policy_digest: str
    finish_reason_counts: dict[str, int]
    rollout_truncation_fraction: float
    rollout_eos_terminated_fraction: float
    old_score_forward_calls: int
    old_score_teacher_forced_examples: int
    old_score_scored_tokens: int


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON mapping")
    return dict(value)


def _load_data(
    config: MatchedLoRAExperimentConfig,
) -> tuple[
    tuple[GSM8KExample, ...],
    tuple[GSM8KExample, ...],
    EvaluationSplitManifest,
]:
    official_train = load_gsm8k_split("train", revision=config.dataset_revision)
    train = select_seeded_subset(
        official_train,
        config.train_size,
        seed=config.subset_seed,
        namespace="matched-standard-lora-train-v1",
    )
    manifest = load_evaluation_split_manifest(config.evaluation_manifest)
    if manifest.dev_count != config.dev_size:
        raise ValueError(
            "config.dev_size must consume the complete precommitted development partition"
        )
    exclusions = _load_json_mapping(config.touched_test_exclusions).get("test_example_ids")
    if not isinstance(exclusions, list) or len(exclusions) != manifest.excluded_test_count:
        raise ValueError("touched-test exclusions do not match the manifest exclusion count")
    if tuple(exclusions) != manifest.excluded_test_example_ids:
        raise ValueError("touched-test exclusions differ from the evaluation manifest quarantine")
    receipt = _load_json_mapping(config.dev_source_index_receipt)
    expected_receipt_keys = {
        "schema",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "evaluation_manifest_sha256",
        "official_test_count",
        "dev_count",
        "entries",
        "receipt_sha256",
    }
    if set(receipt) != expected_receipt_keys:
        raise ValueError("development source-index receipt has missing or unknown fields")
    if receipt["schema"] != "rl-no-backward-gsm8k-dev-source-index-v1":
        raise ValueError("unsupported development source-index receipt schema")
    serialized_digest = receipt["receipt_sha256"]
    receipt_payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not isinstance(serialized_digest, str) or _json_digest(receipt_payload) != serialized_digest:
        raise ValueError("development source-index receipt digest is invalid")
    if (
        receipt["dataset_id"] != GSM8K_DATASET_ID
        or receipt["dataset_config"] != GSM8K_DATASET_CONFIG
        or receipt["dataset_revision"] != config.dataset_revision
        or receipt["evaluation_manifest_sha256"] != manifest.manifest_sha256
        or receipt["official_test_count"] != manifest.official_test_count
        or receipt["dev_count"] != manifest.dev_count
    ):
        raise ValueError("development source-index receipt does not bind the locked dataset/manifest")
    entries = receipt["entries"]
    if not isinstance(entries, list) or any(
        not isinstance(entry, Mapping) or set(entry) != {"source_index", "example_id"}
        for entry in entries
    ):
        raise TypeError("development source-index receipt entries are malformed")
    indices = [entry["source_index"] for entry in entries]
    receipt_ids = [entry["example_id"] for entry in entries]
    if receipt_ids != list(manifest.dev_example_ids):
        raise ValueError("development source-index receipt IDs differ from manifest dev order")
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or len(set(indices)) != len(indices)
        or any(not 0 <= index < manifest.official_test_count for index in indices)
    ):
        raise ValueError("development source indices are invalid or duplicated")

    official_test = _load_pinned_gsm8k_test_dataset(config.dataset_revision)
    if len(official_test) != manifest.official_test_count:
        raise ValueError("pinned GSM8K test row count differs from the source-index receipt")
    selected_rows = official_test.select(indices)
    dev_examples: list[GSM8KExample] = []
    for entry, row in zip(entries, selected_rows, strict=True):
        if not isinstance(row, Mapping) or "question" not in row or "answer" not in row:
            raise TypeError("selected GSM8K development row is malformed")
        example = GSM8KExample(
            question=row["question"],
            answer=row["answer"],
            split="test",
            source_index=entry["source_index"],
        )
        if example.example_id != entry["example_id"]:
            raise ValueError("selected development row does not match its committed ID")
        dev_examples.append(example)
    dev = tuple(dev_examples)
    if {example.example_id for example in dev} & set(manifest.locked_test_example_ids):
        raise RuntimeError("locked-test rows entered the development evaluation payload")
    return train, dev, manifest


def _load_pinned_gsm8k_test_dataset(revision: str) -> Any:
    """Load the Arrow dataset handle without materializing any Python rows."""

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - installed on H100
        raise RuntimeError("matched evaluation requires the pinned datasets runtime") from error
    return load_dataset(
        GSM8K_DATASET_ID,
        GSM8K_DATASET_CONFIG,
        split="test",
        revision=revision,
    )


def build_prompt_schedule(
    examples: Sequence[GSM8KExample],
    config: MatchedLoRAExperimentConfig,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Precompute the exact prompt order and sampling seed before either method."""

    if len(examples) < config.batch_size:
        raise ValueError("training pool is smaller than the prompt batch")
    rng = random.Random((config.subset_seed << 16) ^ seed)
    pool = list(examples)
    cursor = len(pool)
    schedule: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        if config.schedule_mode == "fixed_overfit":
            batch = list(examples)
        else:
            batch = []
            while len(batch) < config.batch_size:
                if cursor >= len(pool):
                    pool = list(examples)
                    rng.shuffle(pool)
                    cursor = 0
                take = min(config.batch_size - len(batch), len(pool) - cursor)
                batch.extend(pool[cursor : cursor + take])
                cursor += take
        # This stable formula is also independently reconstructed by the
        # artifact validator.  The 1,000-step stride exceeds every locked run.
        rollout_seed = 20_000 + seed * 1_000 + step
        schedule.append(
            {
                "step": step,
                "rollout_seed": rollout_seed,
                "example_ids": [example.example_id for example in batch],
            }
        )
    return schedule


def _format_prompts(tokenizer: Any, examples: Sequence[GSM8KExample]) -> tuple[str, ...]:
    def formatter(messages: Sequence[Mapping[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    return tuple(format_prompt(example.question, formatter) for example in examples)


def _prompt_token_ids(
    tokenizer: Any,
    prompts: Sequence[str],
    max_prompt_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]
        ids = tuple(int(token_id) for token_id in encoded)
        if not ids:
            raise ValueError("prompt encoded to no tokens")
        result.append(ids)
    return tuple(result)


def _left_pad_prompt_ids(
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    width = max(map(len, prompt_token_ids))
    ids = torch.full((len(prompt_token_ids), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(prompt_token_ids):
        values = torch.tensor(row, dtype=torch.long, device=device)
        ids[index, -len(row) :] = values
        mask[index, -len(row) :] = True
    return ids, mask


def _parity_metrics(
    old_hf: Tensor,
    sampler: Tensor,
    mask: Tensor,
    config: MatchedLoRAExperimentConfig,
) -> dict[str, float]:
    differences = (old_hf.float() - sampler.float()).abs()[mask]
    metrics = {
        "vllm_hf_logprob_mean_abs_difference": float(differences.mean().item()),
        "vllm_hf_logprob_p99_abs_difference": float(torch.quantile(differences, 0.99).item()),
        "vllm_hf_logprob_max_abs_difference": float(differences.max().item()),
    }
    if (
        metrics["vllm_hf_logprob_mean_abs_difference"]
        > config.vllm_logprob_mean_abs_tolerance
        or metrics["vllm_hf_logprob_p99_abs_difference"]
        > config.vllm_logprob_p99_abs_tolerance
        or metrics["vllm_hf_logprob_max_abs_difference"]
        > config.vllm_logprob_max_abs_tolerance
    ):
        raise RuntimeError(f"vLLM/HF fixed-completion parity gate failed: {metrics}")
    return metrics


def _old_policy_rescore_accounting(
    environment_samples: int,
    valid_response_tokens: int,
    micro_batch_size: int,
) -> dict[str, int]:
    """Account for the mandatory frozen-HF behavior-policy denominator pass."""

    if environment_samples < 1 or valid_response_tokens < 1 or micro_batch_size < 1:
        raise ValueError("old-policy rescore accounting requires positive counts")
    return {
        "forward_calls": math.ceil(environment_samples / micro_batch_size),
        "teacher_forced_examples": environment_samples,
        "scored_tokens": valid_response_tokens,
    }


def _rollout_finish_telemetry(
    finish_reasons: Sequence[Sequence[str | None]],
) -> tuple[dict[str, int], float, float]:
    flattened = [reason for group in finish_reasons for reason in group]
    if not flattened:
        raise ValueError("rollout finish reasons must not be empty")
    counts: dict[str, int] = {}
    for reason in flattened:
        key = "none" if reason is None else str(reason)
        counts[key] = counts.get(key, 0) + 1
    truncation_fraction = counts.get("length", 0) / len(flattened)
    eos_fraction = (counts.get("stop", 0) + counts.get("eos", 0)) / len(flattened)
    return dict(sorted(counts.items())), truncation_fraction, eos_fraction


def build_matched_rollout(
    bundle: ModelBundle,
    generator: ReloadableLoRAGenerator,
    examples: Sequence[GSM8KExample],
    config: MatchedLoRAExperimentConfig,
    *,
    rollout_seed: int,
) -> MatchedRollout:
    behavior_policy_version = generator.policy_version
    behavior_policy_digest = generator.state_digest
    if behavior_policy_version is None or behavior_policy_digest is None:
        raise RuntimeError("vLLM policy must be synchronized before rollout")
    live_digest = lora_state_digest(bundle.model)
    if behavior_policy_digest != live_digest:
        raise RuntimeError("vLLM behavior policy digest differs from the live HF LoRA")
    prompts = _format_prompts(bundle.tokenizer, examples)
    prompt_token_ids = _prompt_token_ids(bundle.tokenizer, prompts, config.max_prompt_tokens)
    eos_token_ids = (int(bundle.tokenizer.eos_token_id),)
    pad_token_id = int(bundle.tokenizer.pad_token_id)
    generation = generator.generate_grouped(
        prompt_token_ids,
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.sampling_temperature,
        seed=rollout_seed,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
        device=bundle.device,
    )
    if generation.policy_version != behavior_policy_version:
        raise RuntimeError("vLLM generated with an unexpected behavior-policy version")
    if generator.state_digest != behavior_policy_digest:
        raise RuntimeError("vLLM behavior policy changed during grouped generation")
    prompt_input_ids, prompt_attention_mask = _left_pad_prompt_ids(
        prompt_token_ids,
        pad_token_id=pad_token_id,
        device=bundle.device,
    )
    completion_groups: list[tuple[str, ...]] = []
    reward_groups: list[list[float]] = []
    for prompt_index, example in enumerate(examples):
        completions: list[str] = []
        rewards: list[float] = []
        for group_index in range(config.group_size):
            token_ids = generation.response_input_ids[prompt_index, group_index][
                generation.response_mask[prompt_index, group_index]
            ]
            completion = bundle.tokenizer.decode(
                token_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            completions.append(completion)
            rewards.append(exact_match_reward(completion, example.canonical_answer))
        completion_groups.append(tuple(completions))
        reward_groups.append(rewards)
    reward_tensor = torch.tensor(reward_groups, dtype=torch.float32, device=bundle.device)
    advantages = trl_group_standardized_advantages(
        reward_tensor,
        config.objective.advantage_epsilon,
    )
    rollout = SequenceRolloutBatch(
        prompts=prompts,
        completions=tuple(completion_groups),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        response_input_ids=generation.response_input_ids,
        response_mask=generation.response_mask,
        old_token_log_probs=torch.zeros_like(generation.old_token_log_probs),
        rewards=reward_tensor,
        advantages=advantages,
        sampling_temperature=config.sampling_temperature,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
    )
    with torch.inference_mode():
        old_hf = teacher_forced_token_log_probs(
            bundle,
            rollout,
            micro_batch_size=config.scoring_micro_batch_size,
        )
    rollout = replace(rollout, old_token_log_probs=old_hf.detach())
    sampler_logps = generation.old_token_log_probs.detach()
    parity = _parity_metrics(old_hf, sampler_logps, rollout.response_mask, config)
    finish_counts, truncation_fraction, eos_fraction = _rollout_finish_telemetry(
        generation.finish_reasons
    )
    old_score_accounting = _old_policy_rescore_accounting(
        rollout.environment_samples,
        rollout.valid_response_tokens,
        config.scoring_micro_batch_size,
    )

    sampler_source = SimpleNamespace(
        prompt_input_ids=rollout.prompt_input_ids,
        prompt_attention_mask=rollout.prompt_attention_mask,
        response_input_ids=rollout.response_input_ids,
        response_mask=rollout.response_mask,
        old_token_log_probs=sampler_logps,
    )
    sampler_provenance = build_rollout_provenance(sampler_source, seed=rollout_seed)
    hf_provenance = build_rollout_provenance(rollout, seed=rollout_seed)
    reward_digest = _tensor_digest(reward_tensor)
    provenance = {
        **sampler_provenance.as_record_fields(),
        "hf_old_logprob_digest": hf_provenance.behavior_logprob_digest,
        "reward_digest": reward_digest,
        "prompt_example_ids": [example.example_id for example in examples],
        "prompt_example_ids_digest": _json_digest([example.example_id for example in examples]),
    }
    return MatchedRollout(
        rollout=rollout,
        sampler_token_log_probs=sampler_logps,
        parity=parity,
        provenance=provenance,
        behavior_policy_version=behavior_policy_version,
        behavior_policy_digest=behavior_policy_digest,
        finish_reason_counts=finish_counts,
        rollout_truncation_fraction=truncation_fraction,
        rollout_eos_terminated_fraction=eos_fraction,
        old_score_forward_calls=old_score_accounting["forward_calls"],
        old_score_teacher_forced_examples=old_score_accounting[
            "teacher_forced_examples"
        ],
        old_score_scored_tokens=old_score_accounting["scored_tokens"],
    )


def evaluate_with_common_backend(
    bundle: ModelBundle,
    generator: ReloadableLoRAGenerator,
    examples: Sequence[GSM8KExample],
    config: MatchedLoRAExperimentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = _format_prompts(bundle.tokenizer, examples)
    samples: list[dict[str, Any]] = []
    finish_reasons: list[tuple[str | None, ...]] = []
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = examples[start : start + config.eval_batch_size]
        batch_prompts = prompts[start : start + config.eval_batch_size]
        prompt_ids = _prompt_token_ids(bundle.tokenizer, batch_prompts, config.max_prompt_tokens)
        generated = generator.generate_greedy(
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            eos_token_ids=(int(bundle.tokenizer.eos_token_id),),
        )
        finish_reasons.append(generated.finish_reasons)
        for example, response_ids, finish_reason in zip(
            batch_examples,
            generated.response_token_ids,
            generated.finish_reasons,
            strict=True,
        ):
            completion = bundle.tokenizer.decode(
                list(response_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            correct = bool(exact_match_reward(completion, example.canonical_answer))
            samples.append(
                {
                    "example_id": example.example_id,
                    "question": example.question,
                    "reference_answer": example.canonical_answer,
                    "completion": completion,
                    "correct": correct,
                    "response_tokens": len(response_ids),
                    "finish_reason": finish_reason,
                }
            )
    accuracy = sum(sample["correct"] for sample in samples) / len(samples)
    mean_tokens = sum(sample["response_tokens"] for sample in samples) / len(samples)
    finish_counts, truncation_fraction, eos_fraction = _rollout_finish_telemetry(finish_reasons)
    return {
        "accuracy": accuracy,
        "mean_response_tokens": mean_tokens,
        "finish_reason_counts": finish_counts,
        "truncation_fraction": truncation_fraction,
        "eos_terminated_fraction": eos_fraction,
    }, samples


def _load_bundle(
    config: MatchedLoRAExperimentConfig,
    *,
    seed: int,
) -> tuple[ModelBundle, str, str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = snapshot_download(config.model_name, revision=config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation=config.attention_implementation,
    )
    model = attach_standard_lora(model, config.lora, initialization_seed=seed)
    model.to(torch.device(config.device))
    model.eval()
    layout = lora_parameter_layout(model)
    if layout.parameter_count != config.expected_lora_parameter_count:
        raise RuntimeError(
            f"actual LoRA count {layout.parameter_count} != {config.expected_lora_parameter_count}"
        )
    initialization_path = Path(config.shared_initialization_path_template.format(seed=seed))
    if initialization_path.exists():
        initialization_digest = load_shared_lora_initialization(
            initialization_path,
            model,
            config.lora,
        )
    else:
        save_shared_lora_initialization(initialization_path, model, config.lora)
        initialization_digest = lora_state_digest(model)
    adapter_names = [name for name, _ in named_lora_parameters(model)]
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        candidate_token_ids=torch.empty(0, dtype=torch.long, device=config.device),
        adapter_names=adapter_names,
        device=torch.device(config.device),
        model_name=config.model_name,
    )
    return bundle, snapshot, initialization_digest


def _run_vllm_same_id_reload_gate(
    bundle: ModelBundle,
    generator: ReloadableLoRAGenerator,
    prompt_token_ids: tuple[int, ...],
    export_root: Path,
    *,
    version_counter: list[int],
    reload_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove change/reload/restore semantics for vLLM's stable LoRA ID."""

    initial_state = lora_state_dict(bundle.model)
    initial_digest = lora_state_digest(initial_state)

    def sync_and_probe(label: str) -> dict[str, Any]:
        version_counter[0] += 1
        receipt = generator.sync(
            bundle.model,
            export_root,
            version=version_counter[0],
            policy_version=f"same-id-reload-gate/{label}",
        )
        reload_receipts.append(asdict(receipt))
        _synchronize_cuda()
        probe = generator.probe_next_token(prompt_token_ids)
        _synchronize_cuda()
        if probe.state_digest != lora_state_digest(bundle.model):
            raise RuntimeError("reload probe state digest differs from live HF adapter")
        return asdict(probe)

    initial_probe = sync_and_probe("initial")
    with torch.inference_mode():
        changed_tensors = 0
        for name, parameter in named_lora_parameters(bundle.model):
            if ".lora_B." in name or name.startswith("lora_B."):
                parameter.add_(0.05)
                changed_tensors += 1
    if changed_tensors == 0:
        raise RuntimeError("reload gate could not find structural LoRA-B tensors")
    changed_digest = lora_state_digest(bundle.model)
    if changed_digest == initial_digest:
        raise RuntimeError("reload-gate perturbation did not change the LoRA state digest")
    changed_probe = sync_and_probe("changed")
    changed_output = (
        changed_probe["token_id"] != initial_probe["token_id"]
        or abs(
            changed_probe["selected_token_logprob"]
            - initial_probe["selected_token_logprob"]
        )
        > 1.0e-6
    )
    if not changed_output:
        raise RuntimeError("same-ID LoRA reload did not change the vLLM next-token signature")

    load_lora_state_dict(bundle.model, initial_state)
    if lora_state_digest(bundle.model) != initial_digest:
        raise RuntimeError("reload gate failed to restore the immutable LoRA initialization")
    restored_probe = sync_and_probe("restored")
    restored_delta = abs(
        restored_probe["selected_token_logprob"] - initial_probe["selected_token_logprob"]
    )
    if (
        restored_probe["token_id"] != initial_probe["token_id"]
        or restored_delta > 1.0e-5
        or restored_probe["state_digest"] != initial_digest
    ):
        raise RuntimeError("same-ID LoRA restore did not reproduce the vLLM signature")
    return {
        "passed": True,
        "lora_int_id": generator.lora_int_id,
        "changed_lora_b_tensor_count": changed_tensors,
        "perturbation": 0.05,
        "initial": initial_probe,
        "changed": changed_probe,
        "restored": restored_probe,
        "restored_selected_logprob_abs_difference": restored_delta,
    }


def _init_wandb(
    config: MatchedLoRAExperimentConfig,
    output: Path,
    *,
    method: str,
    seed: int,
) -> Any | None:
    if config.wandb_mode == "disabled":
        return None
    import wandb

    # W&B itself appends the single ``wandb/`` component.  Pointing both
    # variables at output/wandb would create output/wandb/wandb and evade the
    # artifact validator's exact run count.
    output.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = config.wandb_mode
    os.environ["WANDB_DIR"] = str(output.resolve())
    return wandb.init(
        project=config.wandb_project,
        name=f"matched-lora-{method}-seed-{seed}",
        group="matched-standard-lora-gsm8k",
        config={**config.as_dict(), "method": method, "seed": seed},
        dir=str(output.resolve()),
        mode=config.wandb_mode,
        reinit=True,
    )


def _peak_memory() -> dict[str, int]:
    return {
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _synchronize_cuda() -> None:
    """Fence asynchronous kernels at every persisted timing boundary."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluation_record(
    *,
    method: str,
    seed: int,
    step: int,
    accuracy: float,
    mean_tokens: float,
    wall_time: float,
    evaluation_seconds: float,
    environment_samples: int,
    counters: Mapping[str, int],
    finish_reason_counts: Mapping[str, int],
    truncation_fraction: float,
    eos_terminated_fraction: float,
    policy_sync_seconds: float,
) -> dict[str, Any]:
    return {
        "kind": "evaluation",
        "split": "validation",
        "method": method,
        "seed": seed,
        "step": step,
        "wall_time_seconds": wall_time,
        "evaluation_seconds": evaluation_seconds,
        "policy_sync_seconds": policy_sync_seconds,
        "environment_samples": environment_samples,
        "generated_tokens": counters["generated_tokens"],
        "scored_tokens": counters["scored_tokens"],
        "forward_calls": counters["forward_calls"],
        "full_prefix_calls": counters["forward_calls"],
        "suffix_calls": 0,
        "backward_calls": counters["backward_calls"],
        "teacher_forced_examples": counters["teacher_forced_examples"],
        "cumulative_environment_samples": environment_samples,
        "cumulative_generated_tokens": counters["generated_tokens"],
        "cumulative_scored_tokens": counters["scored_tokens"],
        "cumulative_forward_calls": counters["forward_calls"],
        "cumulative_backward_calls": counters["backward_calls"],
        "cumulative_teacher_forced_examples": counters["teacher_forced_examples"],
        "val_accuracy": accuracy,
        "val_exact_reward": accuracy,
        "val_mean_response_tokens": mean_tokens,
        "val_finish_reason_counts": dict(finish_reason_counts),
        "val_truncation_fraction": truncation_fraction,
        "val_eos_terminated_fraction": eos_terminated_fraction,
        **_peak_memory(),
    }


def _save_trial_artifacts(
    output: Path,
    *,
    method: str,
    seed: int,
    selected_step: int,
    selection_accuracy: float,
    selected_state: Mapping[str, Tensor],
    samples: Sequence[Mapping[str, Any]],
) -> None:
    checkpoint = output / "checkpoints" / f"gsm8k_{method}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "seed": seed,
            "selected_step": selected_step,
            "selection_val_accuracy": selection_accuracy,
            "lora_state": dict(selected_state),
        },
        checkpoint,
    )
    samples_path = output / "samples" / f"gsm8k_{method}_seed{seed}.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(list(samples), indent=2) + "\n", encoding="utf-8")
    selection_path = output / "selection" / f"gsm8k_{method}_seed{seed}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema": "rl-no-backward-validation-selection-v1",
                "method": method,
                "seed": seed,
                "selected_step": selected_step,
                "selection_split": "development",
                "selection_metric": "exact_match",
                "tie_breaker": "latest_checkpoint",
                "selection_val_accuracy": selection_accuracy,
                "selected_lora_state_digest": lora_state_digest(selected_state),
                "checkpoint_path": str(checkpoint.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_learning_gate_receipt(
    output: Path,
    *,
    method: str,
    seed: int,
    payload: Mapping[str, Any],
) -> Path:
    destination = output / "learning_gate" / f"gsm8k_{method}_seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _run_trial(
    bundle: ModelBundle,
    generator: ReloadableLoRAGenerator,
    train_by_id: Mapping[str, GSM8KExample],
    dev_examples: Sequence[GSM8KExample],
    schedule: Sequence[Mapping[str, Any]],
    initial_state: Mapping[str, Tensor],
    initialization_digest: str,
    config: MatchedLoRAExperimentConfig,
    output: Path,
    *,
    method: str,
    seed: int,
    version_counter: list[int],
    first_rollout_reference: dict[int, tuple[str, str]],
    reload_receipts: list[dict[str, Any]],
) -> None:
    load_lora_state_dict(bundle.model, initial_state)
    if lora_state_digest(bundle.model) != initialization_digest:
        raise RuntimeError("trial did not start from the shared LoRA initialization")
    if method in {"base", "fo_npg"}:
        set_adapter_grad_enabled(bundle, False)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        assert_lora_frozen(bundle.model)
    raw_path = output / "raw" / f"gsm8k_{method}_seed{seed}.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to append to existing trial {raw_path}")
    run = _init_wandb(config, output, method=method, seed=seed)
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    counters = {
        "generated_tokens": 0,
        "scored_tokens": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "teacher_forced_examples": 0,
    }
    export_root = output / "vllm_lora_exports" / f"{method}_seed{seed}"

    def sync(purpose: str, step: int) -> float:
        _synchronize_cuda()
        sync_started = time.perf_counter()
        version_counter[0] += 1
        receipt = generator.sync(
            bundle.model,
            export_root,
            version=version_counter[0],
            policy_version=f"{method}/seed={seed}/{purpose}/step={step}",
        )
        if receipt.state_digest != lora_state_digest(bundle.model):
            raise RuntimeError("vLLM export digest differs from live HF LoRA")
        reload_receipts.append(asdict(receipt))
        _synchronize_cuda()
        return time.perf_counter() - sync_started

    initial_sync_seconds = sync("initial-eval", 0)
    _synchronize_cuda()
    evaluation_start = time.perf_counter()
    metrics, samples = evaluate_with_common_backend(bundle, generator, dev_examples, config)
    _synchronize_cuda()
    evaluation = _evaluation_record(
        method=method,
        seed=seed,
        step=0,
        accuracy=metrics["accuracy"],
        mean_tokens=metrics["mean_response_tokens"],
        wall_time=time.perf_counter() - start_time,
        evaluation_seconds=time.perf_counter() - evaluation_start,
        environment_samples=0,
        counters=counters,
        finish_reason_counts=metrics["finish_reason_counts"],
        truncation_fraction=metrics["truncation_fraction"],
        eos_terminated_fraction=metrics["eos_terminated_fraction"],
        policy_sync_seconds=initial_sync_seconds,
    )
    evaluation.update(best_val_accuracy=metrics["accuracy"], best_step=0)
    _append_jsonl(raw_path, evaluation)
    train_gate_examples = tuple(
        train_by_id[example_id] for example_id in schedule[0]["example_ids"]
    )
    initial_train_metrics, _ = evaluate_with_common_backend(
        bundle,
        generator,
        train_gate_examples,
        config,
    )
    if run is not None:
        run.log({"checkpoint_step": 0, "validation/exact_match": metrics["accuracy"]})
    best_accuracy = metrics["accuracy"]
    best_step = 0
    best_state = lora_state_dict(bundle.model)
    best_samples = samples

    if method == "base":
        _write_learning_gate_receipt(
            output,
            method=method,
            seed=seed,
            payload={
                "schema": "rl-no-backward-matched-learning-gate-v1",
                "passed": True,
                "role": "non-updating baseline",
                "initial_train_accuracy": initial_train_metrics["accuracy"],
                "initial_dev_accuracy": metrics["accuracy"],
                "initial_lora_digest": initialization_digest,
            },
        )
        _save_trial_artifacts(
            output,
            method=method,
            seed=seed,
            selected_step=0,
            selection_accuracy=best_accuracy,
            selected_state=best_state,
            samples=best_samples,
        )
        if run is not None:
            run.summary.update({"selected_step": 0, "selection_val_accuracy": best_accuracy})
            run.finish()
        return

    optimizer = None
    scheduler = None
    if method == "bp_grpo":
        from .matched_lora_backprop import (
            make_matched_lora_optimizer,
            make_matched_lr_scheduler,
            matched_backprop_grpo_step,
        )

        optimizer = make_matched_lora_optimizer(bundle, config.backprop)
        scheduler = make_matched_lr_scheduler(
            optimizer,
            config.backprop,
            total_steps=config.steps,
        )
    else:
        set_adapter_grad_enabled(bundle, False)
    direction_generator = torch.Generator(device=bundle.device).manual_seed(seed + 70_000)
    informative_group_observed = False
    nonzero_policy_step_observed = False
    finite_nonzero_bp_gradient_observed = False
    accepted_nonzero_fo_step_observed = False
    final_dev_metrics = metrics
    final_train_metrics = initial_train_metrics

    for schedule_entry in schedule:
        step = int(schedule_entry["step"])
        rollout_seed = int(schedule_entry["rollout_seed"])
        examples = tuple(train_by_id[example_id] for example_id in schedule_entry["example_ids"])
        policy_sync_seconds = sync("rollout", step)
        before_digest = lora_state_digest(bundle.model)
        if generator.state_digest != before_digest:
            raise RuntimeError("rollout behavior-policy digest does not match HF old policy")
        _synchronize_cuda()
        rollout_started = time.perf_counter()
        matched_rollout = build_matched_rollout(
            bundle,
            generator,
            examples,
            config,
            rollout_seed=rollout_seed,
        )
        _synchronize_cuda()
        rollout_seconds = time.perf_counter() - rollout_started
        if matched_rollout.behavior_policy_digest != before_digest:
            raise RuntimeError("recorded behavior policy is not the PPO old-policy center")
        if step == 1:
            receipt = (
                str(matched_rollout.provenance["rollout_token_digest"]),
                str(matched_rollout.provenance["behavior_logprob_digest"]),
            )
            previous = first_rollout_reference.get(seed)
            if previous is None:
                first_rollout_reference[seed] = receipt
            elif previous != receipt:
                raise RuntimeError(
                    "identical initial policy/schedule/seed produced different first rollouts across methods"
                )

        _synchronize_cuda()
        optimizer_started = time.perf_counter()
        if method == "bp_grpo":
            assert optimizer is not None
            result = matched_backprop_grpo_step(
                bundle,
                matched_rollout.rollout,
                matched_rollout.sampler_token_log_probs,
                optimizer,
                config.objective,
                config.backprop,
                scheduler,
            )
            fisher_condition = None
            derivative_variance = None
            line_search_trials = 0
        else:
            result = matched_forward_npg_step(
                bundle,
                matched_rollout.rollout,
                matched_rollout.sampler_token_log_probs,
                direction_generator,
                config.objective,
                config.forward,
            )
            fisher_condition = result.fisher_condition
            derivative_variance = result.derivative_variance
            line_search_trials = result.line_search_trials
        _synchronize_cuda()
        optimizer_seconds = time.perf_counter() - optimizer_started
        after_digest = lora_state_digest(bundle.model)
        informative_group_observed |= result.zero_advantage_fraction < 1.0
        nonzero_policy_step_observed |= result.step_norm > 0.0 and before_digest != after_digest
        if method == "bp_grpo":
            finite_nonzero_bp_gradient_observed |= (
                math.isfinite(result.projected_gradient_norm)
                and result.projected_gradient_norm > 0.0
            )
        else:
            accepted_nonzero_fo_step_observed |= (
                result.accepted and result.step_norm > 0.0 and before_digest != after_digest
            )
        response_tokens = matched_rollout.rollout.valid_response_tokens
        counters["generated_tokens"] += response_tokens
        counters["scored_tokens"] += (
            matched_rollout.old_score_scored_tokens + result.scored_tokens
        )
        counters["forward_calls"] += (
            matched_rollout.old_score_forward_calls + result.forward_calls
        )
        counters["backward_calls"] += result.backward_calls
        counters["teacher_forced_examples"] += (
            matched_rollout.old_score_teacher_forced_examples
            + result.teacher_forced_examples
        )
        correction = detached_inference_correction(
            matched_rollout.rollout.old_token_log_probs,
            matched_rollout.sampler_token_log_probs,
            matched_rollout.rollout.response_mask,
            config.objective,
        )
        if config.objective.inference_correction_mode == "sequence_mask":
            valid_correction = correction.squeeze(-1)
            effective_completion_mask = valid_correction.ne(0)
        else:
            valid_correction = correction[matched_rollout.rollout.response_mask]
            effective_completion_mask = (
                correction.ne(0) & matched_rollout.rollout.response_mask
            ).any(dim=-1)
        effective_completion_count = int(effective_completion_mask.sum().item())
        effective_completion_fraction = (
            effective_completion_count / matched_rollout.rollout.environment_samples
        )
        if effective_completion_fraction < config.minimum_effective_completion_fraction:
            raise RuntimeError(
                "vLLM/HF inference correction annihilated too many sampled completions"
            )
        train_record = {
            "kind": "train_step",
            "method": method,
            "seed": seed,
            "step": step,
            "wall_time_seconds": time.perf_counter() - start_time,
            "rollout_and_old_score_seconds": rollout_seconds,
            "optimizer_seconds": optimizer_seconds,
            "policy_sync_seconds": policy_sync_seconds,
            "training_phase_seconds": (
                policy_sync_seconds + rollout_seconds + optimizer_seconds
            ),
            "environment_samples": step * config.responses_per_step,
            "generated_tokens": counters["generated_tokens"],
            "scored_tokens": counters["scored_tokens"],
            "forward_calls": counters["forward_calls"],
            "full_prefix_calls": counters["forward_calls"],
            "suffix_calls": 0,
            "backward_calls": counters["backward_calls"],
            "teacher_forced_examples": counters["teacher_forced_examples"],
            "cumulative_environment_samples": step * config.responses_per_step,
            "cumulative_generated_tokens": counters["generated_tokens"],
            "cumulative_scored_tokens": counters["scored_tokens"],
            "cumulative_forward_calls": counters["forward_calls"],
            "cumulative_backward_calls": counters["backward_calls"],
            "cumulative_teacher_forced_examples": counters["teacher_forced_examples"],
            "rollout_exact_reward": result.reward_mean,
            "rollout_shaped_reward": result.reward_mean,
            "reward_mean": result.reward_mean,
            "mean_response_tokens": float(
                matched_rollout.rollout.response_lengths.float().mean().item()
            ),
            "zero_advantage_fraction": result.zero_advantage_fraction,
            "accepted": result.accepted,
            "empirical_kl": result.empirical_kl,
            "surrogate_improvement": result.surrogate_improvement,
            "step_norm": result.step_norm,
            "projected_gradient_norm": result.projected_gradient_norm,
            "fisher_condition": fisher_condition,
            "derivative_variance": derivative_variance,
            "line_search_trials": line_search_trials,
            "policy_evaluations": result.policy_evaluations,
            "learning_rate": (
                float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None
            ),
            "lora_digest_before": before_digest,
            "lora_digest_after": after_digest,
            "behavior_policy_version": matched_rollout.behavior_policy_version,
            "behavior_policy_digest": matched_rollout.behavior_policy_digest,
            "old_policy_rescore_forward_calls": matched_rollout.old_score_forward_calls,
            "old_policy_rescore_teacher_forced_examples": (
                matched_rollout.old_score_teacher_forced_examples
            ),
            "old_policy_rescore_scored_tokens": matched_rollout.old_score_scored_tokens,
            "rollout_finish_reason_counts": matched_rollout.finish_reason_counts,
            "rollout_truncation_fraction": matched_rollout.rollout_truncation_fraction,
            "rollout_eos_terminated_fraction": (
                matched_rollout.rollout_eos_terminated_fraction
            ),
            "inference_ratio_mean": float(valid_correction.mean().item()),
            "inference_ratio_min": float(valid_correction.min().item()),
            "inference_ratio_max": float(valid_correction.max().item()),
            "zero_inference_correction_fraction": float(
                valid_correction.eq(0).float().mean().item()
            ),
            "effective_completion_count": effective_completion_count,
            "effective_completion_fraction": effective_completion_fraction,
            "frozen_prefix_scoring": False,
            "fused_residual_probe_scoring": False,
            **matched_rollout.parity,
            **matched_rollout.provenance,
            **_peak_memory(),
        }
        _append_jsonl(raw_path, train_record)
        if run is not None:
            run.log(
                {
                    "training_step": step,
                    "train/exact_reward": result.reward_mean,
                    "train/optimizer_seconds": optimizer_seconds,
                    "train/rollout_seconds": rollout_seconds,
                    "train/policy_sync_seconds": policy_sync_seconds,
                    "train/training_phase_seconds": (
                        policy_sync_seconds + rollout_seconds + optimizer_seconds
                    ),
                    "train/step_norm": result.step_norm,
                    "train/vllm_hf_logprob_mean_abs_difference": matched_rollout.parity[
                        "vllm_hf_logprob_mean_abs_difference"
                    ],
                }
            )

        if step % config.eval_interval == 0 or step == config.steps:
            validation_sync_seconds = sync("validation", step)
            _synchronize_cuda()
            evaluation_start = time.perf_counter()
            eval_metrics, eval_samples = evaluate_with_common_backend(
                bundle,
                generator,
                dev_examples,
                config,
            )
            _synchronize_cuda()
            evaluation = _evaluation_record(
                method=method,
                seed=seed,
                step=step,
                accuracy=eval_metrics["accuracy"],
                mean_tokens=eval_metrics["mean_response_tokens"],
                wall_time=time.perf_counter() - start_time,
                evaluation_seconds=time.perf_counter() - evaluation_start,
                environment_samples=step * config.responses_per_step,
                counters=counters,
                finish_reason_counts=eval_metrics["finish_reason_counts"],
                truncation_fraction=eval_metrics["truncation_fraction"],
                eos_terminated_fraction=eval_metrics["eos_terminated_fraction"],
                policy_sync_seconds=validation_sync_seconds,
            )
            # Prefer the latest checkpoint on an exact dev tie so a learned
            # policy is not silently replaced by its step-zero initialization.
            if eval_metrics["accuracy"] >= best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                best_step = step
                best_state = lora_state_dict(bundle.model)
                best_samples = eval_samples
            final_dev_metrics = eval_metrics
            if step == config.steps:
                final_train_metrics, _ = evaluate_with_common_backend(
                    bundle,
                    generator,
                    train_gate_examples,
                    config,
                )
            evaluation.update(best_val_accuracy=best_accuracy, best_step=best_step)
            _append_jsonl(raw_path, evaluation)
            if run is not None:
                run.log(
                    {
                        "checkpoint_step": step,
                        "validation/exact_match": eval_metrics["accuracy"],
                    }
                )

    if counters["backward_calls"] == 0 and method == "bp_grpo":
        raise RuntimeError("BP trial completed without reverse-mode calls")
    if counters["backward_calls"] != 0 and method == "fo_npg":
        raise RuntimeError("forward-only trial recorded reverse-mode calls")
    if config.steps * config.responses_per_step != config.response_budget:
        raise RuntimeError("response-budget arithmetic changed during the trial")
    live_final_digest = lora_state_digest(bundle.model)
    train_accuracy_gain = (
        final_train_metrics["accuracy"] - initial_train_metrics["accuracy"]
    )
    dev_accuracy_gain = final_dev_metrics["accuracy"] - metrics["accuracy"]
    learning_checks = {
        "policy_digest_changed": live_final_digest != initialization_digest,
        "nonzero_policy_step_observed": nonzero_policy_step_observed,
        "informative_group_observed": informative_group_observed,
        "finite_nonzero_bp_gradient_observed": (
            finite_nonzero_bp_gradient_observed if method == "bp_grpo" else None
        ),
        "accepted_nonzero_fo_step_observed": (
            accepted_nonzero_fo_step_observed if method == "fo_npg" else None
        ),
        "bp_train_gain_met": (
            train_accuracy_gain >= config.minimum_bp_train_accuracy_gain
            if method == "bp_grpo"
            else None
        ),
        "bp_dev_gain_met": (
            dev_accuracy_gain >= config.minimum_bp_dev_accuracy_gain
            if method == "bp_grpo"
            else None
        ),
    }
    required_checks = [
        learning_checks["policy_digest_changed"],
        learning_checks["nonzero_policy_step_observed"],
        learning_checks["informative_group_observed"],
    ]
    if method == "bp_grpo":
        required_checks.extend(
            [
                learning_checks["finite_nonzero_bp_gradient_observed"],
                learning_checks["bp_train_gain_met"],
                learning_checks["bp_dev_gain_met"],
            ]
        )
    else:
        required_checks.append(learning_checks["accepted_nonzero_fo_step_observed"])
    learning_gate_passed = all(check is True for check in required_checks)
    _write_learning_gate_receipt(
        output,
        method=method,
        seed=seed,
        payload={
            "schema": "rl-no-backward-matched-learning-gate-v1",
            "passed": learning_gate_passed,
            "method": method,
            "seed": seed,
            "initial_lora_digest": initialization_digest,
            "live_final_lora_digest": live_final_digest,
            "initial_train_accuracy": initial_train_metrics["accuracy"],
            "final_train_accuracy": final_train_metrics["accuracy"],
            "train_accuracy_gain": train_accuracy_gain,
            "minimum_bp_train_accuracy_gain": config.minimum_bp_train_accuracy_gain,
            "initial_dev_accuracy": metrics["accuracy"],
            "final_dev_accuracy": final_dev_metrics["accuracy"],
            "dev_accuracy_gain": dev_accuracy_gain,
            "minimum_bp_dev_accuracy_gain": config.minimum_bp_dev_accuracy_gain,
            "checks": learning_checks,
        },
    )
    if not learning_gate_passed:
        raise RuntimeError(f"{method} failed the predeclared learning gate: {learning_checks}")
    load_lora_state_dict(bundle.model, best_state)
    _save_trial_artifacts(
        output,
        method=method,
        seed=seed,
        selected_step=best_step,
        selection_accuracy=best_accuracy,
        selected_state=best_state,
        samples=best_samples,
    )
    if run is not None:
        run.summary.update(
            {
                "selected_step": best_step,
                "selection_val_accuracy": best_accuracy,
                "response_budget": config.response_budget,
                "adapter_parameter_count": config.expected_lora_parameter_count,
            }
        )
        run.finish()


def _runtime_metadata() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "safetensors",
        "vllm",
        "flash-attn",
        "wandb",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def _run_projected_gradient_oracle_child(
    config: MatchedLoRAExperimentConfig,
    output: Path,
    *,
    seed: int,
    expected_source_commit: str,
) -> Path:
    """Run the non-updating BP-vs-FD coordinate oracle in its own process."""

    if _git_output("rev-parse", "HEAD") != expected_source_commit:
        raise RuntimeError("gradient-oracle source commit differs from its parent")
    source_status = _git_output("status", "--porcelain")
    if source_status is None or source_status:
        raise RuntimeError("gradient oracle requires the clean parent worktree")
    train_examples, _, _ = _load_data(config)
    train_by_id = {example.example_id: example for example in train_examples}
    schedule = json.loads((output / "schedules" / f"seed{seed}.json").read_text(encoding="utf-8"))
    first_entry = schedule[0]
    examples = tuple(train_by_id[example_id] for example_id in first_entry["example_ids"])
    bundle, snapshot, initialization_digest = _load_bundle(config, seed=seed)
    initial_digest = lora_state_digest(bundle.model)
    if initial_digest != initialization_digest:
        raise RuntimeError("gradient oracle did not load the shared LoRA initialization")
    base_before = frozen_base_parameter_digest(bundle.model)
    engine = create_standard_lora_vllm_engine(
        config.model_name,
        revision=config.model_revision,
        dtype=config.dtype,
        max_model_len=config.max_prompt_tokens + config.max_new_tokens,
        max_lora_rank=config.lora.rank,
        kv_cache_memory_bytes=config.vllm_kv_cache_memory_bytes,
        enforce_eager=config.vllm_enforce_eager,
        flash_attn_version=config.vllm_flash_attn_version,
        max_num_seqs=config.responses_per_step,
        seed=seed,
    )
    generator = ReloadableLoRAGenerator(engine)
    reload_receipt = generator.sync(
        bundle.model,
        output / "vllm_lora_exports" / f"gradient_oracle_seed{seed}",
        version=1,
        policy_version=f"gradient-oracle/seed={seed}/initial",
    )
    _synchronize_cuda()
    rollout = build_matched_rollout(
        bundle,
        generator,
        examples,
        config,
        rollout_seed=int(first_entry["rollout_seed"]),
    )
    from .matched_lora_gradient_gate import fixed_rollout_projected_gradient_gate

    report = fixed_rollout_projected_gradient_gate(
        bundle,
        rollout.rollout,
        rollout.sampler_token_log_probs,
        config.objective,
        config.forward,
        torch.Generator(device=bundle.device).manual_seed(seed + 80_000),
        prompt_groups_per_micro_batch=config.backprop.prompt_groups_per_micro_batch,
    )
    base_after = frozen_base_parameter_digest(bundle.model)
    passed = (
        report.parameter_integrity
        and base_before == base_after
        and report.directions == config.forward.directions == 8
        and report.cosine_similarity >= config.projected_gradient_min_cosine
        and report.relative_l2_error <= config.projected_gradient_max_relative_l2
    )
    payload = {
        "schema": "rl-no-backward-matched-lora-gradient-oracle-v1",
        "passed": passed,
        "seed": seed,
        "source_commit": expected_source_commit,
        "model_snapshot": snapshot,
        "adapter_parameter_count": config.expected_lora_parameter_count,
        "shared_initialization_digest": initialization_digest,
        "frozen_base_parameter_digest_before": base_before,
        "frozen_base_parameter_digest_after": base_after,
        "rollout_token_digest": rollout.provenance["rollout_token_digest"],
        "behavior_logprob_digest": rollout.provenance["behavior_logprob_digest"],
        "vllm_reload_receipt": asdict(reload_receipt),
        "thresholds": {
            "minimum_cosine_similarity": config.projected_gradient_min_cosine,
            "maximum_relative_l2_error": config.projected_gradient_max_relative_l2,
        },
        "report": report.as_dict(),
        "reverse_mode_scope": "dedicated non-updating oracle child only",
    }
    destination = output / "diagnostics" / f"projected_gradient_seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"all-layer LoRA projected-gradient gate failed: {payload['report']}")
    return destination


def _run_trl_oracle_child(output: Path, *, expected_source_commit: str) -> Path:
    """Run the upstream TRL differential before any method can train."""

    if _git_output("rev-parse", "HEAD") != expected_source_commit:
        raise RuntimeError("TRL-oracle source commit differs from its parent")
    source_status = _git_output("status", "--porcelain")
    if source_status is None or source_status:
        raise RuntimeError("TRL differential requires the clean parent worktree")
    from .trl_grpo_oracle import run_trl_110_differential_certification

    report = run_trl_110_differential_certification()
    report["source_commit"] = expected_source_commit
    destination = output / "diagnostics" / "trl_1_10_differential.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.get("passed") is not True:
        raise RuntimeError(f"upstream TRL 1.10 differential failed: {report}")
    return destination


def _run_isolated_trial_child(
    config: MatchedLoRAExperimentConfig,
    output: Path,
    *,
    method: str,
    seed: int,
    expected_source_commit: str,
) -> Path:
    """Run exactly one method/seed in a fresh interpreter and CUDA context."""

    if method not in config.methods or seed not in config.seeds:
        raise ValueError("child method/seed is not declared by the locked config")
    if _git_output("rev-parse", "HEAD") != expected_source_commit:
        raise RuntimeError("child source commit differs from the parent launch commit")
    child_source_status = _git_output("status", "--porcelain")
    if child_source_status is None or child_source_status:
        raise RuntimeError("child must start from the same clean Git worktree as its parent")
    backprop_module_name = "rl_no_backward.matched_lora_backprop"
    if method == "fo_npg" and backprop_module_name in sys.modules:
        raise RuntimeError("forward-only child imported the reverse-mode implementation")
    train_examples, dev_examples, _ = _load_data(config)
    train_by_id = {example.example_id: example for example in train_examples}
    schedule_path = output / "schedules" / f"seed{seed}.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule != build_prompt_schedule(train_examples, config, seed=seed):
        raise RuntimeError("persisted prompt schedule differs from deterministic reconstruction")
    bundle, snapshot, initialization_digest = _load_bundle(config, seed=seed)
    initialization_path = Path(config.shared_initialization_path_template.format(seed=seed))
    if not initialization_path.is_file():
        raise FileNotFoundError("parent did not pre-generate the immutable LoRA initialization")
    # Loading again is intentional: it proves the child starts from the exact
    # artifact rather than merely reproducing the same seed.
    loaded_digest = load_shared_lora_initialization(
        initialization_path,
        bundle.model,
        config.lora,
    )
    if loaded_digest != initialization_digest:
        raise RuntimeError("child initialization digest differs from parent artifact")
    initial_state = lora_state_dict(bundle.model)
    if method in {"base", "fo_npg"}:
        set_adapter_grad_enabled(bundle, False)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        assert_lora_frozen(bundle.model)
    base_before = frozen_base_parameter_digest(bundle.model)
    engine = create_standard_lora_vllm_engine(
        config.model_name,
        revision=config.model_revision,
        dtype=config.dtype,
        max_model_len=config.max_prompt_tokens + config.max_new_tokens,
        max_lora_rank=config.lora.rank,
        kv_cache_memory_bytes=config.vllm_kv_cache_memory_bytes,
        enforce_eager=config.vllm_enforce_eager,
        flash_attn_version=config.vllm_flash_attn_version,
        max_num_seqs=config.responses_per_step,
        seed=seed,
    )
    generator = ReloadableLoRAGenerator(engine)
    receipts: list[dict[str, Any]] = []
    version_counter = [0]
    reload_gate_export_root = output / "vllm_lora_exports" / f"{method}_seed{seed}"
    reload_gate_prompt = _prompt_token_ids(
        bundle.tokenizer,
        _format_prompts(bundle.tokenizer, train_examples[:1]),
        config.max_prompt_tokens,
    )[0]
    reload_gate = _run_vllm_same_id_reload_gate(
        bundle,
        generator,
        reload_gate_prompt,
        reload_gate_export_root,
        version_counter=version_counter,
        reload_receipts=receipts,
    )
    _run_trial(
        bundle,
        generator,
        train_by_id,
        dev_examples,
        schedule,
        initial_state,
        initialization_digest,
        config,
        output,
        method=method,
        seed=seed,
        version_counter=version_counter,
        first_rollout_reference={},
        reload_receipts=receipts,
    )
    if method == "fo_npg" and backprop_module_name in sys.modules:
        raise RuntimeError("forward-only execution imported the reverse-mode implementation")
    base_after = frozen_base_parameter_digest(bundle.model)
    if base_after != base_before:
        raise RuntimeError("frozen base changed within isolated method process")
    trial_metadata = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "resolved_model_snapshot": snapshot,
        "shared_initialization_digest": initialization_digest,
        "frozen_base_parameter_digest_before": base_before,
        "frozen_base_parameter_digest_after": base_after,
        "vllm_lora_reload_receipts": receipts,
        "vllm_same_id_reload_gate": reload_gate,
        "process_isolation": "fresh_python_process_per_method_seed",
        "process_id": os.getpid(),
        "source_commit_verified_in_child": expected_source_commit,
        "memory_measurement_scope": {
            "allocator": "PyTorch CUDA allocator",
            "scope": "combined HF scorer/trainer plus colocated vLLM engine in child process",
            "vllm_worker_excluded": False,
        },
        "fo_reverse_mode_modules_called": False if method == "fo_npg" else None,
        "fo_backprop_module_imported": False if method == "fo_npg" else None,
    }
    destination = output / "trial_metadata" / f"{method}_seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trial_metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prepare_empty_output_directory(path: Path) -> Path:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"benchmark output is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to reuse nonempty benchmark output: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _require_output_outside_worktree(output: Path, worktree: Path) -> None:
    try:
        output.resolve().relative_to(worktree.resolve())
    except ValueError:
        return
    raise ValueError(
        "matched benchmark output must be outside the Git worktree so child clean-source checks remain valid"
    )


def _validate_isolated_trial_outputs(
    output: Path,
    config: MatchedLoRAExperimentConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    trial_metadata: dict[str, dict[str, Any]] = {}
    reload_receipts: list[dict[str, Any]] = []
    first_rollouts: dict[int, dict[str, tuple[str, str]]] = {}
    child_process_ids: set[int] = set()
    for seed in config.seeds:
        methods = config.methods if seed == config.seeds[0] else [m for m in config.methods if m != "base"]
        for method in methods:
            label = f"{method}_seed{seed}"
            metadata = _load_json_mapping(output / "trial_metadata" / f"{label}.json")
            if metadata.get("method") != method or metadata.get("seed") != seed:
                raise RuntimeError(f"{label} trial metadata identity mismatch")
            if metadata.get("process_isolation") != "fresh_python_process_per_method_seed":
                raise RuntimeError(f"{label} lacks the fresh-process isolation receipt")
            process_id = metadata.get("process_id")
            if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id < 1:
                raise RuntimeError(f"{label} lacks a valid child process ID")
            if process_id == os.getpid() or process_id in child_process_ids:
                raise RuntimeError("method/seed trials did not use distinct fresh child processes")
            child_process_ids.add(process_id)
            if (
                metadata.get("frozen_base_parameter_digest_before")
                != metadata.get("frozen_base_parameter_digest_after")
            ):
                raise RuntimeError(f"{label} changed the frozen base model")
            reload_gate = metadata.get("vllm_same_id_reload_gate")
            if not isinstance(reload_gate, Mapping) or reload_gate.get("passed") is not True:
                raise RuntimeError(f"{label} did not pass the same-ID vLLM reload gate")
            if method == "fo_npg" and metadata.get("fo_reverse_mode_modules_called") is not False:
                raise RuntimeError("forward-only child lacks its reverse-mode call-graph receipt")
            if method == "fo_npg" and metadata.get("fo_backprop_module_imported") is not False:
                raise RuntimeError("forward-only child imported the reverse-mode module")
            trial_metadata[label] = metadata
            method_receipts = metadata["vllm_lora_reload_receipts"]
            for receipt in method_receipts:
                if receipt.get("adapter_path_transient") is not True:
                    raise RuntimeError("vLLM GC receipt did not label its adapter path transient")
                if set(receipt.get("durable_hash_fields", ())) != {
                    "state_digest",
                    "adapter_model_sha256",
                }:
                    raise RuntimeError("vLLM GC receipt lacks durable adapter hashes")
            reload_receipts.extend(method_receipts)
            records = _load_jsonl(output / "raw" / f"gsm8k_{method}_seed{seed}.jsonl")
            learning_gate = _load_json_mapping(
                output / "learning_gate" / f"gsm8k_{method}_seed{seed}.json"
            )
            if learning_gate.get("passed") is not True:
                raise RuntimeError(f"{label} did not pass its predeclared learning gate")
            train_records = [record for record in records if record.get("kind") == "train_step"]
            if method == "base":
                if train_records:
                    raise RuntimeError("base trial unexpectedly contains training records")
                continue
            if len(train_records) != config.steps:
                raise RuntimeError(f"{label} has {len(train_records)} training steps")
            expected_old_calls = math.ceil(
                config.responses_per_step / config.scoring_micro_batch_size
            )
            for expected_step, record in enumerate(train_records, start=1):
                if record.get("step") != expected_step:
                    raise RuntimeError(f"{label} has a missing or reordered training step")
                if record.get("behavior_policy_digest") != record.get("lora_digest_before"):
                    raise RuntimeError(f"{label} behavior policy is not the HF old-policy center")
                if not record.get("behavior_policy_version"):
                    raise RuntimeError(f"{label} lacks a behavior-policy version")
                if record.get("old_policy_rescore_forward_calls") != expected_old_calls:
                    raise RuntimeError(f"{label} old-policy rescore call count mismatch")
                if (
                    record.get("old_policy_rescore_teacher_forced_examples")
                    != config.responses_per_step
                ):
                    raise RuntimeError(f"{label} old-policy example accounting mismatch")
                finish_counts = record.get("rollout_finish_reason_counts")
                if not isinstance(finish_counts, Mapping) or sum(finish_counts.values()) != (
                    config.responses_per_step
                ):
                    raise RuntimeError(f"{label} rollout finish-reason accounting mismatch")
                for counter in (
                    "environment_samples",
                    "generated_tokens",
                    "scored_tokens",
                    "forward_calls",
                    "backward_calls",
                    "teacher_forced_examples",
                ):
                    cumulative = record.get(f"cumulative_{counter}")
                    if cumulative != record.get(counter):
                        raise RuntimeError(f"{label} cumulative {counter} counter mismatch")
            final = train_records[-1]
            if final.get("environment_samples") != config.response_budget:
                raise RuntimeError(f"{label} did not consume the locked response budget")
            if method == "bp_grpo":
                expected_backwards = config.steps * math.ceil(
                    config.batch_size / config.backprop.prompt_groups_per_micro_batch
                )
                if final.get("backward_calls") != expected_backwards:
                    raise RuntimeError("BP gradient-accumulation call count mismatch")
            elif final.get("backward_calls") != 0:
                raise RuntimeError("forward-only trial contains reverse-mode calls")
            if final.get("teacher_forced_examples", 0) < (
                config.steps * config.responses_per_step
            ):
                raise RuntimeError(f"{label} omitted frozen-HF old-policy rescoring compute")
            first = train_records[0]
            first_rollouts.setdefault(seed, {})[method] = (
                str(first["rollout_token_digest"]),
                str(first["behavior_logprob_digest"]),
            )
    for seed, method_receipts in first_rollouts.items():
        if method_receipts.get("bp_grpo") != method_receipts.get("fo_npg"):
            raise RuntimeError(
                f"seed {seed} BP/FO first-rollout token or sampler-logprob digests differ"
            )
    return trial_metadata, reload_receipts


def run_matched_lora_benchmark(
    config: MatchedLoRAExperimentConfig,
    output_dir: str | Path,
) -> Path:
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("the matched LoRA benchmark requires an H100 CUDA runtime")
    source_commit = _git_output("rev-parse", "HEAD")
    source_status = _git_output("status", "--porcelain")
    if source_commit is None or len(source_commit) != 40:
        raise RuntimeError("matched benchmark requires a non-null 40-character source commit")
    if source_status is None or source_status:
        raise RuntimeError("matched benchmark must launch from an exactly clean Git worktree")
    worktree_value = _git_output("rev-parse", "--show-toplevel")
    if worktree_value is None:
        raise RuntimeError("could not resolve the matched benchmark Git worktree")
    _require_output_outside_worktree(Path(output_dir), Path(worktree_value))
    output = _prepare_empty_output_directory(Path(output_dir))
    train_examples, dev_examples, eval_manifest = _load_data(config)
    schedules = {
        seed: build_prompt_schedule(train_examples, config, seed=seed) for seed in config.seeds
    }
    schedule_dir = output / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    for seed, schedule in schedules.items():
        (schedule_dir / f"seed{seed}.json").write_text(
            json.dumps(schedule, indent=2) + "\n",
            encoding="utf-8",
        )

    locked_config_path = output / "matched_config.yaml"
    locked_config_path.write_text(
        yaml.safe_dump(config.as_dict(), sort_keys=False),
        encoding="utf-8",
    )
    initial_digests: dict[str, str] = {}

    subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_no_backward.matched_lora_runner",
            "--config",
            str(locked_config_path.resolve()),
            "--output",
            str(output.resolve()),
            "--trl-oracle",
            "--expected-source-commit",
            source_commit,
        ],
        check=True,
    )
    trl_differential = _load_json_mapping(
        output / "diagnostics" / "trl_1_10_differential.json"
    )
    if trl_differential.get("passed") is not True:
        raise RuntimeError("TRL differential did not certify the matched BP objective")

    for seed in config.seeds:
        bundle, _, initialization_digest = _load_bundle(config, seed=seed)
        initial_digests[str(seed)] = initialization_digest
        del bundle
        torch.cuda.empty_cache()

    oracle_seed = config.seeds[0]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_no_backward.matched_lora_runner",
            "--config",
            str(locked_config_path.resolve()),
            "--output",
            str(output.resolve()),
            "--gradient-oracle-seed",
            str(oracle_seed),
            "--expected-source-commit",
            source_commit,
        ],
        check=True,
    )
    projected_gradient_oracle = _load_json_mapping(
        output / "diagnostics" / f"projected_gradient_seed{oracle_seed}.json"
    )
    if projected_gradient_oracle.get("passed") is not True:
        raise RuntimeError("projected-gradient oracle did not certify the matched objective")

    # Every method/seed gets a clean interpreter, CUDA context, model object,
    # optimizer state, and autograd mode.  In particular, FO never follows a
    # BP trial in-process.
    for seed in config.seeds:
        methods = config.methods if seed == config.seeds[0] else [m for m in config.methods if m != "base"]
        for method in methods:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rl_no_backward.matched_lora_runner",
                    "--config",
                    str(locked_config_path.resolve()),
                    "--output",
                    str(output.resolve()),
                    "--child-method",
                    method,
                    "--seed",
                    str(seed),
                    "--expected-source-commit",
                    source_commit,
                ],
                check=True,
            )
    trial_metadata, reload_receipts = _validate_isolated_trial_outputs(output, config)
    frozen_base_digests = {
        label: {
            "before": str(value["frozen_base_parameter_digest_before"]),
            "after": str(value["frozen_base_parameter_digest_after"]),
        }
        for label, value in trial_metadata.items()
    }
    resolved_snapshots = sorted(
        {str(value["resolved_model_snapshot"]) for value in trial_metadata.values()}
    )
    if len(resolved_snapshots) != 1:
        raise RuntimeError(f"isolated trials resolved different model snapshots: {resolved_snapshots}")

    exclusions = _load_json_mapping(config.touched_test_exclusions)["test_example_ids"]
    metadata = {
        "schema_version": 1,
        "config": config.as_dict(),
        "git_commit": source_commit,
        "git_dirty": bool(source_status) if source_status is not None else None,
        "provenance": {
            "source": {"commit": source_commit},
            "model": {"id": config.model_name, "revision": config.model_revision},
            "dataset": {"id": GSM8K_DATASET_ID, "revision": config.dataset_revision},
        },
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": config.dataset_revision,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "resolved_model_snapshot": resolved_snapshots[0],
        "train_example_ids": [example.example_id for example in train_examples],
        "val_example_ids": [example.example_id for example in dev_examples],
        "test_example_ids": [],
        "excluded_test_example_ids": exclusions,
        "evaluation_manifest_path": config.evaluation_manifest,
        "evaluation_manifest_sha256": eval_manifest.manifest_sha256,
        "evaluation_manifest_dev_ids_sha256": eval_manifest.dev_ids_sha256,
        "evaluation_manifest_locked_test_ids_sha256": eval_manifest.locked_test_ids_sha256,
        "dev_source_index_receipt_path": config.dev_source_index_receipt,
        "dev_source_index_receipt_sha256": _load_json_mapping(
            config.dev_source_index_receipt
        )["receipt_sha256"],
        "development_row_loading": "Dataset.select(committed_dev_source_indices)",
        "locked_test_rows_materialized": False,
        "locked_test_accessed": False,
        "locked_test_evaluated": False,
        "prompt_schedule_digests": {
            str(seed): _json_digest(schedule) for seed, schedule in schedules.items()
        },
        "shared_initialization_digests": initial_digests,
        "frozen_base_parameter_digests": frozen_base_digests,
        "adapter_parameter_count": config.expected_lora_parameter_count,
        "lora_parameterization": config.lora.as_dict(),
        "objective_contract": {
            "ppo_denominator": "frozen_hf_old_policy_token_logprobs",
            "sampler_correction": "detached_old_hf_over_q_vllm",
            "advantages": "trl_group_centered_sample_std",
            "clip": "token_local",
            "aggregation": "equal_completion_length_normalized",
        },
        "implementation": "matched custom, TRL-equivalent",
        "upstream_trl_reference_role": "separate engineering/oracle gate; not headline result",
        "rollout_backend": "vllm_0.22_standard_peft_lora_load_inplace",
        "evaluation_backend": "same_vllm_0.22_standard_peft_lora_engine",
        "vllm_attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "hf_attention_implementation": "flash_attention_2",
        "vllm_lora_reload_receipts": reload_receipts,
        "trial_process_metadata": trial_metadata,
        "process_isolation": "fresh_python_process_per_method_seed",
        "projected_gradient_oracle": projected_gradient_oracle,
        "trl_1_10_differential": trl_differential,
        "memory_measurement_scope": {
            "allocator": "PyTorch CUDA allocator",
            "scope": "combined HF scorer/trainer plus colocated vLLM engine per child process",
            "vllm_worker_excluded": False,
        },
        "runtime": _runtime_metadata(),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    from .validate_artifacts import validate_benchmark_artifacts

    validation = validate_benchmark_artifacts(output)
    validation_receipt = validation.to_dict()
    (output / "artifact_validation.json").write_text(
        json.dumps(validation_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not validation.passed:
        raise RuntimeError(
            "matched artifact validation failed: "
            f"incomplete={list(validation.incomplete)} errors={list(validation.errors)}"
        )
    return metadata_path


def load_matched_config(path: str | Path) -> MatchedLoRAExperimentConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("matched LoRA config must contain a mapping")
    return MatchedLoRAExperimentConfig.from_mapping(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--child-method", choices=MATCHED_METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gradient-oracle-seed", type=int)
    parser.add_argument("--trl-oracle", action="store_true")
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args(argv)
    config = load_matched_config(args.config)
    if args.trl_oracle:
        if args.child_method is not None or args.gradient_oracle_seed is not None:
            parser.error("--trl-oracle is mutually exclusive with other child modes")
        if args.output is None or args.expected_source_commit is None:
            parser.error("--trl-oracle requires --output and --expected-source-commit")
        _run_trl_oracle_child(
            args.output,
            expected_source_commit=args.expected_source_commit,
        )
        return
    if args.gradient_oracle_seed is not None:
        if args.child_method is not None:
            parser.error("--gradient-oracle-seed and --child-method are mutually exclusive")
        if args.output is None or args.expected_source_commit is None:
            parser.error(
                "--gradient-oracle-seed requires --output and --expected-source-commit"
            )
        _run_projected_gradient_oracle_child(
            config,
            args.output,
            seed=args.gradient_oracle_seed,
            expected_source_commit=args.expected_source_commit,
        )
        return
    if args.child_method is not None:
        if args.output is None or args.seed is None or args.expected_source_commit is None:
            parser.error(
                "--child-method requires --output, --seed, and --expected-source-commit"
            )
        _run_isolated_trial_child(
            config,
            args.output,
            method=args.child_method,
            seed=args.seed,
            expected_source_commit=args.expected_source_commit,
        )
        return
    if args.print_plan:
        print(
            json.dumps(
                {
                    "methods": config.methods,
                    "seeds": config.seeds,
                    "steps": config.steps,
                    "prompt_groups_per_step": config.batch_size,
                    "generations_per_prompt": config.group_size,
                    "responses_per_step": config.responses_per_step,
                    "response_budget_per_trained_method": config.response_budget,
                    "lora_parameter_count": config.expected_lora_parameter_count,
                    "locked_test_evaluation": False,
                    "vllm_attention_config": {
                        "backend": "FLASH_ATTN",
                        "flash_attn_version": 2,
                    },
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --print-plan is used")
    run_matched_lora_benchmark(config, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
