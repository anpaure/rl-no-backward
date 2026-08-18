"""Official TRL/PEFT standard-LoRA GRPO baseline for GSM8K.

This path is deliberately isolated from the residual-core experiment.  It
uses TRL's token-level GRPO objective and PEFT's conventional LoRA layers, but
shares :mod:`rl_no_backward.standard_lora` so a backward-free trainer can load
the byte-identical adapter initialization and parameter vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8KExample,
    build_messages,
    exact_match_reward,
    filter_by_difficulty,
    load_gsm8k_split,
    select_seeded_subset,
)
from .standard_lora import (
    StandardLoRAConfig,
    attach_standard_lora,
    load_shared_lora_initialization,
    lora_parameter_layout,
    lora_state_digest,
    save_shared_lora_initialization,
)


@dataclass(slots=True)
class StandardGRPOConfig:
    """Reproducible single-GPU TRL GRPO configuration."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    dataset_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    dtype: str = "bfloat16"
    attention_implementation: str = "flash_attention_2"
    train_size: int = 512
    val_size: int = 96
    test_size: int = 256
    run_test_evaluation: bool = True
    test_exclusion_metadata: str = "configs/gsm8k_touched_test_exclusions.json"
    subset_seed: int = 314159
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    steps: int = 300
    prompts_per_update: int = 2
    group_size: int = 8
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_tokens: int = 256
    max_completion_tokens: int = 512
    sampling_temperature: float = 0.8
    learning_rate: float = 1.0e-5
    warmup_ratio: float = 0.03
    epsilon: float = 0.2
    beta: float = 0.0
    loss_type: str = "grpo"
    scale_rewards: str = "group"
    num_iterations: int = 1
    max_grad_norm: float = 1.0
    eval_interval: int = 50
    eval_batch_size: int = 4
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.25
    vllm_enable_sleep_mode: bool = False
    vllm_model_impl: str = "vllm"
    wandb_project: str = "rl-no-backward"
    wandb_mode: str = "offline"
    expected_lora_parameter_count: int = 1_089_536
    shared_initialization_path_template: str = (
        "artifacts/shared_initializations/qwen25_1p5b_qv_r8_all28_seed{seed}.pt"
    )
    lora: StandardLoRAConfig = field(default_factory=StandardLoRAConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> StandardGRPOConfig:
        payload = dict(mapping)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown standard GRPO config keys: {unknown}")
        payload["lora"] = StandardLoRAConfig.from_mapping(payload.get("lora"))
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "train_size",
            "val_size",
            "steps",
            "prompts_per_update",
            "group_size",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_prompt_tokens",
            "max_completion_tokens",
            "eval_interval",
            "eval_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.test_size < 0 or (self.run_test_evaluation and self.test_size < 1):
            raise ValueError("test_size must be positive when test evaluation is enabled")
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must contain at least one integer")
        generation_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        expected = self.prompts_per_update * self.group_size
        if generation_batch != expected:
            raise ValueError(
                "per_device_train_batch_size * gradient_accumulation_steps must equal "
                "prompts_per_update * group_size so each optimizer step uses one matched rollout"
            )
        if generation_batch % self.group_size:
            raise ValueError("generation batch must be divisible by group_size")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if self.loss_type != "grpo":
            raise ValueError("the standard baseline must use TRL loss_type='grpo'")
        if self.scale_rewards != "group":
            raise ValueError("the standard baseline must use group-standardized rewards")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.learning_rate <= 0 or self.sampling_temperature <= 0:
            raise ValueError("learning rate and sampling temperature must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")
        if self.epsilon < 0 or self.beta < 0 or self.max_grad_norm <= 0:
            raise ValueError("epsilon/beta must be non-negative and max_grad_norm positive")
        if not 0.0 < self.vllm_gpu_memory_utilization < 1.0:
            raise ValueError("vllm_gpu_memory_utilization must lie in (0, 1)")
        if self.vllm_model_impl not in {"vllm", "transformers"}:
            raise ValueError("vllm_model_impl must be vllm or transformers")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.dtype not in {"bfloat16", "float16"}:
            raise ValueError("dtype must be bfloat16 or float16")
        if (
            isinstance(self.expected_lora_parameter_count, bool)
            or not isinstance(self.expected_lora_parameter_count, int)
            or self.expected_lora_parameter_count < 1
        ):
            raise ValueError("expected_lora_parameter_count must be a positive integer")
        if (
            not isinstance(self.shared_initialization_path_template, str)
            or "{seed}" not in self.shared_initialization_path_template
        ):
            raise ValueError("shared_initialization_path_template must contain {seed}")
        for name in ("model_name", "model_revision", "dataset_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def responses_per_update(self) -> int:
        return self.prompts_per_update * self.group_size

    @property
    def response_budget(self) -> int:
        return self.steps * self.responses_per_update

    @property
    def backward_call_budget(self) -> int:
        return self.steps * self.gradient_accumulation_steps

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora"] = self.lora.as_dict()
        return payload


@dataclass(slots=True)
class _Telemetry:
    started_at: float
    generated_tokens: int = 0
    last_step: int = 0


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        for item in reversed(completion):
            text = _completion_text(item)
            if text:
                return text
    return ""


def gsm8k_exact_reward(
    completions: Sequence[Any],
    reference_answer: Sequence[str],
    **_: Any,
) -> list[float]:
    """TRL-compatible exact binary GSM8K reward."""

    if len(completions) != len(reference_answer):
        raise ValueError("completion and reference batches differ in length")
    return [
        exact_match_reward(_completion_text(completion), reference)
        for completion, reference in zip(completions, reference_answer, strict=True)
    ]


def trl_log_to_record(
    logs: Mapping[str, Any],
    *,
    step: int,
    seed: int,
    config: StandardGRPOConfig,
    telemetry: _Telemetry,
    peak_gpu_memory_bytes: int = 0,
) -> dict[str, Any] | None:
    """Convert one TRL training log to the repository's plotting schema."""

    reward = logs.get("rewards/gsm8k_exact_reward/mean", logs.get("reward"))
    if (
        not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or step < 1
        or step <= telemetry.last_step
    ):
        return None
    mean_length = logs.get("completions/mean_length", 0.0)
    if not isinstance(mean_length, (int, float)) or not math.isfinite(float(mean_length)):
        mean_length = 0.0
    completed_steps = max(0, step - telemetry.last_step)
    telemetry.generated_tokens += round(float(mean_length) * config.responses_per_update * completed_steps)
    telemetry.last_step = max(telemetry.last_step, step)
    environment_samples = step * config.responses_per_update
    record = {
        "kind": "train_step",
        "method": "standard_grpo",
        "seed": seed,
        "step": step,
        "wall_time_seconds": time.perf_counter() - telemetry.started_at,
        "environment_samples": environment_samples,
        "generated_tokens": telemetry.generated_tokens,
        "scored_tokens": telemetry.generated_tokens,
        "backward_calls": step * config.gradient_accumulation_steps,
        "forward_calls": step * config.gradient_accumulation_steps,
        "teacher_forced_examples": environment_samples,
        "rollout_exact_reward": float(reward),
        "rollout_shaped_reward": float(reward),
        "reward_mean": float(reward),
        "mean_response_tokens": float(mean_length),
        "zero_advantage_fraction": _finite_number(logs.get("frac_reward_zero_std")),
        "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
        "loss": _finite_number(logs.get("loss")),
        "grad_norm": _finite_number(logs.get("grad_norm")),
        "learning_rate": _finite_number(logs.get("learning_rate")),
        "entropy": _finite_number(logs.get("entropy")),
        "clip_ratio": _finite_number(
            logs.get("clip_ratio/region_mean", logs.get("clip_ratio"))
        ),
        "sampling_logprob_mean_abs_difference": _finite_number(
            logs.get("sampling/sampling_logp_difference/mean")
        ),
        "step_time_seconds": _finite_number(logs.get("step_time")),
        "trl_loss_type": config.loss_type,
        "importance_sampling_level": "token",
    }
    return record


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()


def _example_fingerprint(examples: Sequence[GSM8KExample]) -> str:
    return hashlib.sha256("\n".join(example.example_id for example in examples).encode()).hexdigest()


def _excluded_test_ids(metadata_path: str | None) -> set[str]:
    if metadata_path is None:
        return set()
    path = Path(metadata_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read test exclusion metadata {path}: {error}") from error
    values = payload.get("test_example_ids") if isinstance(payload, Mapping) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path} must contain a string list named test_example_ids")
    return set(values)


def select_standard_splits(
    config: StandardGRPOConfig,
) -> tuple[tuple[GSM8KExample, ...], tuple[GSM8KExample, ...], tuple[GSM8KExample, ...]]:
    """Select the same locked train/validation/test examples as the benchmark."""

    official_train = filter_by_difficulty(
        load_gsm8k_split("train", revision=config.dataset_revision)
    )
    train_and_val = select_seeded_subset(
        official_train,
        config.train_size + config.val_size,
        seed=config.subset_seed,
        namespace="train-and-validation",
    )
    train = train_and_val[: config.train_size]
    validation = train_and_val[config.train_size :]
    test: tuple[GSM8KExample, ...] = ()
    if config.run_test_evaluation:
        excluded = _excluded_test_ids(config.test_exclusion_metadata)
        official_test = tuple(
            example
            for example in load_gsm8k_split("test", revision=config.dataset_revision)
            if example.example_id not in excluded
        )
        test = select_seeded_subset(
            official_test,
            config.test_size,
            seed=config.subset_seed,
            namespace="locked-final-lora-test",
        )
    return train, validation, test


def _dataset(examples: Sequence[GSM8KExample]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("standard GRPO requires datasets") from error
    return Dataset.from_list(
        [
            {
                "prompt": list(build_messages(example.question)),
                "reference_answer": example.canonical_answer,
                "example_id": example.example_id,
            }
            for example in examples
        ]
    )


@torch.inference_mode()
def evaluate_standard_lora(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[GSM8KExample],
    config: StandardGRPOConfig,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Greedy exact-match evaluation used only for checkpoint selection/test."""

    was_training = model.training
    model.eval()
    samples: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for start in range(0, len(examples), config.eval_batch_size):
        batch = examples[start : start + config.eval_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                list(build_messages(example.question)),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_tokens,
            add_special_tokens=False,
        ).to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=config.max_completion_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        response_ids = generated[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        for example, text, token_ids in zip(batch, texts, response_ids, strict=True):
            correct = bool(exact_match_reward(text, example.canonical_answer))
            samples.append(
                {
                    "example_id": example.example_id,
                    "question": example.question,
                    "reference_answer": example.canonical_answer,
                    "completion": text,
                    "correct": correct,
                    "response_tokens": int(token_ids.ne(tokenizer.pad_token_id).sum().item()),
                }
            )
    if was_training:
        model.train()
    accuracy = sum(sample["correct"] for sample in samples) / max(1, len(samples))
    mean_tokens = sum(sample["response_tokens"] for sample in samples) / max(1, len(samples))
    return {
        "accuracy": float(accuracy),
        "exact_reward": float(accuracy),
        "mean_response_tokens": float(mean_tokens),
    }, samples


def _load_adapter_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    try:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("checkpoint selection requires PEFT") from error
    state = load_peft_weights(str(checkpoint), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"unexpected PEFT checkpoint keys: {unexpected[:8]}")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _runtime_versions() -> dict[str, Any]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "trl", "peft", "vllm", "flash-attn", "wandb"):
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


def _init_wandb(config: StandardGRPOConfig, output: Path, seed: int) -> Any | None:
    if config.wandb_mode == "disabled":
        return None
    import wandb

    os.environ["WANDB_MODE"] = config.wandb_mode
    os.environ["WANDB_DIR"] = str((output / "wandb").resolve())
    (output / "wandb").mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        name=f"standard-lora-grpo-seed-{seed}",
        config={**config.as_dict(), "seed": seed},
        dir=str(output / "wandb"),
        mode=config.wandb_mode,
    )
    run.define_metric("training_step")
    run.define_metric("train/*", step_metric="training_step")
    run.define_metric("checkpoint_step")
    run.define_metric("validation/*", step_metric="checkpoint_step")
    return run


def _load_base_with_lora(config: StandardGRPOConfig, *, seed: int) -> tuple[nn.Module, Any, str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = snapshot_download(config.model_name, revision=config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    base_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=dtype,
        attn_implementation=config.attention_implementation,
    )
    model = attach_standard_lora(base_model, config.lora, initialization_seed=seed)
    layout = lora_parameter_layout(model)
    if layout.parameter_count != config.expected_lora_parameter_count:
        raise RuntimeError(
            f"actual PEFT LoRA count {layout.parameter_count:,} does not match locked count "
            f"{config.expected_lora_parameter_count:,}"
        )
    return model, tokenizer, snapshot


def prepare_shared_initialization(config: StandardGRPOConfig, *, seed: int) -> Path:
    """Create the immutable initialization once, before either method runs."""

    config.validate()
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not declared in config.seeds")
    destination = Path(config.shared_initialization_path_template.format(seed=seed))
    model, _, _ = _load_base_with_lora(config, seed=seed)
    return save_shared_lora_initialization(destination, model, config.lora)


def build_trl_grpo_arguments(
    config: StandardGRPOConfig,
    trainer_output: str | Path,
    *,
    seed: int,
) -> Any:
    """Build the pinned TRL 1.10/Transformers 5.15 argument object."""

    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(trainer_output),
        seed=seed,
        data_seed=config.subset_seed,
        max_steps=config.steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        generation_batch_size=config.responses_per_update,
        num_generations=config.group_size,
        max_completion_length=config.max_completion_tokens,
        temperature=config.sampling_temperature,
        learning_rate=config.learning_rate,
        # Transformers 5.15 removed TrainingArguments.warmup_ratio; TRL 1.10's
        # installed GRPOConfig accepts the resolved step count.
        warmup_steps=round(config.steps * config.warmup_ratio),
        max_grad_norm=config.max_grad_norm,
        beta=config.beta,
        epsilon=config.epsilon,
        num_iterations=config.num_iterations,
        loss_type=config.loss_type,
        scale_rewards=config.scale_rewards,
        importance_sampling_level="token",
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="sequence_mask",
        use_vllm=config.use_vllm,
        vllm_mode="colocate",
        vllm_model_impl=config.vllm_model_impl,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_max_model_length=config.max_prompt_tokens + config.max_completion_tokens,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.eval_interval,
        save_only_model=True,
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=False,
    )


def run_standard_grpo(config: StandardGRPOConfig, output_dir: str | Path, *, seed: int) -> Path:
    """Run one official TRL/PEFT LoRA-GRPO seed and select on validation."""

    config.validate()
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not declared in config.seeds")
    if not torch.cuda.is_available():
        raise RuntimeError("standard GRPO requires a CUDA GPU")
    # Capture source cleanliness before any artifact directory is created.
    source_status = _git_value("status", "--porcelain")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw" / f"gsm8k_standard_grpo_seed{seed}.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to append to existing run {raw_path}")
    train_examples, val_examples, test_examples = select_standard_splits(config)

    from transformers import TrainerCallback
    from trl import GRPOTrainer

    model, tokenizer, snapshot = _load_base_with_lora(config, seed=seed)
    layout = lora_parameter_layout(model)
    initialization_path = Path(config.shared_initialization_path_template.format(seed=seed))
    if not initialization_path.is_file():
        raise FileNotFoundError(
            f"shared initialization {initialization_path} is missing; run this module with "
            "--prepare-initialization before launching either matched method"
        )
    initialization_digest = load_shared_lora_initialization(
        initialization_path,
        model,
        config.lora,
    )
    if lora_state_digest(model) != initialization_digest:
        raise RuntimeError("shared LoRA initialization digest changed after load")

    trainer_output = output / "trainer" / f"seed{seed}"
    args = build_trl_grpo_arguments(config, trainer_output, seed=seed)
    telemetry = _Telemetry(started_at=time.perf_counter())
    wandb_run = _init_wandb(config, output, seed)

    class ArtifactCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            record = trl_log_to_record(
                logs or {},
                step=int(state.global_step),
                seed=seed,
                config=config,
                telemetry=telemetry,
                peak_gpu_memory_bytes=peak,
            )
            if record is None:
                return
            _append_jsonl(raw_path, record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "training_step": record["step"],
                        **{
                            f"train/{key}": value
                            for key, value in record.items()
                            if isinstance(value, (int, float)) and value is not None
                        },
                    }
                )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_exact_reward,
        args=args,
        train_dataset=_dataset(train_examples),
        processing_class=tokenizer,
        callbacks=[ArtifactCallback()],
    )
    checkpoint_zero = trainer_output / "checkpoint-0"
    trainer.model.save_pretrained(checkpoint_zero, safe_serialization=True)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    checkpoints = [checkpoint_zero]
    checkpoints.extend(
        sorted(
            (path for path in trainer_output.glob("checkpoint-*") if path != checkpoint_zero),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
    )
    best_accuracy = -1.0
    best_step = -1
    best_checkpoint: Path | None = None
    validation_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_step = int(checkpoint.name.rsplit("-", 1)[1])
        _load_adapter_checkpoint(unwrapped, checkpoint)
        started = time.perf_counter()
        metrics, _ = evaluate_standard_lora(unwrapped, tokenizer, val_examples, config)
        record = {
            "kind": "evaluation",
            "split": "validation",
            "method": "standard_grpo",
            "seed": seed,
            "step": checkpoint_step,
            "wall_time_seconds": time.perf_counter() - telemetry.started_at,
            "evaluation_seconds": time.perf_counter() - started,
            "environment_samples": checkpoint_step * config.responses_per_update,
            "generated_tokens": telemetry.generated_tokens,
            "backward_calls": checkpoint_step * config.gradient_accumulation_steps,
            "val_accuracy": metrics["accuracy"],
            "val_exact_reward": metrics["exact_reward"],
            "val_mean_response_tokens": metrics["mean_response_tokens"],
        }
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_step = checkpoint_step
            best_checkpoint = checkpoint
        record.update(best_val_accuracy=best_accuracy, best_step=best_step)
        validation_records.append(record)
        _append_jsonl(raw_path, record)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "checkpoint_step": checkpoint_step,
                    "validation/exact_match": metrics["accuracy"],
                    "validation/evaluation_seconds": record["evaluation_seconds"],
                }
            )

    if best_checkpoint is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    _load_adapter_checkpoint(unwrapped, best_checkpoint)
    selected_adapter = output / "checkpoints" / f"gsm8k_standard_grpo_seed{seed}"
    unwrapped.save_pretrained(selected_adapter, safe_serialization=True)
    test_samples: list[dict[str, Any]] = []
    if config.run_test_evaluation:
        started = time.perf_counter()
        metrics, test_samples = evaluate_standard_lora(unwrapped, tokenizer, test_examples, config)
        test_record = {
            "kind": "evaluation",
            "split": "test",
            "method": "standard_grpo",
            "seed": seed,
            "step": config.steps,
            "selected_step": best_step,
            "selection_val_accuracy": best_accuracy,
            "wall_time_seconds": time.perf_counter() - telemetry.started_at,
            "evaluation_seconds": time.perf_counter() - started,
            "environment_samples": config.response_budget,
            "generated_tokens": telemetry.generated_tokens,
            "backward_calls": config.backward_call_budget,
            "test_accuracy": metrics["accuracy"],
            "test_exact_reward": metrics["exact_reward"],
            "test_mean_response_tokens": metrics["mean_response_tokens"],
        }
        _append_jsonl(raw_path, test_record)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "checkpoint_step": best_step,
                    "test/exact_match": metrics["accuracy"],
                }
            )
    samples_path = output / "samples" / f"gsm8k_standard_grpo_seed{seed}.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(test_samples, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "method": "standard_grpo",
        "trainer": "trl.GRPOTrainer",
        "config": config.as_dict(),
        "seed": seed,
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": config.dataset_revision,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "resolved_model_snapshot": str(Path(snapshot).resolve()),
        "train_example_ids": [example.example_id for example in train_examples],
        "val_example_ids": [example.example_id for example in val_examples],
        "test_example_ids": [example.example_id for example in test_examples],
        "dataset_split_fingerprints": {
            "train": _example_fingerprint(train_examples),
            "validation": _example_fingerprint(val_examples),
            "test": _example_fingerprint(test_examples),
        },
        "lora_parameterization": config.lora.as_dict(),
        "adapter_parameter_count": layout.parameter_count,
        "adapter_parameter_layout": layout.as_dict(),
        "shared_initialization_path": str(initialization_path),
        "shared_initialization_digest": initialization_digest,
        "response_budget": {
            "unique_prompts_per_update": config.prompts_per_update,
            "generations_per_prompt": config.group_size,
            "responses_per_update": config.responses_per_update,
            "optimizer_steps": config.steps,
            "total_responses": config.response_budget,
            "backward_calls": config.backward_call_budget,
        },
        "objective": {
            "implementation": "TRL token-level GRPO",
            "loss_type": config.loss_type,
            "importance_sampling_level": "token",
            "reward_centering": "within-group mean",
            "reward_scaling": "within-group standard deviation",
            "vllm_importance_sampling_correction": True,
        },
        "selected_step": best_step,
        "selection_val_accuracy": best_accuracy,
        "validation_records": validation_records,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(source_status) if source_status is not None else None,
        "runtime": _runtime_versions(),
    }
    metadata_path = output / f"metadata_standard_grpo_seed{seed}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "selected_step": best_step,
                "selection_val_accuracy": best_accuracy,
                "adapter_parameter_count": layout.parameter_count,
                "total_responses": config.response_budget,
            }
        )
        wandb_run.finish()
    return metadata_path


def load_standard_grpo_config(path: str | Path) -> StandardGRPOConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("standard GRPO config must be a YAML mapping")
    return StandardGRPOConfig.from_mapping(payload)


def _plan(config: StandardGRPOConfig) -> dict[str, Any]:
    # Qwen2.5-1.5B architectural constants; runtime verifies the actual PEFT
    # layout after model construction.
    from .standard_lora import qwen_projection_lora_parameter_count

    expected_count = qwen_projection_lora_parameter_count(
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=128,
        config=config.lora,
    )
    if expected_count != config.expected_lora_parameter_count:
        raise ValueError(
            f"configured LoRA scope implies {expected_count:,} parameters, but locked count is "
            f"{config.expected_lora_parameter_count:,}"
        )
    return {
        "model": f"{config.model_name}@{config.model_revision}",
        "trainer": "TRL GRPOTrainer + PEFT LoRA",
        "optimizer_steps": config.steps,
        "unique_prompts_per_update": config.prompts_per_update,
        "responses_per_update": config.responses_per_update,
        "total_responses": config.response_budget,
        "backward_calls": config.backward_call_budget,
        "expected_trainable_lora_parameters": expected_count,
        "shared_initialization_path_template": config.shared_initialization_path_template,
        "lora": config.lora.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--prepare-initialization", action="store_true")
    args = parser.parse_args(argv)
    config = load_standard_grpo_config(args.config)
    if args.print_plan:
        print(json.dumps(_plan(config), indent=2))
        return
    if args.prepare_initialization:
        if args.seed is None:
            parser.error("--seed is required with --prepare-initialization")
        print(prepare_shared_initialization(config, seed=args.seed))
        return
    if args.output is None or args.seed is None:
        parser.error("--output and --seed are required unless --print-plan is used")
    run_standard_grpo(config, args.output, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
