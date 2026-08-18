"""Small-model GSM8K RLVR benchmark for backprop and forward-only updates."""

from __future__ import annotations

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
    ModelBundle,
    load_model_bundle,
    parameter_vector,
    set_adapter_grad_enabled,
    set_parameter_vector,
)
from .sequence_optimizers import (
    BackpropSequenceConfig,
    ForwardSequenceConfig,
    SequenceActiveSubspace,
    forward_sequence_step,
    make_sequence_grpo_optimizer,
    sequence_grpo_step,
)
from .sequence_policy import CompletionSample, generate_sequence_rollouts
from .task import CANDIDATE_ACTIONS

GSM8K_METHODS = ("base", "bp_grpo", "fo_pg", "fo_npg", "focus_npg")


@dataclass
class GSM8KExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str | None = None
    dataset_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
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
    eval_interval: int = 10
    eval_batch_size: int = 4
    numeric_shaping_weight: float = 0.1
    calibration_examples: int = 32
    wandb_project: str = "rl-no-backward"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
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
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if not self.seeds:
            raise ValueError("at least one seed is required")


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
        "backward_calls",
        "peak_gpu_memory_bytes",
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


@torch.inference_mode()
def evaluate_gsm8k(
    bundle: ModelBundle,
    examples: Sequence[GSM8KExample],
    config: GSM8KExperimentConfig,
    metric_prefix: str = "val",
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
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = examples[start : start + config.eval_batch_size]
        batch_prompts = prompts[start : start + config.eval_batch_size]
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
        completions = bundle.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        for example, completion, token_ids in zip(
            batch_examples, completions, response_ids, strict=True
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
    cumulative_backward_calls = 0
    cumulative_teacher_forced_examples = 0
    final_samples = [dict(sample) for sample in initial_val_samples]
    initial_record = {
        "kind": "evaluation",
        "method": method,
        "seed": seed,
        "step": 0,
        "wall_time_seconds": 0.0,
        "environment_samples": 0,
        "generated_tokens": 0,
        "scored_tokens": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "teacher_forced_examples": 0,
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
            test_metrics, final_samples = evaluate_gsm8k(
                bundle, test_examples, config, metric_prefix="test"
            )
            test_record = {
                **initial_record,
                "split": "test",
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
        rollout = generate_sequence_rollouts(
            bundle,
            prompts,
            _reward_callback(batch, config.numeric_shaping_weight),
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sampling_temperature,
            max_prompt_tokens=config.max_prompt_tokens,
            seed=20_000 + seed * 1_000 + step,
            scoring_micro_batch_size=config.scoring_micro_batch_size,
        )
        rollout_exact_reward, exact_zero_advantage_fraction = _exact_rollout_metrics(
            rollout.completions, batch
        )

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

        # One teacher-forced pass (possibly micro-batched) records old-policy
        # log probabilities immediately after generation.
        old_score_calls = math.ceil(rollout.environment_samples / config.scoring_micro_batch_size)
        cumulative_environment_samples += rollout.environment_samples
        cumulative_generated_tokens += rollout.valid_response_tokens
        cumulative_scored_tokens += result.scored_tokens + rollout.valid_response_tokens
        cumulative_forward_calls += result.forward_calls + old_score_calls
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
            "environment_samples": cumulative_environment_samples,
            "generated_tokens": cumulative_generated_tokens,
            "scored_tokens": cumulative_scored_tokens,
            "forward_calls": cumulative_forward_calls,
            "backward_calls": cumulative_backward_calls,
            "teacher_forced_examples": cumulative_teacher_forced_examples,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(bundle.device))
                if bundle.device.type == "cuda"
                else 0
            ),
            "rollout_exact_reward": rollout_exact_reward,
            "exact_zero_advantage_fraction": exact_zero_advantage_fraction,
            "rollout_shaped_reward": float(rollout.rewards.mean().item()),
            "mean_response_tokens": float(rollout.response_lengths.float().mean().item()),
            "rollout_truncation_fraction": _rollout_truncation_fraction(rollout, bundle.tokenizer),
            **asdict(result),
        }
        # Result counters are per-step; the canonical top-level counters are cumulative.
        step_record.update(
            {
                "environment_samples": cumulative_environment_samples,
                "generated_tokens": cumulative_generated_tokens,
                "scored_tokens": cumulative_scored_tokens,
                "forward_calls": cumulative_forward_calls,
                "backward_calls": cumulative_backward_calls,
                "teacher_forced_examples": cumulative_teacher_forced_examples,
            }
        )
        _append_jsonl(raw_path, step_record)
        _log_wandb(wandb_run, step_record)

        if step % config.eval_interval == 0 or step == config.steps:
            set_adapter_grad_enabled(bundle, False)
            metrics, final_samples = evaluate_gsm8k(
                bundle, val_examples, config, metric_prefix="val"
            )
            _sync(bundle.device)
            evaluation_record = {
                "kind": "evaluation",
                "method": method,
                "seed": seed,
                "step": step,
                "wall_time_seconds": time.perf_counter() - start_time,
                "environment_samples": cumulative_environment_samples,
                "generated_tokens": cumulative_generated_tokens,
                "scored_tokens": cumulative_scored_tokens,
                "forward_calls": cumulative_forward_calls,
                "backward_calls": cumulative_backward_calls,
                "teacher_forced_examples": cumulative_teacher_forced_examples,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(bundle.device))
                    if bundle.device.type == "cuda"
                    else 0
                ),
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
        test_metrics, final_samples = evaluate_gsm8k(
            bundle, test_examples, config, metric_prefix="test"
        )
        _sync(bundle.device)
        test_record = {
            "kind": "evaluation",
            "split": "test",
            "method": method,
            "seed": seed,
            "step": config.steps,
            "selected_step": best_step,
            "selection_val_accuracy": best_val_accuracy,
            "wall_time_seconds": time.perf_counter() - start_time,
            "environment_samples": cumulative_environment_samples,
            "generated_tokens": cumulative_generated_tokens,
            "scored_tokens": cumulative_scored_tokens,
            "forward_calls": cumulative_forward_calls,
            "backward_calls": cumulative_backward_calls,
            "teacher_forced_examples": cumulative_teacher_forced_examples,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(bundle.device))
                if bundle.device.type == "cuda"
                else 0
            ),
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
    )
    initial_parameters = parameter_vector(bundle).clone()
    set_adapter_grad_enabled(bundle, False)
    initial_val_metrics, initial_val_samples = evaluate_gsm8k(
        bundle,
        val_examples,
        config,
        metric_prefix="val",
    )

    metadata = {
        "config": asdict(config),
        "adapter_parameter_count": bundle.parameter_count,
        "adapter_names": bundle.adapter_names,
        "train_example_ids": [example.example_id for example in train_examples],
        "val_example_ids": [example.example_id for example in val_examples],
        "test_example_ids": [example.example_id for example in test_examples],
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "resolved_model_revision": getattr(bundle.model.config, "_commit_hash", None),
        "dataset_revision": config.dataset_revision,
        "excluded_test_example_ids": sorted(excluded_test_ids),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(bundle.device),
        "hostname": platform.node(),
        "pid": os.getpid(),
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
            )
    return output
