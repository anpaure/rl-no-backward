"""End-to-end experiment runner for matched GRPO and forward-only trials."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .backprop import BackpropConfig, grpo_step, make_grpo_optimizer
from .common import build_rollout_batch, categorical_kl, entropy
from .config import TaskConfig
from .forward_only import (
    ActiveSubspace,
    ForwardConfig,
    evolution_strategy_step,
    forward_policy_step,
)
from .model import (
    ModelBundle,
    candidate_log_probs,
    load_model_bundle,
    parameter_vector,
    set_adapter_grad_enabled,
    set_parameter_vector,
    tokenize_prompts,
)
from .task import (
    CANDIDATE_ACTIONS,
    ChecksumExample,
    codebook_target,
    format_codebook_prompt,
)

TRAINABLE_METHODS = ("bp_grpo", "es", "fo_pg", "fo_npg", "focus_npg")
ALL_METHODS = ("base", *TRAINABLE_METHODS)


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    adapter_rank: int = 16
    adapter_layers: int = 4
    adapter_scale: float = 1.0
    action_count: int = 4
    task_variant: str = "codebook"
    prompt_mode: str = "disclosed"
    split_seed: int = 0
    methods: list[str] = field(
        default_factory=lambda: ["base", "bp_grpo", "es", "fo_pg", "fo_npg", "focus_npg"]
    )
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    steps: int = 100
    batch_size: int = 16
    group_size: int = 16
    eval_interval: int = 5
    baseline: str = "loo"
    rollout_kl_beta: float = 0.0
    calibration_examples: int = 64
    wandb_project: str = "rl-no-backward"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    backprop: dict[str, Any] = field(default_factory=dict)
    forward: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ExperimentConfig:
        known = {field_.name for field_ in cls.__dataclass_fields__.values()}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown experiment config keys: {unknown}")
        config = cls(**mapping)
        config.validate()
        return config

    def validate(self) -> None:
        unknown_methods = sorted(set(self.methods) - set(ALL_METHODS))
        if unknown_methods:
            raise ValueError(f"unknown methods: {unknown_methods}")
        if self.prompt_mode not in {"disclosed", "latent"}:
            raise ValueError("prompt_mode must be 'disclosed' or 'latent'")
        if not 2 <= self.action_count <= len(CANDIDATE_ACTIONS):
            raise ValueError("action_count must be between 2 and 10")
        if self.task_variant not in {"codebook", "checksum_mod"}:
            raise ValueError("task_variant must be codebook or checksum_mod")
        if self.task_variant == "codebook" and self.action_count != 4:
            raise ValueError("the codebook task requires action_count=4")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.steps < 1 or self.batch_size < 1 or self.group_size < 2:
            raise ValueError("steps/batch_size must be positive and group_size at least 2")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")


def prompt_for_example(
    example: ChecksumExample,
    mode: str,
    action_count: int = 10,
    task_variant: str = "checksum_mod",
) -> str:
    if task_variant == "codebook":
        return format_codebook_prompt(example.a, example.b)
    if task_variant != "checksum_mod":
        raise ValueError(f"unknown task variant {task_variant!r}")
    if mode == "disclosed":
        prompt = example.prompt
        if action_count < 10:
            prompt = prompt.replace(
                "Reply with exactly one digit from 0 to 9.",
                f"Report c mod {action_count}. "
                f"Reply with exactly one digit from 0 to {action_count - 1}.",
            )
        return prompt
    if mode == "latent":
        return (
            "A hidden deterministic verifier assigns one checksum digit to each pair.\n"
            f"a = {example.a}\n"
            f"b = {example.b}\n"
            "Predict the verifier's checksum class. "
            f"Reply with exactly one digit from 0 to {action_count - 1}."
        )
    raise ValueError(f"unknown prompt mode {mode!r}")


def _targets(
    examples: Sequence[ChecksumExample],
    device: torch.device,
    action_count: int,
    task_variant: str,
) -> Tensor:
    if task_variant == "codebook":
        values = [codebook_target(example.a, example.b) for example in examples]
    elif task_variant == "checksum_mod":
        values = [example.target % action_count for example in examples]
    else:
        raise ValueError(f"unknown task variant {task_variant!r}")
    return torch.tensor(
        values,
        device=device,
        dtype=torch.long,
    )


@torch.inference_mode()
def _log_probs_for_examples(
    bundle: ModelBundle,
    examples: Sequence[ChecksumExample],
    prompt_mode: str,
    action_count: int,
    task_variant: str,
) -> Tensor:
    encoded = tokenize_prompts(
        bundle,
        [prompt_for_example(e, prompt_mode, action_count, task_variant) for e in examples],
    )
    return candidate_log_probs(bundle, encoded)


@torch.inference_mode()
def evaluate_examples(
    bundle: ModelBundle,
    examples: Sequence[ChecksumExample],
    prompt_mode: str,
    action_count: int = 10,
    task_variant: str = "checksum_mod",
    reference_log_probs: Tensor | None = None,
) -> dict[str, float]:
    log_probs = _log_probs_for_examples(bundle, examples, prompt_mode, action_count, task_variant)
    targets = _targets(examples, bundle.device, action_count, task_variant)
    target_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
    metrics = {
        "accuracy": float(log_probs.argmax(dim=-1).eq(targets).float().mean().item()),
        "expected_reward": float(target_log_probs.exp().mean().item()),
        "target_nll": float(-target_log_probs.mean().item()),
        "entropy": float(entropy(log_probs).item()),
    }
    metrics["reference_kl"] = (
        float(categorical_kl(log_probs, reference_log_probs).item())
        if reference_log_probs is not None
        else 0.0
    )
    return metrics


def _batch_examples(
    train_examples: Sequence[ChecksumExample],
    batch_size: int,
    rng: random.Random,
) -> list[ChecksumExample]:
    return [train_examples[rng.randrange(len(train_examples))] for _ in range(batch_size)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_finite_or_none(record), sort_keys=True) + "\n")


def _start_wandb_run(
    config: ExperimentConfig,
    method: str,
    seed: int,
    output_dir: Path,
) -> Any | None:
    """Create one W&B run per method/seed, or return None when disabled."""

    if config.wandb_mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=output_dir.name,
        name=f"{method}-seed-{seed}",
        job_type="baseline" if method == "base" else "train",
        tags=[method, "forward-only" if method not in {"base", "bp_grpo"} else "backprop"],
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


def _log_wandb_record(run: Any | None, record: dict[str, Any]) -> None:
    if run is None:
        return
    namespace = "eval" if record["kind"] == "evaluation" else "train"
    payload: dict[str, Any] = {"optimizer_step": record["step"]}
    progress_names = {
        "wall_time_seconds",
        "environment_samples",
        "forward_calls",
        "backward_calls",
        "teacher_forced_examples",
        "peak_gpu_memory_bytes",
        "cumulative_environment_samples",
        "cumulative_forward_calls",
        "cumulative_backward_calls",
        "cumulative_teacher_forced_examples",
    }
    for key, value in _finite_or_none(record).items():
        if isinstance(value, (int, float)) and key not in {"step", "seed"}:
            prefix = "progress" if key in progress_names else namespace
            payload[f"{prefix}/{key}"] = value
    run.log(payload)


def _finish_wandb_run(run: Any | None, run_path: Path, checkpoint_path: Path) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=f"run-data-{run.id}",
        type="training-run",
        description="Raw JSONL metrics and the tiny trained adapter vector.",
    )
    artifact.add_file(str(run_path), name=run_path.name)
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact)
    run.finish()


def _reference_lookup(examples: Sequence[ChecksumExample], reference: Tensor) -> dict[str, Tensor]:
    return {example.example_id: reference[index] for index, example in enumerate(examples)}


def _select_reference(lookup: dict[str, Tensor], examples: Sequence[ChecksumExample]) -> Tensor:
    return torch.stack([lookup[example.example_id] for example in examples])


def _evaluation_record(
    bundle: ModelBundle,
    splits: dict[str, Sequence[ChecksumExample]],
    references: dict[str, Tensor],
    prompt_mode: str,
    action_count: int,
    task_variant: str,
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for split_name, examples in splits.items():
        metrics = evaluate_examples(
            bundle,
            examples,
            prompt_mode,
            action_count,
            task_variant,
            reference_log_probs=references[split_name],
        )
        flattened.update({f"{split_name}_{name}": value for name, value in metrics.items()})
    return flattened


def run_one(
    bundle: ModelBundle,
    config: ExperimentConfig,
    method: str,
    seed: int,
    splits: dict[str, Sequence[ChecksumExample]],
    references: dict[str, Tensor],
    reference_by_id: dict[str, Tensor],
    initial_parameters: Tensor,
    raw_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    """Run one method/seed trial and write append-only raw records."""

    set_parameter_vector(bundle, initial_parameters)
    bundle.model.eval()
    set_adapter_grad_enabled(bundle, method == "bp_grpo")
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(bundle.device)

    prompt_rng = random.Random(10_000 + seed)
    rollout_generator = torch.Generator(device=bundle.device).manual_seed(20_000 + seed)
    direction_generator = torch.Generator(device=bundle.device).manual_seed(30_000 + seed)
    run_path = raw_dir / f"{method}_seed{seed}.jsonl"
    if run_path.exists():
        run_path.unlink()
    wandb_run = _start_wandb_run(config, method, seed, raw_dir.parent)

    optimizer = None
    bp_config = BackpropConfig(**config.backprop)
    if method == "bp_grpo":
        optimizer = make_grpo_optimizer(bundle, bp_config)
    focus_state = None
    if method == "focus_npg":
        focus_template = ForwardConfig(method="focus_npg", **config.forward)
        focus_state = ActiveSubspace(focus_template.active_rank, focus_template.history_size)

    cumulative_environment_samples = 0
    cumulative_forward_calls = 0
    cumulative_backward_calls = 0
    cumulative_teacher_forced_examples = 0
    _sync(bundle.device)
    start_time = time.perf_counter()

    initial_eval = _evaluation_record(
        bundle,
        splits,
        references,
        config.prompt_mode,
        config.action_count,
        config.task_variant,
    )
    initial_record = {
        "kind": "evaluation",
        "method": method,
        "seed": seed,
        "step": 0,
        "wall_time_seconds": 0.0,
        "environment_samples": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "teacher_forced_examples": 0,
        **initial_eval,
    }
    _append_jsonl(run_path, initial_record)
    _log_wandb_record(wandb_run, initial_record)

    if method == "base":
        checkpoint_path = checkpoint_dir / f"{method}_seed{seed}.pt"
        torch.save(initial_parameters.cpu(), checkpoint_path)
        _finish_wandb_run(wandb_run, run_path, checkpoint_path)
        return run_path

    for step in range(1, config.steps + 1):
        batch = _batch_examples(splits["train"], config.batch_size, prompt_rng)
        encoded = tokenize_prompts(
            bundle,
            [
                prompt_for_example(
                    example,
                    config.prompt_mode,
                    config.action_count,
                    config.task_variant,
                )
                for example in batch
            ],
        )
        targets = _targets(
            batch,
            bundle.device,
            config.action_count,
            config.task_variant,
        )
        reference_batch = _select_reference(reference_by_id, batch)

        if method == "es":
            forward_config = ForwardConfig(method="es", **config.forward)
            result = evolution_strategy_step(
                bundle,
                encoded,
                targets,
                config.group_size,
                direction_generator,
                forward_config,
            )
        else:
            set_adapter_grad_enabled(bundle, False)
            # ``no_grad`` avoids a tape while producing ordinary tensors that
            # can safely serve as indices/constants in the backprop baseline.
            with torch.no_grad():
                old_log_probs = candidate_log_probs(bundle, encoded)
                rollout = build_rollout_batch(
                    old_log_probs,
                    targets,
                    config.group_size,
                    rollout_generator,
                    baseline=config.baseline,
                    reference_log_probs=reference_batch,
                    kl_beta=config.rollout_kl_beta,
                )
            if method == "bp_grpo":
                assert optimizer is not None
                result = grpo_step(
                    bundle,
                    encoded,
                    rollout,
                    optimizer,
                    bp_config,
                    reference_log_probs=reference_batch,
                )
            else:
                forward_config = ForwardConfig(method=method, **config.forward)
                result = forward_policy_step(
                    bundle,
                    encoded,
                    rollout,
                    direction_generator,
                    forward_config,
                    active_subspace=focus_state,
                )
            result.forward_calls += 1
            result.teacher_forced_examples += config.batch_size

        cumulative_environment_samples += result.environment_samples
        cumulative_forward_calls += result.forward_calls
        cumulative_backward_calls += result.backward_calls
        cumulative_teacher_forced_examples += result.teacher_forced_examples
        _sync(bundle.device)
        elapsed = time.perf_counter() - start_time
        step_record = {
            "kind": "train_step",
            "method": method,
            "seed": seed,
            "step": step,
            "wall_time_seconds": elapsed,
            "environment_samples": cumulative_environment_samples,
            "forward_calls": cumulative_forward_calls,
            "backward_calls": cumulative_backward_calls,
            "teacher_forced_examples": cumulative_teacher_forced_examples,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(bundle.device))
                if bundle.device.type == "cuda"
                else 0
            ),
            **asdict(result),
        }
        # Keep cumulative counters authoritative when StepResult has per-step values.
        step_record.update(
            {
                "cumulative_environment_samples": cumulative_environment_samples,
                "cumulative_forward_calls": cumulative_forward_calls,
                "cumulative_backward_calls": cumulative_backward_calls,
                "cumulative_teacher_forced_examples": cumulative_teacher_forced_examples,
            }
        )
        _append_jsonl(run_path, step_record)
        _log_wandb_record(wandb_run, step_record)

        if step % config.eval_interval == 0 or step == config.steps:
            set_adapter_grad_enabled(bundle, False)
            evaluation = _evaluation_record(
                bundle,
                splits,
                references,
                config.prompt_mode,
                config.action_count,
                config.task_variant,
            )
            _sync(bundle.device)
            evaluation_record = {
                "kind": "evaluation",
                "method": method,
                "seed": seed,
                "step": step,
                "wall_time_seconds": time.perf_counter() - start_time,
                "environment_samples": cumulative_environment_samples,
                "forward_calls": cumulative_forward_calls,
                "backward_calls": cumulative_backward_calls,
                "teacher_forced_examples": cumulative_teacher_forced_examples,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(bundle.device))
                    if bundle.device.type == "cuda"
                    else 0
                ),
                **evaluation,
            }
            _append_jsonl(run_path, evaluation_record)
            _log_wandb_record(wandb_run, evaluation_record)

    checkpoint_path = checkpoint_dir / f"{method}_seed{seed}.pt"
    torch.save(parameter_vector(bundle).cpu(), checkpoint_path)
    _finish_wandb_run(wandb_run, run_path, checkpoint_path)
    del optimizer
    return run_path


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_benchmark(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Run every configured method/seed with one shared calibrated model."""

    config.validate()
    output = Path(output_dir)
    raw_dir = output / "raw"
    checkpoint_dir = output / "checkpoints"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    task_splits = TaskConfig(split_seed=config.split_seed).build_splits()
    splits = task_splits.as_dict()
    all_examples = (*splits["train"], *splits["val"], *splits["test"])
    calibration = all_examples[: config.calibration_examples]
    calibration_prompts = [
        prompt_for_example(
            example,
            config.prompt_mode,
            config.action_count,
            config.task_variant,
        )
        for example in calibration
    ]

    bundle = load_model_bundle(
        model_name=config.model_name,
        calibration_prompts=calibration_prompts,
        candidates=CANDIDATE_ACTIONS[: config.action_count],
        adapter_rank=config.adapter_rank,
        adapter_layers=config.adapter_layers,
        adapter_scale=config.adapter_scale,
        dtype=config.dtype,
        device=config.device,
    )
    initial_parameters = parameter_vector(bundle).clone()
    references = {
        name: _log_probs_for_examples(
            bundle,
            examples,
            config.prompt_mode,
            config.action_count,
            config.task_variant,
        ).clone()
        for name, examples in splits.items()
    }
    all_reference = _log_probs_for_examples(
        bundle,
        all_examples,
        config.prompt_mode,
        config.action_count,
        config.task_variant,
    ).clone()
    reference_by_id = _reference_lookup(all_examples, all_reference)

    metadata = {
        "config": asdict(config),
        "model_name": bundle.model_name,
        "adapter_parameter_count": bundle.parameter_count,
        "adapter_names": bundle.adapter_names,
        "candidate_token_ids": bundle.candidate_token_ids.tolist(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(bundle.device) if bundle.device.type == "cuda" else None,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for method in config.methods:
        method_seeds: Iterable[int] = config.seeds if method != "base" else config.seeds[:1]
        for seed in method_seeds:
            run_one(
                bundle=bundle,
                config=config,
                method=method,
                seed=seed,
                splits=splits,
                references=references,
                reference_by_id=reference_by_id,
                initial_parameters=initial_parameters,
                raw_dir=raw_dir,
                checkpoint_dir=checkpoint_dir,
            )
    return output
