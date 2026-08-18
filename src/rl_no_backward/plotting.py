"""Publication-quality plots and summaries for the matched-LoRA benchmark.

The experiment runner deliberately writes append-only JSONL rather than a
plotting-specific table.  This module is the small compatibility layer between
those raw records and the final report.  The public report entry point accepts
only an artifact carrying the corrected matched-LoRA metadata contract; this is
intentional, because recursively mixing the earlier residual-core pilots into
the headline comparison would produce plausible-looking but invalid figures.

Examples
--------
Generate all report figures from a benchmark directory::

    python -m rl_no_backward.plotting runs/matched-final artifacts/final-figures

The public :func:`plot_results` function provides the same operation to Python
callers and returns the paths of every generated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, PercentFormatter

from .evaluation_manifest import load_evaluation_split_manifest
from .gsm8k import extract_model_answer
from .locked_source_sealing import (
    load_locked_source_index_receipt,
    validate_locked_source_audit,
)
from .standard_lora import lora_state_digest
from .validate_artifacts import ArtifactValidationResult, validate_benchmark_artifacts

if not os.environ.get("DISPLAY"):
    plt.switch_backend("Agg")


DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 2026
TRAIN_REWARD_ROLLING_WINDOW = 10

# Okabe--Ito plus neutral grey.  These retain contrast under the most common
# forms of colour-vision deficiency and also work in greyscale via markers.
METHOD_COLOURS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)
METHOD_MARKERS = ("o", "s", "^", "D", "P", "v", "X", "<")
METHOD_LABELS = {
    "base": "Base",
    "bp_grpo": "BP-GRPO",
    "grpo": "BP-GRPO",
    "standard_grpo": "BP-GRPO",
    "fo_pg": "Forward-only PG",
    "fo_npg": "FO-NPG",
    "fo_focus_npg": "FO-FOCUS-NPG",
    "focus_npg": "FO-FOCUS-NPG",
    "es": "Evolution strategies",
    "zero_order": "Zeroth-order",
    "forward_only": "Forward-only",
}
METHOD_PRIORITY = {
    "base": 0,
    "bp_grpo": 1,
    "grpo": 1,
    "standard_grpo": 1,
    "fo_pg": 2,
    "fo_npg": 3,
    "fo_focus_npg": 4,
    "focus_npg": 4,
    "es": 5,
}

MATCHED_METHODS = ("base", "bp_grpo", "fo_npg", "fo_focus_npg")
MATCHED_HEADLINE_METHODS = ("bp_grpo", "fo_npg")
MATCHED_IMPLEMENTATION = "matched custom, TRL-equivalent"
MATCHED_MEMORY_SCOPE = "combined in-process HF scorer/trainer + colocated vLLM engine"
MATCHED_CI_DESCRIPTION = "95% percentile bootstrap across independent seeds"
MATCHED_LOCKED_TEST_EXAMPLES = 679
MATCHED_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MATCHED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MATCHED_DATASET_ID = "openai/gsm8k"
MATCHED_DATASET_CONFIG = "main"
MATCHED_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATCHED_LORA_PARAMETER_COUNT = 1_089_536
GRADIENT_NORM_SCOPE_NOTE = (
    "Different coordinate spaces — raw magnitudes are not comparable.\n"
    "BP: full 1,089,536-D autograd gradient · FO methods: q=8 projected coordinates.\n"
    "Each seed is divided by its first nonzero norm; compare trace shape only."
)
MATCHED_LORA_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "layer_indices": list(range(28)),
    "bias": "none",
}
MATCHED_GRADIENT_ORACLE_SCHEMA = "rl-no-backward-matched-lora-gradient-oracle-v1"
LOCKED_PLAN_SCHEMA = "rl-no-backward-locked-evaluation-plan-v1"
LOCKED_RESULT_SCHEMA = "rl-no-backward-locked-evaluation-result-v1"
LOCKED_SOURCE_SCHEMA = "rl-no-backward-gsm8k-locked-source-index-v2"
LOCKED_SOURCE_AUDIT_SCHEMA = "rl-no-backward-gsm8k-locked-source-sealing-audit-v1"
_MATCHED_RUN_NAME = re.compile(
    r"^(?:gsm8k_)?(?P<method>base|bp_grpo|fo_npg|fo_focus_npg)_seed(?P<seed>\d+)\.jsonl$"
)

PLOT_STYLE: dict[str, Any] = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "axes.labelcolor": "#222222",
    "axes.labelsize": 10,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlepad": 10,
    "axes.titlesize": 12,
    "figure.constrained_layout.use": True,
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "grid.alpha": 0.20,
    "grid.color": "#6B7280",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "savefig.dpi": 240,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}


@dataclass(frozen=True)
class BootstrapEstimate:
    """Mean and deterministic percentile-bootstrap confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    count: int


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapEstimate:
    """Estimate a mean and percentile bootstrap CI, ignoring non-finite values.

    A local random generator makes the result independent of global NumPy
    state.  A singleton has no interval (NaN bounds), because one run cannot
    measure across-seed uncertainty; plotting code still renders its mean as a
    point estimate.
    """

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return BootstrapEstimate(float("nan"), float("nan"), float("nan"), 0)
    mean = float(array.mean())
    if array.size == 1:
        return BootstrapEstimate(mean, float("nan"), float("nan"), 1)
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return BootstrapEstimate(mean, float(low), float(high), int(array.size))


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (base_seed ^ int.from_bytes(digest, "little")) % (2**32)


def _normalise_key(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return re.sub(r"_+", "_", key)


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = _normalise_key(raw_key)
        joined = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, joined))
        else:
            flattened[joined] = value
    return flattened


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "accepted"}:
            return 1.0
        if lowered in {"false", "no", "rejected"}:
            return 0.0
        try:
            number = float(lowered)
        except ValueError:
            return float("nan")
        return number if math.isfinite(number) else float("nan")
    return float("nan")


def _find_value(record: Mapping[str, Any], aliases: Sequence[str]) -> Any | None:
    for alias in aliases:
        if alias in record:
            return record[alias]
    for alias in aliases:
        suffix = f"_{alias}"
        candidates = [key for key in record if key.endswith(suffix)]
        if candidates:
            # The shortest prefix is normally the direct namespace (for
            # example progress_environment_samples rather than a copied config).
            return record[min(candidates, key=len)]
    return None


def _find_number(record: Mapping[str, Any], aliases: Sequence[str]) -> float:
    value = _find_value(record, aliases)
    return _as_float(value)


def _metric_scope_rank(key: str) -> int:
    tokens = set(key.split("_"))
    if "test" in tokens:
        return 0
    if "eval" in tokens or "evaluation" in tokens or "gsm8k" in tokens:
        return 1
    if "validation" in tokens or "val" in tokens:
        return 2
    if "train" in tokens or "training" in tokens:
        return 20
    return 5


def _select_accuracy(record: Mapping[str, Any]) -> tuple[float, str | None]:
    candidates: list[tuple[tuple[int, int, int], str, float]] = []
    for key, raw_value in record.items():
        value = _as_float(raw_value)
        if not math.isfinite(value):
            continue
        exact = "exact_match" in key or "exact_accuracy" in key or "reward_exact" in key
        general = (
            "accuracy" in key
            or key.endswith(("correct_rate", "success_rate", "pass_1"))
            or "pass_at_1" in key
        )
        if not (exact or general):
            continue
        if "zero_advantage" in key or "acceptance" in key:
            continue
        semantic_rank = 0 if exact else 1
        candidates.append(((_metric_scope_rank(key), semantic_rank, len(key)), key, value))
    if not candidates:
        return float("nan"), None
    _, key, value = min(candidates, key=lambda item: item[0])
    return value, key


def _select_expected_reward(record: Mapping[str, Any]) -> tuple[float, str | None]:
    candidates: list[tuple[tuple[int, int, int], str, float]] = []
    excluded = ("advantage", "surrogate", "gradient", "derivative", "kl_beta")
    for key, raw_value in record.items():
        value = _as_float(raw_value)
        if not math.isfinite(value) or any(token in key for token in excluded):
            continue
        if "expected_reward" in key:
            semantic_rank = 0
        elif "shaped_reward" in key:
            semantic_rank = 1
        elif "exact_reward" in key:
            semantic_rank = 2
        elif key == "reward" or key.endswith("_reward"):
            semantic_rank = 3
        elif key.endswith(("reward_mean", "mean_reward")):
            semantic_rank = 4
        else:
            continue
        candidates.append(((_metric_scope_rank(key), semantic_rank, len(key)), key, value))
    if not candidates:
        return float("nan"), None
    _, key, value = min(candidates, key=lambda item: item[0])
    return value, key


def _infer_method(record: Mapping[str, Any], path: Path) -> str:
    value = _find_value(
        record,
        ("method", "algorithm", "optimizer_method", "config_method", "run_method"),
    )
    if value is not None and str(value).strip():
        return _normalise_key(value)
    stem = re.sub(r"(?:[_-]?seed[_=-]?\d+).*$", "", path.stem, flags=re.IGNORECASE)
    return _normalise_key(stem) or "unknown"


def _infer_seed(record: Mapping[str, Any], path: Path) -> int | str:
    value = _find_value(record, ("seed", "random_seed", "config_seed", "run_seed"))
    number = _as_float(value)
    if math.isfinite(number):
        return int(number)
    match = re.search(r"seed[_=-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else path.stem


def _infer_kind(record: Mapping[str, Any], accuracy_source: str | None) -> str:
    value = _find_value(record, ("kind", "record_type", "event_type", "phase"))
    if value is not None:
        kind = _normalise_key(value)
        if "eval" in kind or kind in {"test", "validation", "val"}:
            return "evaluation"
        if "train" in kind or "optim" in kind or kind == "step":
            return "train_step"
        return kind
    if accuracy_source is not None and _metric_scope_rank(accuracy_source) < 5:
        return "evaluation"
    return "record"


def _infer_evaluation_split(record: Mapping[str, Any], kind: str) -> str:
    """Canonicalise the split attached to an evaluation record.

    The GSM8K runner writes validation evaluations without a ``split`` field
    and marks the one locked official-test evaluation with ``split="test"``.
    Treating an omitted split as validation therefore preserves compatibility
    with older result files while making the held-out test row unambiguous.
    """

    if kind != "evaluation":
        return ""
    value = _find_value(record, ("evaluation_split", "eval_split", "split"))
    if value is None or not str(value).strip():
        return "validation"
    split = _normalise_key(value)
    if split in {"val", "validation", "valid", "dev", "eval", "evaluation"}:
        return "validation"
    if split in {"test", "official_test", "held_out_test", "heldout_test"}:
        return "test"
    if split in {"train", "training"}:
        return "train"
    return split


def _canonical_record(raw: Mapping[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    flat = _flatten_mapping(raw)
    accuracy, accuracy_source = _select_accuracy(flat)
    expected_reward, reward_source = _select_expected_reward(flat)
    method = _infer_method(flat, path)
    seed = _infer_seed(flat, path)
    kind = _infer_kind(flat, accuracy_source)

    canonical = dict(flat)
    canonical.update(
        {
            "method": method,
            "seed": seed,
            "kind": kind,
            "evaluation_split": _infer_evaluation_split(flat, kind),
            "step": _find_number(
                flat,
                ("optimizer_step", "global_step", "step", "iteration", "update"),
            ),
            # Prefer explicit cumulative counters.  Current train records contain
            # both per-step and cumulative versions under these names.
            "environment_samples": _find_number(
                flat,
                (
                    "cumulative_environment_samples",
                    "total_environment_samples",
                    "environment_samples",
                    "env_samples",
                    "samples_seen",
                    "episodes_seen",
                ),
            ),
            "wall_time_seconds": _find_number(
                flat,
                (
                    "wall_time_seconds",
                    "elapsed_time_seconds",
                    "elapsed_seconds",
                    "wall_clock_seconds",
                    "runtime_seconds",
                ),
            ),
            "rollout_and_old_score_seconds": _find_number(
                flat,
                (
                    "rollout_and_old_score_seconds",
                    "rollout_old_score_seconds",
                    "rollout_seconds",
                ),
            ),
            "policy_sync_seconds": _find_number(
                flat,
                (
                    "policy_sync_seconds",
                    "adapter_sync_seconds",
                    "vllm_sync_seconds",
                ),
            ),
            "optimizer_seconds": _find_number(
                flat,
                ("optimizer_seconds", "optimiser_seconds", "optimization_seconds"),
            ),
            "training_phase_seconds": _find_number(
                flat,
                ("training_phase_seconds", "total_training_phase_seconds"),
            ),
            "evaluation_seconds": _find_number(
                flat,
                ("evaluation_seconds", "eval_seconds", "validation_seconds"),
            ),
            "forward_calls": _find_number(
                flat,
                (
                    "cumulative_forward_calls",
                    "total_forward_calls",
                    "forward_calls",
                    "model_forward_calls",
                ),
            ),
            "full_prefix_calls": _find_number(
                flat,
                (
                    "cumulative_full_prefix_calls",
                    "total_full_prefix_calls",
                    "full_prefix_calls",
                ),
            ),
            "suffix_calls": _find_number(
                flat,
                ("cumulative_suffix_calls", "total_suffix_calls", "suffix_calls"),
            ),
            "backward_calls": _find_number(
                flat,
                (
                    "cumulative_backward_calls",
                    "total_backward_calls",
                    "backward_calls",
                    "backprop_calls",
                ),
            ),
            "teacher_forced_examples": _find_number(
                flat,
                (
                    "cumulative_teacher_forced_examples",
                    "total_teacher_forced_examples",
                    "teacher_forced_examples",
                    "tokens_processed",
                ),
            ),
            "peak_gpu_memory_bytes": _find_number(
                flat,
                (
                    "peak_gpu_memory_allocated_bytes",
                    "peak_gpu_memory_bytes",
                    "max_gpu_memory_bytes",
                    "peak_memory_bytes",
                    "cuda_peak_memory_bytes",
                ),
            ),
            "peak_gpu_memory_allocated_bytes": _find_number(
                flat,
                (
                    "peak_gpu_memory_allocated_bytes",
                    "peak_gpu_memory_bytes",
                    "max_gpu_memory_allocated_bytes",
                    "cuda_peak_memory_allocated_bytes",
                ),
            ),
            "peak_gpu_memory_reserved_bytes": _find_number(
                flat,
                (
                    "peak_gpu_memory_reserved_bytes",
                    "max_gpu_memory_reserved_bytes",
                    "cuda_peak_memory_reserved_bytes",
                ),
            ),
            "accuracy": accuracy,
            # Selection metrics are kept separate from the measurement at the
            # current step. A run may finish after its best checkpoint;
            # collapsing ``best_val_accuracy`` into ``accuracy`` would make
            # the learning curve lie, while ignoring it would make the
            # endpoint chart disagree with the checkpoint actually saved.
            "selection_accuracy": _find_number(
                flat,
                (
                    "selection_val_accuracy",
                    "selection_accuracy",
                    "best_val_accuracy",
                    "best_accuracy",
                ),
            ),
            "selected_step": _find_number(
                flat,
                ("selected_step", "selection_step", "best_step"),
            ),
            "expected_reward": expected_reward,
            "rollout_exact_reward": _find_number(
                flat,
                ("rollout_exact_reward", "train_exact_reward", "exact_reward"),
            ),
            "score": accuracy if math.isfinite(accuracy) else expected_reward,
            "score_source": accuracy_source or reward_source,
            "empirical_kl": _find_number(
                flat,
                ("empirical_kl", "approx_kl", "policy_kl", "observed_kl", "kl"),
            ),
            "surrogate_improvement": _find_number(
                flat,
                (
                    "surrogate_improvement",
                    "surrogate_objective_change",
                    "objective_improvement",
                ),
            ),
            "step_norm": _find_number(
                flat,
                ("step_norm", "update_norm", "parameter_step_norm"),
            ),
            "projected_gradient_norm": _find_number(
                flat,
                (
                    "projected_gradient_norm",
                    "projected_grad_norm",
                    "gradient_norm",
                ),
            ),
            "zero_advantage_fraction": _find_number(
                flat,
                (
                    "exact_zero_advantage_fraction",
                    "zero_advantage_fraction",
                    "zero_advantage_rate",
                    "zero_advantages_fraction",
                    "degenerate_group_fraction",
                ),
            ),
            "acceptance_rate": _find_number(
                flat,
                ("acceptance_rate", "update_accepted", "accepted", "line_search_accepted"),
            ),
            "source_file": str(path),
            "source_line": line_number,
        }
    )
    canonical["run_id"] = f"{path.resolve()}::{method}::{seed}"
    return canonical


def _load_jsonl_paths(paths: Sequence[Path], *, source: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    warnings.warn(
                        f"skipping invalid JSON at {path}:{line_number}: {error.msg}",
                        stacklevel=2,
                    )
                    continue
                if not isinstance(raw, Mapping):
                    warnings.warn(f"skipping non-object JSON at {path}:{line_number}", stacklevel=2)
                    continue
                record = _canonical_record(raw, path, line_number)
                record["record_index"] = len(records)
                records.append(record)
    if not records:
        raise ValueError(f"no valid JSON object records found beneath {source}")

    frame = pd.DataFrame.from_records(records)
    for column in (
        "step",
        "environment_samples",
        "wall_time_seconds",
        "rollout_and_old_score_seconds",
        "policy_sync_seconds",
        "optimizer_seconds",
        "training_phase_seconds",
        "evaluation_seconds",
        "forward_calls",
        "full_prefix_calls",
        "suffix_calls",
        "backward_calls",
        "teacher_forced_examples",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "accuracy",
        "selection_accuracy",
        "selected_step",
        "expected_reward",
        "rollout_exact_reward",
        "score",
        "empirical_kl",
        "surrogate_improvement",
        "step_norm",
        "projected_gradient_norm",
        "zero_advantage_fraction",
        "acceptance_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(
        ["method", "run_id", "step", "record_index"], na_position="last"
    ).reset_index(drop=True)


def load_results(input_dir: str | Path) -> pd.DataFrame:
    """Recursively load and canonicalise JSONL run records.

    This low-level compatibility loader intentionally accepts arbitrary run
    layouts.  Headline report generation uses :func:`load_matched_results`,
    which additionally verifies the corrected matched-LoRA metadata and run
    set before reading anything.

    Invalid individual lines are skipped with a warning so an interrupted
    append cannot destroy an otherwise complete experiment.  A missing input,
    missing JSONL files, or a directory with no valid object records is an
    error because producing an empty report would be misleading.
    """

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"results input does not exist: {root}")
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL files found beneath {root}")
    return _load_jsonl_paths(paths, source=root)


def _read_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON: {path}: {error.msg}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return dict(payload)


def _canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_live_artifact_validation(benchmark_root: Path) -> ArtifactValidationResult:
    result = validate_benchmark_artifacts(benchmark_root)
    if not result.passed:
        details = [*result.incomplete, *result.errors]
        preview = "; ".join(details[:4]) or "unknown validation failure"
        raise ValueError(
            f"matched benchmark failed live artifact validation ({result.status}): {preview}"
        )
    if result.validated_runs != result.expected_runs or result.expected_runs < 1:
        raise ValueError("matched benchmark live validation did not cover every configured run")
    return result


def _matched_input_layout(input_dir: str | Path) -> tuple[Path, Path, list[Path]]:
    source = Path(input_dir)
    if not source.exists():
        raise FileNotFoundError(f"matched results input does not exist: {source}")
    if source.is_file():
        raw_root = source.parent
        benchmark_root = raw_root.parent if raw_root.name == "raw" else raw_root
        paths = [source]
    elif source.name == "raw":
        raw_root = source
        benchmark_root = source.parent
        paths = sorted(raw_root.rglob("*.jsonl"))
    elif (source / "raw").is_dir():
        benchmark_root = source
        raw_root = source / "raw"
        paths = sorted(raw_root.rglob("*.jsonl"))
    else:
        raise ValueError(
            "matched report input must be one benchmark artifact, its raw directory, "
            "or one run file; refusing an ambiguous recursive search"
        )
    if not paths:
        raise FileNotFoundError(f"no matched JSONL run files found beneath {raw_root}")
    return benchmark_root, raw_root, paths


def _validate_matched_metadata(
    metadata: Mapping[str, Any], *, path: Path
) -> tuple[list[str], list[int]]:
    if metadata.get("implementation") != MATCHED_IMPLEMENTATION:
        raise ValueError(
            f"{path} is not a corrected matched-LoRA artifact; refusing to mix pilot results"
        )
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise TypeError(f"{path} is missing its matched experiment config")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("git_dirty") is not False
        or not isinstance(metadata.get("git_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("git_commit"))) is None
    ):
        raise ValueError(f"{path} does not certify a full clean source commit")
    if (
        metadata.get("model_name") != MATCHED_MODEL_ID
        or metadata.get("model_revision") != MATCHED_MODEL_REVISION
        or config.get("model_name") != MATCHED_MODEL_ID
        or config.get("model_revision") != MATCHED_MODEL_REVISION
        or metadata.get("dataset_id") != MATCHED_DATASET_ID
        or metadata.get("dataset_config") != MATCHED_DATASET_CONFIG
        or metadata.get("dataset_revision") != MATCHED_DATASET_REVISION
        or config.get("dataset_revision") != MATCHED_DATASET_REVISION
    ):
        raise ValueError(f"{path} does not use the pinned Qwen-1.5B/GSM8K revisions")
    resolved_snapshot = metadata.get("resolved_model_snapshot")
    if (
        not isinstance(resolved_snapshot, str)
        or Path(resolved_snapshot).name != MATCHED_MODEL_REVISION
    ):
        raise ValueError(f"{path} does not bind the resolved pinned model snapshot")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("source") != {"commit": metadata["git_commit"]}
        or provenance.get("model") != {"id": MATCHED_MODEL_ID, "revision": MATCHED_MODEL_REVISION}
        or provenance.get("dataset")
        != {"id": MATCHED_DATASET_ID, "revision": MATCHED_DATASET_REVISION}
    ):
        raise ValueError(f"{path} pinned provenance differs from its top-level metadata")
    parameterization = metadata.get("lora_parameterization")
    if (
        parameterization != MATCHED_LORA_CONFIG
        or config.get("lora") != MATCHED_LORA_CONFIG
        or metadata.get("adapter_parameter_count") != MATCHED_LORA_PARAMETER_COUNT
        or config.get("expected_lora_parameter_count") != MATCHED_LORA_PARAMETER_COUNT
    ):
        raise ValueError(
            f"{path} must use all 28 q/v LoRA blocks at r8/alpha16 "
            f"({MATCHED_LORA_PARAMETER_COUNT:,} parameters)"
        )
    objective = metadata.get("objective_contract")
    required_objective = {
        "ppo_denominator": "frozen_hf_old_policy_token_logprobs",
        "sampler_correction": "detached_old_hf_over_q_vllm",
        "advantages": "trl_group_centered_sample_std",
        "clip": "token_local",
        "aggregation": "equal_completion_length_normalized",
    }
    if not isinstance(objective, Mapping) or any(
        objective.get(key) != value for key, value in required_objective.items()
    ):
        raise ValueError(f"{path} does not declare the matched GRPO objective contract")
    memory = metadata.get("memory_measurement_scope")
    memory_scope = str(memory.get("scope", "")) if isinstance(memory, Mapping) else ""
    if (
        not isinstance(memory, Mapping)
        or "combined" not in memory_scope.lower()
        or "hf" not in memory_scope.lower()
        or "vllm" not in memory_scope.lower()
        or memory.get("vllm_worker_excluded") is not False
    ):
        raise ValueError(f"{path} does not declare combined in-process HF + vLLM memory scope")
    if (
        config.get("dtype") != "bfloat16"
        or not str(config.get("device", "")).startswith("cuda")
        or config.get("attention_implementation") != "flash_attention_2"
        or config.get("rollout_backend") != "vllm_lora"
        or config.get("vllm_flash_attn_version") != 2
        or config.get("vllm_batch_invariant") is not False
        or config.get("vllm_enable_v1_multiprocessing") is not False
        or config.get("vllm_allow_insecure_serialization") is not False
        or config.get("record_rollout_provenance") is not True
        or config.get("wandb_mode") != "offline"
        or metadata.get("rollout_backend") != "vllm_0.22_standard_peft_lora_load_inplace"
        or metadata.get("evaluation_backend") != "same_vllm_0.22_standard_peft_lora_engine"
        or metadata.get("vllm_attention_config")
        != {"backend": "FLASH_ATTN", "flash_attn_version": 2}
        or metadata.get("hf_attention_implementation") != "flash_attention_2"
    ):
        raise ValueError(f"{path} does not certify the common CUDA BF16/vLLM FA2 backend")
    if (
        metadata.get("test_example_ids") != []
        or metadata.get("locked_test_rows_materialized") is not False
        or metadata.get("locked_test_accessed") is not False
        or metadata.get("locked_test_evaluated") is not False
        or config.get("run_test_evaluation") is not False
        or config.get("test_size") != 0
        or not _is_sha256(metadata.get("evaluation_manifest_sha256"))
        or not _is_sha256(metadata.get("evaluation_manifest_locked_test_ids_sha256"))
        or not _is_sha256(metadata.get("dev_source_index_receipt_sha256"))
    ):
        raise ValueError(f"{path} does not certify the answer-sealed locked-test contract")
    oracle = metadata.get("projected_gradient_oracle")
    if (
        not isinstance(oracle, Mapping)
        or oracle.get("schema") != MATCHED_GRADIENT_ORACLE_SCHEMA
        or oracle.get("passed") is not True
        or oracle.get("adapter_parameter_count") != MATCHED_LORA_PARAMETER_COUNT
        or oracle.get("source_commit") != metadata["git_commit"]
        or oracle.get("model_snapshot") != resolved_snapshot
    ):
        raise ValueError(f"{path} has no passing matched-LoRA projected-gradient oracle")
    raw_methods = config.get("methods")
    raw_seeds = config.get("seeds")
    if not isinstance(raw_methods, Sequence) or isinstance(raw_methods, (str, bytes)):
        raise TypeError(f"{path} has no ordered method list")
    methods = [str(method) for method in raw_methods]
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(MATCHED_METHODS):
        raise ValueError(f"{path} methods are not the corrected matched-LoRA methods")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise TypeError(f"{path} has no ordered seed list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise TypeError(f"{path} seeds must be integers")
    seeds = [int(seed) for seed in raw_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} seeds must be non-empty and unique")
    return methods, seeds


def load_matched_results(input_dir: str | Path) -> pd.DataFrame:
    """Load exactly one complete corrected matched-LoRA benchmark artifact.

    The artifact-level metadata, standard-LoRA/objective contract, allowed
    methods, filename identities, and configured method/seed coverage are
    checked before plotting.  In particular, this rejects the discarded
    residual-core pilots even though some of their filenames use ``bp_grpo``
    and ``fo_npg`` labels.
    """

    benchmark_root, raw_root, paths = _matched_input_layout(input_dir)
    live_validation = _require_live_artifact_validation(benchmark_root)
    metadata_path = benchmark_root / "metadata.json"
    metadata = _read_json_mapping(metadata_path, description="matched benchmark metadata")
    methods, seeds = _validate_matched_metadata(metadata, path=metadata_path)

    identities: dict[tuple[str, int], Path] = {}
    unexpected: list[Path] = []
    for path in paths:
        match = _MATCHED_RUN_NAME.fullmatch(path.name)
        if match is None:
            unexpected.append(path)
            continue
        identity = (match.group("method"), int(match.group("seed")))
        if identity in identities:
            raise ValueError(
                f"duplicate matched run {identity} in {identities[identity]} and {path}"
            )
        identities[identity] = path
    if unexpected:
        relative = [str(path.relative_to(raw_root)) for path in unexpected]
        raise ValueError(f"unexpected JSONL files could contaminate the matched report: {relative}")

    expected = {
        (method, seeds[0]) if method == "base" else (method, seed)
        for method in methods
        for seed in (seeds[:1] if method == "base" else seeds)
    }
    actual = set(identities)
    if actual != expected:
        raise ValueError(
            "matched run set differs from metadata: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    frame = _load_jsonl_paths([identities[key] for key in sorted(identities)], source=raw_root)
    for (method, seed), path in identities.items():
        rows = frame[frame["source_file"] == str(path)]
        observed = set(zip(rows["method"], rows["seed"], strict=False))
        if observed != {(method, seed)}:
            raise ValueError(f"records in {path} do not match its method/seed filename identity")
    # The strict loader has established one and only one file per identity, so
    # a stable key lets a separately sealed test receipt attach to the same run.
    frame["run_id"] = [
        f"matched::{method}::seed={int(seed)}"
        for method, seed in zip(frame["method"], frame["seed"], strict=True)
    ]
    frame.attrs["matched_metadata"] = metadata
    frame.attrs["matched_benchmark_root"] = str(benchmark_root.resolve())
    frame.attrs["live_artifact_validation"] = live_validation.to_dict()
    return frame.sort_values(
        ["method", "run_id", "step", "record_index"], na_position="last"
    ).reset_index(drop=True)


def _contained_file(root: Path, relative: Any, *, description: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{description} must be a non-empty relative path")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} escapes its artifact root") from error
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError(
            f"{description} is missing or is not a regular contained file: {target}"
        )
    return target


def _worktree_root_from_bound_relative_path(target: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("training config input paths must be safe worktree-relative paths")
    root = target.resolve()
    for _ in Path(relative).parts:
        root = root.parent
    if (root / relative).resolve() != target.resolve():
        raise ValueError(
            "training config evaluation manifest path is not bound to the supplied file"
        )
    return root


def _strict_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{description} contains a blank row at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{description} is malformed at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, Mapping):
                raise TypeError(f"{description} line {line_number} is not an object")
            rows.append(dict(value))
    return rows


def _validate_persisted_artifact_validation(
    benchmark_root: Path,
    live: ArtifactValidationResult,
) -> tuple[Path, dict[str, Any]]:
    path = benchmark_root / "artifact_validation.json"
    payload = _read_json_mapping(path, description="persisted artifact validation")
    if (
        payload.get("passed") is not True
        or payload.get("status") != "complete"
        or payload.get("expected_runs") != live.expected_runs
        or payload.get("validated_runs") != live.validated_runs
        or payload.get("incomplete") != []
        or payload.get("errors") != []
        or payload.get("warnings") != []
    ):
        raise ValueError(
            "persisted artifact validation is not complete or differs from live validation"
        )
    return path, payload


def _validate_locked_training_inputs(
    *,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Any,
    source_receipt: Any,
    source_audit: Mapping[str, Any],
    manifest_target: Path,
    plan_inputs: Any,
) -> Path:
    expected_input_keys = {
        "development_source_receipt_relpath",
        "development_source_receipt_file_sha256",
        "development_source_receipt_sha256",
        "development_source_indices_sha256",
        "development_ids_sha256",
        "quarantine_relpath",
        "quarantine_file_sha256",
        "excluded_test_ids_sha256",
        "development_row_loading",
    }
    if not isinstance(plan_inputs, Mapping) or set(plan_inputs) != expected_input_keys:
        raise ValueError("locked plan training-evaluation inputs are missing or malformed")
    config_manifest = config.get("evaluation_manifest")
    worktree_root = _worktree_root_from_bound_relative_path(manifest_target, config_manifest)
    dev_relative = config.get("dev_source_index_receipt")
    quarantine_relative = config.get("touched_test_exclusions")
    dev_path = _contained_file(
        worktree_root,
        dev_relative,
        description="committed development source receipt",
    )
    quarantine_path = _contained_file(
        worktree_root,
        quarantine_relative,
        description="committed touched-test quarantine",
    )
    dev = _read_json_mapping(dev_path, description="development source receipt")
    unsigned_dev = dict(dev)
    dev_receipt_sha = unsigned_dev.pop("receipt_sha256", None)
    entries = dev.get("entries")
    if (
        dev.get("schema") != "rl-no-backward-gsm8k-dev-source-index-v1"
        or dev.get("dataset_id") != MATCHED_DATASET_ID
        or dev.get("dataset_config") != MATCHED_DATASET_CONFIG
        or dev.get("dataset_revision") != MATCHED_DATASET_REVISION
        or dev.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or dev.get("official_test_count") != 1_319
        or dev.get("dev_count") != 256
        or not isinstance(entries, list)
        or len(entries) != 256
        or any(
            not isinstance(entry, Mapping)
            or set(entry) != {"source_index", "example_id"}
            or isinstance(entry.get("source_index"), bool)
            or not isinstance(entry.get("source_index"), int)
            or not isinstance(entry.get("example_id"), str)
            for entry in entries
        )
        or dev_receipt_sha != _canonical_json_digest(unsigned_dev)
        or tuple(entry["example_id"] for entry in entries) != manifest.dev_example_ids
        or tuple(sorted(entry["source_index"] for entry in entries))
        != source_receipt.dev_source_indices
    ):
        raise ValueError("committed development receipt differs from the sealed split")
    quarantine = _read_json_mapping(quarantine_path, description="touched-test quarantine")
    if quarantine.get("test_example_ids") != list(manifest.excluded_test_example_ids):
        raise ValueError("committed touched-test quarantine differs from the manifest")
    access_sources = source_audit.get("access_sources")
    dev_access = (
        access_sources.get("development_source_receipt")
        if isinstance(access_sources, Mapping)
        else None
    )
    manifest_access = (
        access_sources.get("manifest") if isinstance(access_sources, Mapping) else None
    )
    if (
        not isinstance(dev_access, Mapping)
        or not isinstance(manifest_access, Mapping)
        or dev_access.get("file_sha256") != _file_sha256(dev_path)
        or dev_access.get("receipt_sha256") != dev_receipt_sha
        or manifest_access.get("file_sha256") != _file_sha256(manifest_target)
        or metadata.get("dev_source_index_receipt_sha256") != dev_receipt_sha
        or metadata.get("dev_source_index_receipt_path") != dev_relative
        or metadata.get("evaluation_manifest_path") != config_manifest
    ):
        raise ValueError("training metadata and source audit do not bind the dev/manifest files")
    expected_inputs = {
        "development_source_receipt_relpath": str(Path(str(dev_relative))),
        "development_source_receipt_file_sha256": _file_sha256(dev_path),
        "development_source_receipt_sha256": dev_receipt_sha,
        "development_source_indices_sha256": source_receipt.dev_source_indices_sha256,
        "development_ids_sha256": manifest.dev_ids_sha256,
        "quarantine_relpath": str(Path(str(quarantine_relative))),
        "quarantine_file_sha256": _file_sha256(quarantine_path),
        "excluded_test_ids_sha256": manifest.excluded_test_ids_sha256,
        "development_row_loading": "Dataset.select(committed_dev_source_indices)",
    }
    if dict(plan_inputs) != expected_inputs:
        raise ValueError("locked plan training-evaluation inputs differ from committed files")
    return worktree_root


def _validate_locked_prior_question_evidence(
    *,
    plan_evidence: Any,
    source_audit: Mapping[str, Any],
    worktree_root: Path,
) -> None:
    access_sources = source_audit.get("access_sources")
    prior_sources = (
        access_sources.get("prior_exposure_question_samples")
        if isinstance(access_sources, Mapping)
        else None
    )
    if not isinstance(prior_sources, list) or not prior_sources:
        raise ValueError("source audit has no prior-question evidence")
    expected: list[dict[str, Any]] = []
    for source in prior_sources:
        if not isinstance(source, Mapping):
            raise TypeError("source-audit prior-question entries must be objects")
        relative = source.get("committed_question_source")
        metadata_relative = source.get("committed_metadata_source")
        metadata_path = _contained_file(
            worktree_root,
            metadata_relative,
            description="committed prior metadata",
        )
        evidence_path = _contained_file(
            worktree_root,
            relative,
            description="committed prior-question evidence",
        )
        if _file_sha256(metadata_path) != source.get("metadata_file_sha256") or _file_sha256(
            evidence_path
        ) != source.get("question_sample_file_sha256"):
            raise ValueError("committed prior-question evidence differs from the source audit")
        expected.append(
            {
                "metadata_file_sha256": source.get("metadata_file_sha256"),
                "metadata_relpath": metadata_relative,
                "evidence_relpath": relative,
                "evidence_file_sha256": source.get("question_sample_file_sha256"),
                "question_projection_sha256": source.get("question_projection_sha256"),
                "row_count": source.get("row_count"),
                "access_scope": (
                    "example_id and question only; all other values lexically skipped"
                ),
            }
        )
    expected.sort(key=lambda item: str(item["metadata_file_sha256"]))
    if not isinstance(plan_evidence, list) or plan_evidence != expected:
        raise ValueError("locked plan prior-question evidence differs from committed audit inputs")


def attach_locked_test_results(
    frame: pd.DataFrame,
    results_path: str | Path,
    *,
    evaluation_manifest_path: str | Path,
    source_index_receipt_path: str | Path,
) -> pd.DataFrame:
    """Attach the one-shot locked-test receipt to its matched training runs.

    Locked-test results live in a separate answer-sealed artifact and therefore
    must never be discovered recursively.  Callers opt in with its exact
    ``results.json`` path plus the exact committed manifest and v2 source seal;
    every digest and method/seed identity must match the already validated
    training frame before any row is added.
    """

    metadata = frame.attrs.get("matched_metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("locked-test results can only attach to load_matched_results output")
    benchmark_root_value = frame.attrs.get("matched_benchmark_root")
    if not isinstance(benchmark_root_value, str):
        raise TypeError("matched frame has no benchmark-root binding")
    benchmark_root = Path(benchmark_root_value).resolve()
    live_validation = _require_live_artifact_validation(benchmark_root)
    recorded_live = frame.attrs.get("live_artifact_validation")
    if not isinstance(recorded_live, Mapping) or any(
        recorded_live.get(key) != live_validation.to_dict().get(key)
        for key in (
            "status",
            "passed",
            "expected_runs",
            "validated_runs",
            "incomplete",
            "errors",
            "warnings",
        )
    ):
        raise ValueError("live artifact validation changed after the training frame was loaded")
    metadata_path = benchmark_root / "metadata.json"
    current_metadata = _read_json_mapping(metadata_path, description="matched benchmark metadata")
    if current_metadata != dict(metadata):
        raise ValueError("matched benchmark metadata changed after the training frame was loaded")
    validation_path, _ = _validate_persisted_artifact_validation(benchmark_root, live_validation)

    path = Path(results_path).resolve()
    if path.name != "results.json" or (path.parent / "FAILED.json").exists():
        raise ValueError(
            "locked result must be results.json from a completed output with no FAILED receipt"
        )
    locked_root = path.parent
    payload = _read_json_mapping(path, description="locked-test result receipt")
    expected_result_keys = {
        "schema",
        "status",
        "plan_sha256",
        "checkpoint_set_sha256",
        "evaluation_manifest_sha256",
        "locked_test_ids_sha256",
        "source_index_receipt_sha256",
        "consumption_ledger_sha256",
        "resolved_model_snapshot",
        "locked_rows_materialized",
        "locked_rows_materialized_after_authorization",
        "locked_test_example_count",
        "total_wall_time_seconds",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "results",
        "runtime",
        "result_receipt_sha256",
    }
    if (
        set(payload) != expected_result_keys
        or payload.get("schema") != LOCKED_RESULT_SCHEMA
        or payload.get("status") != "complete"
    ):
        raise ValueError(
            f"locked-test result receipt is not complete or has the wrong schema: {path}"
        )
    receipt_digest = payload.get("result_receipt_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("result_receipt_sha256", None)
    if not isinstance(receipt_digest, str) or receipt_digest != _canonical_json_digest(
        unsigned_payload
    ):
        raise ValueError(f"locked-test result receipt digest does not match its contents: {path}")
    if payload.get("locked_test_example_count") != MATCHED_LOCKED_TEST_EXAMPLES:
        raise ValueError(
            f"locked-test result receipt must cover {MATCHED_LOCKED_TEST_EXAMPLES} examples"
        )
    if (
        not math.isfinite(_as_float(payload.get("total_wall_time_seconds")))
        or _as_float(payload.get("total_wall_time_seconds")) < 0.0
        or any(
            isinstance(payload.get(key), bool)
            or not isinstance(payload.get(key), int)
            or payload.get(key, -1) < 0
            for key in (
                "peak_gpu_memory_allocated_bytes",
                "peak_gpu_memory_reserved_bytes",
            )
        )
    ):
        raise ValueError("locked-test top-level timing/memory telemetry is malformed")
    raw_results = payload.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise TypeError(f"locked-test result receipt has no results array: {path}")

    plan_path = locked_root / "plan.json"
    complete_path = locked_root / "COMPLETE.json"
    plan = _read_json_mapping(plan_path, description="locked-test evaluation plan")
    complete = _read_json_mapping(complete_path, description="locked-test COMPLETE receipt")
    plan_digest = plan.get("plan_sha256")
    unsigned_plan = dict(plan)
    unsigned_plan.pop("plan_sha256", None)
    expected_plan_keys = {
        "schema",
        "training_source_commit",
        "evaluator_source_commit",
        "benchmark_metadata_sha256",
        "artifact_validation_sha256",
        "model",
        "dataset",
        "evaluation_manifest_sha256",
        "locked_test_ids_sha256",
        "manifest_file_sha256",
        "source_index_receipt_sha256",
        "source_index_file_sha256",
        "source_sealing_audit_file_sha256",
        "official_question_ids_sha256",
        "locked_source_indices_sha256",
        "source_sealing_audit_sha256",
        "training_evaluation_inputs",
        "prior_question_evidence",
        "row_loading",
        "selection_split",
        "selection_metric",
        "generation",
        "methods",
        "seeds",
        "base_policy_semantics",
        "checkpoints",
        "checkpoint_set_sha256",
        "plan_sha256",
    }
    if (
        set(plan) != expected_plan_keys
        or plan.get("schema") != LOCKED_PLAN_SCHEMA
        or not _is_sha256(plan_digest)
        or plan_digest != _canonical_json_digest(unsigned_plan)
        or payload.get("plan_sha256") != plan_digest
        or complete
        != {
            "schema": LOCKED_RESULT_SCHEMA,
            "status": "complete",
            "plan_sha256": plan_digest,
            "result_receipt_sha256": receipt_digest,
        }
    ):
        raise ValueError(
            "locked results, plan, and COMPLETE receipt are not cryptographically bound"
        )
    if (
        plan.get("benchmark_metadata_sha256") != _file_sha256(metadata_path)
        or plan.get("artifact_validation_sha256") != _file_sha256(validation_path)
        or plan.get("training_source_commit") != metadata.get("git_commit")
    ):
        raise ValueError("locked evaluation plan belongs to a different training benchmark")

    config = metadata["config"]
    expected_model = {
        "id": MATCHED_MODEL_ID,
        "revision": MATCHED_MODEL_REVISION,
        "resolved_snapshot": metadata["resolved_model_snapshot"],
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
    }
    expected_dataset = {
        "id": MATCHED_DATASET_ID,
        "config": MATCHED_DATASET_CONFIG,
        "revision": MATCHED_DATASET_REVISION,
        "official_test_count": 1319,
        "locked_test_count": MATCHED_LOCKED_TEST_EXAMPLES,
    }
    expected_generation = {
        "backend": "vllm_0.22_standard_peft_lora_load_inplace",
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "greedy": True,
        "max_prompt_tokens": config["max_prompt_tokens"],
        "max_new_tokens": config["max_new_tokens"],
        "eval_batch_size": config["eval_batch_size"],
    }
    if (
        plan.get("model") != expected_model
        or plan.get("dataset") != expected_dataset
        or plan.get("generation") != expected_generation
        or plan.get("selection_split") != "development"
        or plan.get("selection_metric") != "exact_match"
        or plan.get("methods") != config.get("methods")
        or plan.get("seeds") != config.get("seeds")
        or payload.get("resolved_model_snapshot") != metadata["resolved_model_snapshot"]
        or plan.get("row_loading")
        != "Dataset.select(committed_locked_source_indices), then authorized opaque-ID reorder"
        or re.fullmatch(r"[0-9a-f]{40}", str(plan.get("evaluator_source_commit"))) is None
    ):
        raise ValueError("locked plan differs from the frozen model/dataset/generation contract")

    manifest_target = Path(evaluation_manifest_path).resolve()
    source_target = Path(source_index_receipt_path).resolve()
    source_audit_target = source_target.with_suffix(".audit.json")
    manifest = load_evaluation_split_manifest(manifest_target)
    source_receipt = load_locked_source_index_receipt(
        source_target,
        manifest,
        dataset_revision=MATCHED_DATASET_REVISION,
    )
    source_audit = _read_json_mapping(
        source_audit_target, description="locked-source sealing audit"
    )
    validate_locked_source_audit(source_receipt, source_audit, manifest)
    if source_receipt.as_dict().get("schema") != LOCKED_SOURCE_SCHEMA:
        raise ValueError("headline locked-test attachment requires the v2 source-index seal")
    if (
        _file_sha256(manifest_target) != plan.get("manifest_file_sha256")
        or manifest.manifest_sha256 != plan.get("evaluation_manifest_sha256")
        or manifest.manifest_sha256 != metadata.get("evaluation_manifest_sha256")
        or manifest.locked_test_ids_sha256 != plan.get("locked_test_ids_sha256")
        or manifest.locked_test_ids_sha256
        != metadata.get("evaluation_manifest_locked_test_ids_sha256")
        or _file_sha256(source_target) != plan.get("source_index_file_sha256")
        or source_receipt.receipt_sha256 != plan.get("source_index_receipt_sha256")
        or source_receipt.receipt_sha256 != payload.get("source_index_receipt_sha256")
        or _file_sha256(source_audit_target) != plan.get("source_sealing_audit_file_sha256")
        or source_receipt.official_question_ids_sha256 != plan.get("official_question_ids_sha256")
        or source_receipt.locked_source_indices_sha256 != plan.get("locked_source_indices_sha256")
        or source_receipt.sealing_audit_sha256 != plan.get("source_sealing_audit_sha256")
        or payload.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or payload.get("locked_test_ids_sha256") != manifest.locked_test_ids_sha256
    ):
        raise ValueError(
            "locked manifest/source seal differs from the plan, results, or training metadata"
        )
    worktree_root = _validate_locked_training_inputs(
        metadata=metadata,
        config=config,
        manifest=manifest,
        source_receipt=source_receipt,
        source_audit=source_audit,
        manifest_target=manifest_target,
        plan_inputs=plan.get("training_evaluation_inputs"),
    )
    _validate_locked_prior_question_evidence(
        plan_evidence=plan.get("prior_question_evidence"),
        source_audit=source_audit,
        worktree_root=worktree_root,
    )

    ledger_path = benchmark_root / "locked_test_consumed.json"
    ledger = _read_json_mapping(ledger_path, description="locked-test consumption ledger")
    if (
        set(ledger)
        != {
            "schema",
            "status",
            "plan_sha256",
            "checkpoint_set_sha256",
            "evaluation_manifest_sha256",
            "locked_test_ids_sha256",
            "output",
            "started_unix_seconds",
        }
        or _file_sha256(ledger_path) != payload.get("consumption_ledger_sha256")
        or ledger.get("schema") != "rl-no-backward-locked-consumption-v1"
        or ledger.get("status") != "started"
        or ledger.get("plan_sha256") != plan_digest
        or ledger.get("checkpoint_set_sha256") != plan.get("checkpoint_set_sha256")
        or ledger.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or ledger.get("locked_test_ids_sha256") != manifest.locked_test_ids_sha256
        or payload.get("locked_rows_materialized") is not True
        or payload.get("locked_rows_materialized_after_authorization") is not True
        or not math.isfinite(_as_float(ledger.get("started_unix_seconds")))
    ):
        raise ValueError("locked-test consumption ledger is missing or differs from the result")

    run_ids = {
        (str(method), int(seed)): str(run_id)
        for method, seed, run_id in frame[["method", "seed", "run_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if config.get("methods") != ["base", "bp_grpo", "fo_npg"]:
        raise ValueError("locked headline results support only Base, BP-GRPO, and FO-NPG")
    seeds = list(config["seeds"])
    expected_identity_order = [
        (method, seed)
        for seed_index, seed in enumerate(seeds)
        for method in (
            config["methods"]
            if seed_index == 0
            else [method for method in config["methods"] if method != "base"]
        )
    ]
    if plan.get("base_policy_semantics") != {
        "evaluated_once": True,
        "seed": seeds[0],
        "role": "shared non-updating reference for every trained seed",
    }:
        raise ValueError("locked plan has incorrect shared-base semantics")
    plan_checkpoints = plan.get("checkpoints")
    if not isinstance(plan_checkpoints, list) or not plan_checkpoints:
        raise TypeError("locked plan has no checkpoint array")
    observed_plan_order = [
        (checkpoint.get("method"), checkpoint.get("seed"))
        for checkpoint in plan_checkpoints
        if isinstance(checkpoint, Mapping)
    ]
    observed_result_order = [
        (result.get("method"), result.get("seed"))
        for result in raw_results
        if isinstance(result, Mapping)
    ]
    if (
        observed_plan_order != expected_identity_order
        or observed_result_order != expected_identity_order
    ):
        raise ValueError(
            "locked plan/results policy identities are missing, duplicated, or reordered"
        )
    if plan.get("checkpoint_set_sha256") != _canonical_json_digest(plan_checkpoints) or payload.get(
        "checkpoint_set_sha256"
    ) != plan.get("checkpoint_set_sha256"):
        raise ValueError("locked checkpoint-set digest is invalid")
    checkpoint_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    expected_checkpoint_keys = {
        "method",
        "seed",
        "selected_step",
        "selection_val_accuracy",
        "selected_lora_state_digest",
        "lora_parameter_count",
        "lora_layout_sha256",
        "selection_relpath",
        "selection_file_sha256",
        "checkpoint_relpath",
        "checkpoint_file_sha256",
        "raw_relpath",
        "raw_file_sha256",
        "raw_development_evaluation_count",
        "recomputed_selection_rule",
        "learning_gate_relpath",
        "learning_gate_file_sha256",
    }
    expected_selection_keys = {
        "schema",
        "method",
        "seed",
        "selected_step",
        "selection_split",
        "selection_metric",
        "tie_breaker",
        "selection_val_accuracy",
        "selected_lora_state_digest",
        "checkpoint_path",
    }
    for checkpoint in plan_checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("locked plan checkpoint entries must be objects")
        method = checkpoint.get("method")
        seed = checkpoint.get("seed")
        identity = (
            (str(method), seed) if isinstance(seed, int) and not isinstance(seed, bool) else None
        )
        if identity is None or identity in checkpoint_by_identity:
            raise ValueError("locked plan has invalid or duplicate checkpoint identities")
        selection_relpath = f"selection/gsm8k_{identity[0]}_seed{identity[1]}.json"
        checkpoint_relpath = f"checkpoints/gsm8k_{identity[0]}_seed{identity[1]}.pt"
        raw_relpath = f"raw/gsm8k_{identity[0]}_seed{identity[1]}.jsonl"
        learning_gate_relpath = f"learning_gate/gsm8k_{identity[0]}_seed{identity[1]}.json"
        if (
            set(checkpoint) != expected_checkpoint_keys
            or checkpoint.get("selection_relpath") != selection_relpath
            or checkpoint.get("checkpoint_relpath") != checkpoint_relpath
            or checkpoint.get("raw_relpath") != raw_relpath
            or checkpoint.get("learning_gate_relpath") != learning_gate_relpath
            or checkpoint.get("recomputed_selection_rule")
            != "maximum val_accuracy, latest step on exact tie"
            or checkpoint.get("lora_parameter_count") != MATCHED_LORA_PARAMETER_COUNT
            or not _is_sha256(checkpoint.get("lora_layout_sha256"))
            or not _is_sha256(checkpoint.get("selected_lora_state_digest"))
        ):
            raise ValueError(f"locked checkpoint contract is malformed: {identity}")
        selection_file = _contained_file(
            benchmark_root, selection_relpath, description=f"{identity} selection receipt"
        )
        checkpoint_file = _contained_file(
            benchmark_root, checkpoint_relpath, description=f"{identity} checkpoint"
        )
        raw_file = _contained_file(
            benchmark_root, raw_relpath, description=f"{identity} raw training log"
        )
        learning_gate_file = _contained_file(
            benchmark_root,
            learning_gate_relpath,
            description=f"{identity} learning-gate receipt",
        )
        selection = _read_json_mapping(selection_file, description=f"{identity} selection receipt")
        learning_gate = _read_json_mapping(
            learning_gate_file, description=f"{identity} learning-gate receipt"
        )
        raw_records = _strict_jsonl(raw_file, description=f"{identity} raw training log")
        raw_development = [
            record
            for record in raw_records
            if record.get("kind") == "evaluation" and record.get("split") == "validation"
        ]
        raw_steps: set[int] = set()
        raw_scored: list[tuple[float, int]] = []
        for record in raw_development:
            raw_step = record.get("step")
            raw_accuracy = record.get("val_accuracy")
            if (
                record.get("method") != identity[0]
                or record.get("seed") != identity[1]
                or isinstance(raw_step, bool)
                or not isinstance(raw_step, int)
                or raw_step < 0
                or raw_step in raw_steps
                or isinstance(raw_accuracy, bool)
                or not isinstance(raw_accuracy, (int, float))
                or not math.isfinite(float(raw_accuracy))
                or not 0.0 <= float(raw_accuracy) <= 1.0
            ):
                raise ValueError(f"raw development evaluations are malformed: {identity}")
            raw_steps.add(raw_step)
            raw_scored.append((float(raw_accuracy), raw_step))
        if (
            not raw_scored
            or max(raw_scored)
            != (
                float(checkpoint.get("selection_val_accuracy", float("nan"))),
                checkpoint.get("selected_step"),
            )
            or (identity[0] == "base" and raw_steps != {0})
        ):
            raise ValueError(f"raw development records do not reproduce selection: {identity}")
        structural_checks = learning_gate.get("hard_structural_checks")
        if not isinstance(structural_checks, Mapping) or any(
            value is not None and not isinstance(value, bool)
            for value in structural_checks.values()
        ):
            raise ValueError(f"learning-gate structural checks are malformed: {identity}")
        failed_structural = sorted(
            name
            for name, value in structural_checks.items()
            if value is not None and value is not True
        )
        gate_identity_valid = (
            learning_gate.get("role") == "non-updating baseline"
            and "method" not in learning_gate
            and "seed" not in learning_gate
            if identity[0] == "base"
            else learning_gate.get("method") == identity[0]
            and learning_gate.get("seed") == identity[1]
        )
        best_dev_accuracy = learning_gate.get("best_dev_accuracy")
        gate_best_valid = identity[0] == "base" or (
            not isinstance(best_dev_accuracy, bool)
            and isinstance(best_dev_accuracy, (int, float))
            and math.isfinite(float(best_dev_accuracy))
            and float(best_dev_accuracy)
            == float(checkpoint.get("selection_val_accuracy", float("nan")))
        )
        if (
            set(selection) != expected_selection_keys
            or selection.get("schema") != "rl-no-backward-validation-selection-v1"
            or selection.get("selection_split") != "development"
            or selection.get("selection_metric") != "exact_match"
            or selection.get("tie_breaker") != "latest_checkpoint"
        ):
            raise ValueError(f"training selection receipt is malformed: {identity}")
        if (
            _file_sha256(selection_file) != checkpoint.get("selection_file_sha256")
            or _file_sha256(checkpoint_file) != checkpoint.get("checkpoint_file_sha256")
            or _file_sha256(raw_file) != checkpoint.get("raw_file_sha256")
            or len(raw_development) != checkpoint.get("raw_development_evaluation_count")
            or _file_sha256(learning_gate_file) != checkpoint.get("learning_gate_file_sha256")
            or learning_gate.get("schema") != "rl-no-backward-matched-learning-gate-v2"
            or not gate_identity_valid
            or learning_gate.get("passed") is not True
            or learning_gate.get("hard_structural_checks_passed") is not (not failed_structural)
            or learning_gate.get("failed_hard_structural_checks") != failed_structural
            or failed_structural
            or not gate_best_valid
            or selection.get("method") != identity[0]
            or selection.get("seed") != identity[1]
            or selection.get("selected_step") != checkpoint.get("selected_step")
            or float(selection.get("selection_val_accuracy", float("nan")))
            != float(checkpoint.get("selection_val_accuracy", float("nan")))
            or selection.get("selected_lora_state_digest")
            != checkpoint.get("selected_lora_state_digest")
            or Path(str(selection.get("checkpoint_path"))).name != checkpoint_file.name
        ):
            raise ValueError(f"locked checkpoint differs from training selection/files: {identity}")
        import torch

        checkpoint_payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_payload, Mapping) or set(checkpoint_payload) != {
            "method",
            "seed",
            "selected_step",
            "selection_val_accuracy",
            "lora_state",
        }:
            raise ValueError(f"training checkpoint payload is malformed: {identity}")
        state = checkpoint_payload.get("lora_state")
        if (
            checkpoint_payload.get("method") != identity[0]
            or checkpoint_payload.get("seed") != identity[1]
            or checkpoint_payload.get("selected_step") != checkpoint.get("selected_step")
            or float(checkpoint_payload.get("selection_val_accuracy", float("nan")))
            != float(checkpoint.get("selection_val_accuracy", float("nan")))
            or not isinstance(state, Mapping)
            or not state
            or lora_state_digest(state) != checkpoint.get("selected_lora_state_digest")
        ):
            raise ValueError(f"training checkpoint state differs from its selection: {identity}")
        parameter_count = sum(tensor.numel() for tensor in state.values())
        layout = [
            {
                "name": name,
                "shape": list(state[name].shape),
                "dtype": str(state[name].dtype).removeprefix("torch."),
            }
            for name in sorted(state)
        ]
        if parameter_count != MATCHED_LORA_PARAMETER_COUNT or _canonical_json_digest(
            layout
        ) != checkpoint.get("lora_layout_sha256"):
            raise ValueError(f"training checkpoint LoRA layout/count is invalid: {identity}")
        validation_rows = frame[
            (frame["method"] == identity[0])
            & (pd.to_numeric(frame["seed"], errors="coerce") == identity[1])
            & (frame["kind"] == "evaluation")
            & (frame["evaluation_split"] == "validation")
        ]
        valid_accuracy = pd.to_numeric(validation_rows["accuracy"], errors="coerce")
        if valid_accuracy.empty or not np.isfinite(valid_accuracy).any():
            raise ValueError(f"training validation curve is missing for selection: {identity}")
        best_accuracy = float(valid_accuracy[np.isfinite(valid_accuracy)].max())
        best_rows = validation_rows[np.isclose(valid_accuracy, best_accuracy, rtol=0.0, atol=0.0)]
        latest_best_step = int(pd.to_numeric(best_rows["step"], errors="raise").max())
        if (
            float(checkpoint["selection_val_accuracy"]) != best_accuracy
            or int(checkpoint["selected_step"]) != latest_best_step
        ):
            raise ValueError(f"training selection is not max-validation/latest-tie: {identity}")
        checkpoint_by_identity[identity] = dict(checkpoint)
    if set(checkpoint_by_identity) != set(run_ids):
        raise ValueError("locked plan checkpoint identities differ from matched training runs")
    if len({value["lora_layout_sha256"] for value in checkpoint_by_identity.values()}) != 1:
        raise ValueError("locked plan checkpoints do not share one exact LoRA layout")

    attached: dict[tuple[str, int], dict[str, Any]] = {}
    sample_paths: set[Path] = set()
    canonical_sample_mapping: tuple[tuple[str, int], ...] | None = None
    expected_result_entry_keys = {
        "method",
        "seed",
        "selected_step",
        "selection_val_accuracy",
        "selected_lora_state_digest",
        "checkpoint_file_sha256",
        "policy_sync_seconds",
        "evaluation_seconds",
        "accuracy",
        "correct_count",
        "example_count",
        "mean_response_tokens",
        "finish_reason_counts",
        "truncation_fraction",
        "eos_terminated_fraction",
        "samples_relpath",
        "samples_sha256",
        "prediction_sha256",
        "vllm_lora_reload_receipt",
        "vllm_policy_probe",
    }
    expected_sample_keys = {
        "example_id",
        "source_index",
        "completion",
        "predicted_answer",
        "correct",
        "response_tokens",
        "finish_reason",
    }
    for version, result in enumerate(raw_results, start=1):
        if not isinstance(result, Mapping):
            raise TypeError(f"locked-test result entries must be objects: {path}")
        method = str(result.get("method"))
        seed_value = result.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError(f"locked-test result has an invalid seed: {result!r}")
        identity = (method, int(seed_value))
        if identity in attached:
            raise ValueError(f"locked-test result duplicates method/seed {identity}")
        accuracy = _as_float(result.get("accuracy"))
        selection_accuracy = _as_float(result.get("selection_val_accuracy"))
        selected_step = _as_float(result.get("selected_step"))
        raw_selected_step = result.get("selected_step")
        example_count = result.get("example_count")
        correct_count = result.get("correct_count")
        checkpoint = checkpoint_by_identity.get(identity)
        reload_receipt = result.get("vllm_lora_reload_receipt")
        policy_probe = result.get("vllm_policy_probe")
        expected_policy_version = (
            f"locked/{identity[0]}/seed={identity[1]}/selected-step={result.get('selected_step')}"
        )
        finish_counts = result.get("finish_reason_counts")
        if (
            set(result) != expected_result_entry_keys
            or identity not in run_ids
            or checkpoint is None
            or not math.isfinite(accuracy)
            or not 0.0 <= accuracy <= 1.0
            or not math.isfinite(selection_accuracy)
            or not 0.0 <= selection_accuracy <= 1.0
            or not math.isfinite(selected_step)
            or isinstance(raw_selected_step, bool)
            or not isinstance(raw_selected_step, int)
            or raw_selected_step < 0
            or isinstance(example_count, bool)
            or not isinstance(example_count, int)
            or example_count != MATCHED_LOCKED_TEST_EXAMPLES
            or isinstance(correct_count, bool)
            or not isinstance(correct_count, int)
            or not 0 <= correct_count <= example_count
            or not math.isclose(accuracy, correct_count / example_count, abs_tol=1e-12)
            or result.get("selected_step") != checkpoint.get("selected_step")
            or float(result.get("selection_val_accuracy"))
            != float(checkpoint.get("selection_val_accuracy"))
            or result.get("selected_lora_state_digest")
            != checkpoint.get("selected_lora_state_digest")
            or result.get("checkpoint_file_sha256") != checkpoint.get("checkpoint_file_sha256")
            or not isinstance(reload_receipt, Mapping)
            or set(reload_receipt)
            != {
                "version",
                "policy_version",
                "state_digest",
                "adapter_model_sha256",
                "parameter_count",
                "adapter_path",
                "active_lora_ids",
                "prefix_cache_reset",
                "load_inplace",
                "adapter_path_transient",
                "durable_hash_fields",
            }
            or reload_receipt.get("version") != version
            or reload_receipt.get("policy_version") != expected_policy_version
            or reload_receipt.get("state_digest") != checkpoint.get("selected_lora_state_digest")
            or reload_receipt.get("parameter_count") != MATCHED_LORA_PARAMETER_COUNT
            or reload_receipt.get("active_lora_ids") != [1]
            or reload_receipt.get("prefix_cache_reset") is not True
            or reload_receipt.get("load_inplace") is not True
            or reload_receipt.get("adapter_path_transient") is not True
            or reload_receipt.get("durable_hash_fields") != ["state_digest", "adapter_model_sha256"]
            or not _is_sha256(reload_receipt.get("adapter_model_sha256"))
            or not isinstance(policy_probe, Mapping)
            or set(policy_probe)
            != {"policy_version", "state_digest", "token_id", "selected_token_logprob"}
            or policy_probe.get("policy_version") != expected_policy_version
            or policy_probe.get("state_digest") != checkpoint.get("selected_lora_state_digest")
            or isinstance(policy_probe.get("token_id"), bool)
            or not isinstance(policy_probe.get("token_id"), int)
            or policy_probe.get("token_id", -1) < 0
            or not math.isfinite(_as_float(policy_probe.get("selected_token_logprob")))
            or not math.isfinite(_as_float(result.get("policy_sync_seconds")))
            or _as_float(result.get("policy_sync_seconds")) < 0.0
            or not math.isfinite(_as_float(result.get("evaluation_seconds")))
            or _as_float(result.get("evaluation_seconds")) < 0.0
            or not math.isfinite(_as_float(result.get("mean_response_tokens")))
            or _as_float(result.get("mean_response_tokens")) < 0.0
            or not isinstance(finish_counts, Mapping)
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in finish_counts.items()
            )
            or sum(finish_counts.values()) != MATCHED_LOCKED_TEST_EXAMPLES
        ):
            raise ValueError(
                f"locked-test result violates its matched evaluation contract: {identity}"
            )
        canonical_samples_relpath = f"samples/gsm8k_{identity[0]}_seed{identity[1]}.jsonl"
        if result.get("samples_relpath") != canonical_samples_relpath:
            raise ValueError(f"locked sample path is not canonical: {identity}")
        sample_path = _contained_file(
            locked_root,
            result.get("samples_relpath"),
            description=f"{identity} locked samples",
        )
        if sample_path in sample_paths or _file_sha256(sample_path) != result.get("samples_sha256"):
            raise ValueError(f"locked sample path/digest is duplicate or invalid: {identity}")
        sample_paths.add(sample_path)
        samples = _strict_jsonl(sample_path, description=f"{identity} locked samples")
        if len(samples) != MATCHED_LOCKED_TEST_EXAMPLES:
            raise ValueError(f"locked sample file has {len(samples)}/679 rows: {identity}")
        expected_ids = manifest.locked_test_example_ids
        if tuple(sample.get("example_id") for sample in samples) != expected_ids or any(
            set(sample) != expected_sample_keys
            or isinstance(sample.get("source_index"), bool)
            or not isinstance(sample.get("source_index"), int)
            or not isinstance(sample.get("correct"), bool)
            or not isinstance(sample.get("completion"), str)
            or sample.get("predicted_answer") != extract_model_answer(sample.get("completion", ""))
            or isinstance(sample.get("response_tokens"), bool)
            or not isinstance(sample.get("response_tokens"), int)
            or sample.get("response_tokens", -1) < 0
            or (
                sample.get("finish_reason") is not None
                and not isinstance(sample.get("finish_reason"), str)
            )
            for sample in samples
        ):
            raise ValueError(
                f"locked samples differ from the sealed ID/source-index order: {identity}"
            )
        observed_mapping = tuple(
            (str(sample["example_id"]), int(sample["source_index"])) for sample in samples
        )
        if (
            {source_index for _, source_index in observed_mapping}
            != set(source_receipt.locked_source_indices)
            or len({source_index for _, source_index in observed_mapping})
            != MATCHED_LOCKED_TEST_EXAMPLES
            or (
                canonical_sample_mapping is not None
                and observed_mapping != canonical_sample_mapping
            )
        ):
            raise ValueError(f"locked sample source-index mapping is inconsistent: {identity}")
        canonical_sample_mapping = observed_mapping
        observed_correct = sum(int(sample["correct"]) for sample in samples)
        observed_finish_counts: dict[str, int] = {}
        for sample in samples:
            finish_key = "none" if sample["finish_reason"] is None else str(sample["finish_reason"])
            observed_finish_counts[finish_key] = observed_finish_counts.get(finish_key, 0) + 1
        observed_mean_tokens = sum(int(sample["response_tokens"]) for sample in samples) / len(
            samples
        )
        predictions = [
            {
                "example_id": sample["example_id"],
                "predicted_answer": sample.get("predicted_answer"),
                "correct": sample["correct"],
            }
            for sample in samples
        ]
        if (
            observed_correct != correct_count
            or _canonical_json_digest(predictions) != result.get("prediction_sha256")
            or dict(sorted(observed_finish_counts.items())) != dict(finish_counts)
            or not math.isclose(
                observed_mean_tokens, float(result["mean_response_tokens"]), abs_tol=1e-12
            )
            or not math.isclose(
                observed_finish_counts.get("length", 0) / MATCHED_LOCKED_TEST_EXAMPLES,
                _as_float(result.get("truncation_fraction")),
                abs_tol=1e-12,
            )
            or not math.isclose(
                (observed_finish_counts.get("stop", 0) + observed_finish_counts.get("eos", 0))
                / MATCHED_LOCKED_TEST_EXAMPLES,
                _as_float(result.get("eos_terminated_fraction")),
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"locked samples do not reproduce result metrics/digest: {identity}")
        attached[identity] = dict(result)
    samples_directory = locked_root / "samples"
    extras = (
        {candidate.resolve() for candidate in samples_directory.rglob("*.jsonl")} - sample_paths
        if samples_directory.is_dir()
        else set()
    )
    if extras:
        raise ValueError(
            f"unexpected locked sample files could contaminate results: {sorted(extras)}"
        )
    if set(attached) != set(run_ids):
        raise ValueError(
            "locked-test identities differ from matched training runs: "
            f"missing={sorted(set(run_ids) - set(attached))}, "
            f"extra={sorted(set(attached) - set(run_ids))}"
        )

    next_index = int(pd.to_numeric(frame["record_index"], errors="coerce").max()) + 1
    rows: list[dict[str, Any]] = []
    for offset, (identity, result) in enumerate(sorted(attached.items())):
        raw = {
            "kind": "evaluation",
            "split": "test",
            "method": identity[0],
            "seed": identity[1],
            "step": result["selected_step"],
            "selected_step": result["selected_step"],
            "selection_val_accuracy": result["selection_val_accuracy"],
            "test_accuracy": result["accuracy"],
            "evaluation_seconds": result.get("evaluation_seconds"),
            "policy_sync_seconds": result.get("policy_sync_seconds"),
            "locked_example_count": result["example_count"],
            "locked_correct_count": result["correct_count"],
        }
        row = _canonical_record(raw, path, offset + 1)
        row["record_index"] = next_index + offset
        row["run_id"] = run_ids[identity]
        rows.append(row)
    combined = pd.concat([frame, pd.DataFrame.from_records(rows)], ignore_index=True, sort=False)
    combined.attrs.update(frame.attrs)
    combined.attrs["locked_test_results_path"] = str(path.resolve())
    return combined.sort_values(
        ["method", "run_id", "step", "record_index"], na_position="last"
    ).reset_index(drop=True)


def _deduplicate_run_points(
    frame: pd.DataFrame, *, metric: str, coordinates: Sequence[str]
) -> pd.DataFrame:
    required = {"method", "run_id", metric, *coordinates}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"results are missing required columns: {sorted(missing)}")
    subset = frame.copy()
    subset = subset[np.isfinite(pd.to_numeric(subset[metric], errors="coerce"))]
    for coordinate in coordinates:
        subset = subset[np.isfinite(pd.to_numeric(subset[coordinate], errors="coerce"))]
    if subset.empty:
        return subset
    order = ["record_index"] if "record_index" in subset else coordinates
    subset = subset.sort_values(order)
    return subset.drop_duplicates(["run_id", *coordinates], keep="last")


def _evaluation_records(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select canonical validation or official-test evaluation records."""

    subset = frame[frame["kind"] == "evaluation"]
    if "evaluation_split" not in subset:
        # DataFrames assembled by callers rather than :func:`load_results`
        # predate split canonicalisation and therefore follow the historical
        # convention that recurring evaluations are validation measurements.
        return subset if split == "validation" else subset.iloc[0:0]
    canonical_split = subset["evaluation_split"].fillna("validation").astype(str)
    return subset[canonical_split == split]


def aggregate_metrics(
    frame: pd.DataFrame,
    *,
    metric: str = "score",
    by: str = "step",
    kind: str | None = "evaluation",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Aggregate a metric by method and step/sample count with a 95% CI."""

    if by not in {"step", "environment_samples"}:
        raise ValueError("by must be 'step' or 'environment_samples'")
    if kind == "evaluation":
        # Learning-curve aggregation must never fold the one-shot official
        # test measurement into the validation trajectory at the same step.
        subset = _evaluation_records(frame, "validation")
    else:
        subset = frame
    if kind is not None and kind != "evaluation":
        subset = subset[subset["kind"] == kind]
    subset = _deduplicate_run_points(subset, metric=metric, coordinates=(by,))
    rows: list[dict[str, Any]] = []
    for (method, coordinate), group in subset.groupby(["method", by], sort=True):
        estimate = bootstrap_mean_ci(
            group[metric],
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, metric, by, method, coordinate),
        )
        rows.append(
            {
                "method": method,
                by: float(coordinate),
                "metric": metric,
                "mean": estimate.mean,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "count": estimate.count,
            }
        )
    return pd.DataFrame.from_records(rows)


def _curve_by_step(
    frame: pd.DataFrame,
    *,
    metric: str,
    x_column: str,
    kind: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    subset = (
        _evaluation_records(frame, "validation")
        if kind == "evaluation"
        else frame[frame["kind"] == kind]
    )
    subset = _deduplicate_run_points(subset, metric=metric, coordinates=("step",))
    subset = subset[np.isfinite(pd.to_numeric(subset[x_column], errors="coerce"))]
    rows: list[dict[str, Any]] = []
    for (method, step), group in subset.groupby(["method", "step"], sort=True):
        estimate = bootstrap_mean_ci(
            group[metric],
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, metric, x_column, method, step),
        )
        x_estimate = bootstrap_mean_ci(
            group[x_column],
            samples=bootstrap_samples,
            seed=_stable_seed(bootstrap_seed, x_column, method, step),
        )
        rows.append(
            {
                "method": method,
                "step": float(step),
                x_column: x_estimate.mean,
                "mean": estimate.mean,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "count": estimate.count,
            }
        )
    return pd.DataFrame.from_records(rows)


def _run_auc(group: pd.DataFrame, x_column: str, metric: str) -> tuple[float, float]:
    points = group[[x_column, metric]].apply(pd.to_numeric, errors="coerce").dropna()
    points = points[np.isfinite(points[x_column]) & np.isfinite(points[metric])]
    if points.empty:
        return float("nan"), float("nan")
    points = points.groupby(x_column, as_index=False)[metric].mean().sort_values(x_column)
    x = points[x_column].to_numpy(dtype=float)
    y = points[metric].to_numpy(dtype=float)
    if x.size == 1:
        return 0.0, float(y[0])
    area = float(np.trapezoid(y, x))
    span = float(x[-1] - x[0])
    normalised = area / span if span > 0 else float(y[-1])
    return area, normalised


def _last_finite(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _sum_finite(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.sum()) if not values.empty else float("nan")


def _sum_complete(group: pd.DataFrame, column: str) -> float:
    """Sum telemetry only when every row reports the component.

    A partial sum would understate training time while still looking precise.
    The matched compute comparison therefore treats an omitted per-step phase
    component as unavailable rather than silently converting it to zero.
    """

    if group.empty or column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    if len(values) != int(np.isfinite(values).sum()):
        return float("nan")
    return float(values.sum())


def _max_finite(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.max()) if not values.empty else float("nan")


def _per_run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in frame.groupby("run_id", sort=True):
        group = group.sort_values(["step", "record_index"], na_position="last")
        evaluations = group[group["kind"] == "evaluation"]
        if evaluations.empty:
            evaluations = group[
                np.isfinite(group["accuracy"]) | np.isfinite(group["expected_reward"])
            ]
        validation_evaluations = _evaluation_records(group, "validation")
        test_evaluations = _evaluation_records(group, "test")
        reported_evaluations = (
            test_evaluations
            if not test_evaluations.empty
            else validation_evaluations
            if not validation_evaluations.empty
            else evaluations
        )
        final_step_validation_accuracy = _last_finite(validation_evaluations, "accuracy")
        locked_test_accuracy = _last_finite(test_evaluations, "accuracy")
        selected_validation_accuracy = _last_finite(test_evaluations, "selection_accuracy")
        if not math.isfinite(selected_validation_accuracy):
            selected_validation_accuracy = _last_finite(
                validation_evaluations, "selection_accuracy"
            )
        selected_step = _last_finite(test_evaluations, "selected_step")
        if not math.isfinite(selected_step):
            selected_step = _last_finite(validation_evaluations, "selected_step")
        if math.isfinite(selected_step) and not math.isfinite(selected_validation_accuracy):
            at_selected_step = validation_evaluations[
                pd.to_numeric(validation_evaluations["step"], errors="coerce") == selected_step
            ]
            selected_validation_accuracy = _last_finite(at_selected_step, "accuracy")
        has_official_test = not test_evaluations.empty
        if has_official_test:
            # The sealed test evaluates the development-selected checkpoint;
            # it is neither the development selection metric nor necessarily
            # the last optimizer-step policy.
            reported_accuracy = locked_test_accuracy
            performance_source = "locked_test_selected_checkpoint"
        elif math.isfinite(selected_validation_accuracy):
            reported_accuracy = selected_validation_accuracy
            performance_source = "validation_selected_checkpoint"
        else:
            reported_accuracy = final_step_validation_accuracy
            performance_source = "validation_final_step"
        training = group[group["kind"] == "train_step"]
        score_sources = [
            str(value)
            for value in reported_evaluations.get("score_source", pd.Series(dtype=object)).dropna()
        ]
        total_policy_sync_seconds = _sum_complete(training, "policy_sync_seconds")
        total_rollout_seconds = _sum_complete(training, "rollout_and_old_score_seconds")
        total_optimizer_seconds = _sum_complete(training, "optimizer_seconds")
        reported_training_phase_seconds = _sum_complete(training, "training_phase_seconds")
        phase_components = (
            total_policy_sync_seconds,
            total_rollout_seconds,
            total_optimizer_seconds,
        )
        total_training_phase_seconds = (
            sum(phase_components)
            if all(math.isfinite(value) for value in phase_components)
            else float("nan")
        )
        if (
            math.isfinite(reported_training_phase_seconds)
            and math.isfinite(total_training_phase_seconds)
            and not math.isclose(
                reported_training_phase_seconds,
                total_training_phase_seconds,
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"run {run_id} training-phase telemetry does not equal "
                "policy sync + rollout/old score + optimizer"
            )
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": str(group["method"].iloc[0]),
            "seed": group["seed"].iloc[0],
            "score_source": Counter(score_sources).most_common(1)[0][0] if score_sources else "",
            "performance_source": performance_source,
            "selected_step": selected_step,
            "selected_checkpoint_validation_accuracy": selected_validation_accuracy,
            "final_step_validation_accuracy": final_step_validation_accuracy,
            "locked_test_accuracy": locked_test_accuracy,
            # Backward-compatible aliases. ``final_accuracy`` is the primary
            # reported measurement, while ``endpoint_accuracy`` is explicitly
            # the final optimizer-step validation measurement.
            "endpoint_accuracy": final_step_validation_accuracy,
            "final_score": (
                reported_accuracy
                if math.isfinite(reported_accuracy)
                else _last_finite(reported_evaluations, "score")
            ),
            "final_accuracy": reported_accuracy,
            "final_expected_reward": _last_finite(reported_evaluations, "expected_reward"),
            "final_environment_samples": _last_finite(group, "environment_samples"),
            "final_wall_time_seconds": _last_finite(group, "wall_time_seconds"),
            "final_forward_calls": _last_finite(group, "forward_calls"),
            "final_full_prefix_calls": _last_finite(group, "full_prefix_calls"),
            "final_suffix_calls": _last_finite(group, "suffix_calls"),
            "final_backward_calls": _last_finite(group, "backward_calls"),
            "final_teacher_forced_examples": _last_finite(group, "teacher_forced_examples"),
            "total_policy_sync_seconds": total_policy_sync_seconds,
            "total_rollout_and_old_score_seconds": total_rollout_seconds,
            "total_optimizer_seconds": total_optimizer_seconds,
            "total_training_phase_seconds": total_training_phase_seconds,
            "total_evaluation_seconds": _sum_finite(evaluations, "evaluation_seconds"),
            "peak_gpu_memory_bytes": _max_finite(group, "peak_gpu_memory_bytes"),
            "peak_gpu_memory_allocated_bytes": _max_finite(
                group, "peak_gpu_memory_allocated_bytes"
            ),
            "peak_gpu_memory_reserved_bytes": _max_finite(group, "peak_gpu_memory_reserved_bytes"),
            "acceptance_rate": float(
                pd.to_numeric(training["acceptance_rate"], errors="coerce").mean()
            )
            if not training.empty
            else float("nan"),
        }
        for x_column, suffix in (
            ("environment_samples", "environment_samples"),
            ("wall_time_seconds", "wall_time_seconds"),
        ):
            area, normalised = _run_auc(validation_evaluations, x_column, "score")
            row[f"auc_{suffix}"] = area
            row[f"normalised_auc_{suffix}"] = normalised
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _add_estimate_columns(
    destination: dict[str, Any],
    values: Iterable[float],
    prefix: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    group_key: str,
) -> None:
    estimate = bootstrap_mean_ci(
        values,
        samples=bootstrap_samples,
        seed=_stable_seed(bootstrap_seed, group_key, prefix),
    )
    destination[f"{prefix}_mean"] = estimate.mean
    destination[f"{prefix}_ci_low"] = estimate.ci_low
    destination[f"{prefix}_ci_high"] = estimate.ci_high
    destination[f"{prefix}_count"] = estimate.count


def summarize_results(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Create one method-level row of final, efficiency, and resource metrics."""

    run_summary = _per_run_summary(frame)
    rows: list[dict[str, Any]] = []
    estimate_columns = (
        "final_score",
        "final_accuracy",
        "endpoint_accuracy",
        "selected_checkpoint_validation_accuracy",
        "final_step_validation_accuracy",
        "locked_test_accuracy",
        "selected_step",
        "final_expected_reward",
        "auc_environment_samples",
        "normalised_auc_environment_samples",
        "auc_wall_time_seconds",
        "normalised_auc_wall_time_seconds",
        "final_environment_samples",
        "final_wall_time_seconds",
        "final_forward_calls",
        "final_full_prefix_calls",
        "final_suffix_calls",
        "final_backward_calls",
        "final_teacher_forced_examples",
        "total_policy_sync_seconds",
        "total_rollout_and_old_score_seconds",
        "total_optimizer_seconds",
        "total_training_phase_seconds",
        "total_evaluation_seconds",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "acceptance_rate",
    )
    for method, group in run_summary.groupby("method", sort=False):
        if group["seed"].duplicated().any():
            duplicates = sorted(str(seed) for seed in group.loc[group["seed"].duplicated(), "seed"])
            raise ValueError(
                f"method {method!r} has multiple run files for the same seed: {duplicates}"
            )
        sources = [source for source in group["score_source"] if source]
        row: dict[str, Any] = {
            "method": method,
            "display_name": _display_name(method),
            "runs": len(group),
            "seeds": int(group["seed"].nunique()),
            "ci_description": (
                MATCHED_CI_DESCRIPTION
                if group["seed"].nunique() > 1
                else "point estimate only (n=1); no across-seed uncertainty interval"
            ),
            "uncertainty_status": (
                "seed_bootstrap_interval"
                if group["seed"].nunique() > 1
                else "single_seed_no_interval"
            ),
            "training_time_scope": "policy sync + rollout/old-policy score + optimizer",
            "memory_measurement_scope": MATCHED_MEMORY_SCOPE,
            "forward_call_accounting_scope": "physical audit invocations; not FLOP-equivalent",
            "primary_metric_sources": ";".join(sorted(set(sources))),
            "performance_sources": ";".join(
                sorted({str(value) for value in group["performance_source"].dropna()})
            ),
        }
        for column in estimate_columns:
            _add_estimate_columns(
                row,
                group[column],
                column,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                group_key=method,
            )
        rows.append(row)
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return summary
    order = {method: index for index, method in enumerate(_ordered_methods(summary["method"]))}
    summary["_order"] = summary["method"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def paired_method_comparisons(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Summarize seed-paired forward-only differences relative to BP-GRPO."""

    runs = _per_run_summary(frame)
    if runs.duplicated(["method", "seed"]).any():
        raise ValueError("paired comparisons require one run per method/seed identity")
    reference = runs[runs["method"] == "bp_grpo"].set_index("seed")
    rows: list[dict[str, Any]] = []
    for method in ("fo_npg", "fo_focus_npg"):
        candidate = runs[runs["method"] == method].set_index("seed")
        common = sorted(set(reference.index) & set(candidate.index))
        if not common:
            continue
        pairs = candidate.loc[common].join(
            reference.loc[common],
            how="inner",
            lsuffix="_candidate",
            rsuffix="_reference",
        )
        primary_delta = pairs["final_accuracy_candidate"] - pairs["final_accuracy_reference"]
        time_ratio = (
            pairs["total_training_phase_seconds_candidate"]
            / pairs["total_training_phase_seconds_reference"]
        )
        memory_delta = (
            pairs["peak_gpu_memory_allocated_bytes_candidate"]
            - pairs["peak_gpu_memory_allocated_bytes_reference"]
        )
        row: dict[str, Any] = {
            "method": method,
            "display_name": _display_name(method),
            "reference_method": "bp_grpo",
            "reference_display_name": _display_name("bp_grpo"),
            "paired_seeds": ";".join(str(seed) for seed in common),
            "ci_description": (
                MATCHED_CI_DESCRIPTION
                if len(common) > 1
                else "point estimate only (n=1); no across-seed uncertainty interval"
            ),
            "uncertainty_status": (
                "seed_bootstrap_interval" if len(common) > 1 else "single_seed_no_interval"
            ),
        }
        for prefix, values in (
            ("primary_accuracy_delta", primary_delta),
            ("training_time_ratio", time_ratio),
            ("allocated_memory_delta_bytes", memory_delta),
        ):
            _add_estimate_columns(
                row,
                values,
                prefix,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                group_key=f"{method}-minus-bp_grpo",
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _ordered_methods(methods: Iterable[str]) -> list[str]:
    unique = {str(method) for method in methods}
    return sorted(unique, key=lambda method: (METHOD_PRIORITY.get(method, 100), method))


def _display_name(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _method_styles(methods: Iterable[str]) -> dict[str, tuple[str, str]]:
    ordered = _ordered_methods(methods)
    canonical = {
        "base": ("#6B7280", "o"),
        "bp_grpo": ("#D55E00", "s"),
        "grpo": ("#D55E00", "s"),
        "standard_grpo": ("#D55E00", "s"),
        "fo_npg": ("#009E73", "^"),
        "fo_focus_npg": ("#CC79A7", "D"),
        "focus_npg": ("#CC79A7", "D"),
    }
    styles: dict[str, tuple[str, str]] = {}
    for index, method in enumerate(ordered):
        if method in canonical:
            styles[method] = canonical[method]
        else:
            styles[method] = (
                METHOD_COLOURS[index % len(METHOD_COLOURS)],
                METHOD_MARKERS[index % len(METHOD_MARKERS)],
            )
    return styles


def _score_axis(ax: plt.Axes, values: Iterable[float], label: str) -> None:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.set_ylabel(label)
    if finite.size and finite.min() >= -0.02 and finite.max() <= 1.02:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        lower = min(0.0, float(finite.min()) - 0.04)
        upper = min(1.02, max(0.10, float(finite.max()) + 0.08))
        ax.set_ylim(lower, upper)


def _x_score_axis(ax: plt.Axes, values: Iterable[float]) -> None:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size and finite.min() >= -0.02 and finite.max() <= 1.02:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlim(min(0.0, float(finite.min()) - 0.04), 1.02)


def _primary_score_label(frame: pd.DataFrame) -> str:
    if np.isfinite(pd.to_numeric(frame["accuracy"], errors="coerce")).any():
        return "Exact-match accuracy"
    return "Expected reward"


def _ci_note(counts: Iterable[float]) -> str:
    finite = np.asarray(list(counts), dtype=float)
    finite = finite[np.isfinite(finite)]
    unique = sorted({int(value) for value in finite})
    if unique == [1]:
        return "One seed per method: points only; no variance estimate"
    if unique == [1, 3]:
        return "95% seed-bootstrap CI for trained methods (n=3); Base (n=1) is a point"
    if unique == [3]:
        return "95% percentile-bootstrap CI across three independent seeds"
    return "95% seed-bootstrap CI; singleton methods are point estimates"


def _plotting_estimate(row: Mapping[str, Any], metric: str) -> tuple[float, float, float]:
    mean = float(row.get(f"{metric}_mean", float("nan")))
    low = float(row.get(f"{metric}_ci_low", float("nan")))
    high = float(row.get(f"{metric}_ci_high", float("nan")))
    count = _as_float(row.get(f"{metric}_count"))
    if math.isfinite(mean) and (count == 1 or not (math.isfinite(low) and math.isfinite(high))):
        low = high = mean
    return mean, low, high


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color="#6B7280",
        transform=ax.transAxes,
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[extension] = path
    plt.close(fig)
    return paths


def find_gsm8k_fd_diagnostic(input_dir: str | Path) -> Path | None:
    """Find the real-model finite-difference diagnostic near benchmark runs.

    The benchmark and diagnostic commands intentionally write independent
    artifacts. Report generation therefore looks first below ``input_dir``
    and then in a sibling ``diagnostics`` directory. The latter matches the
    standard layout ``artifacts/{final_gsm8k,diagnostics}``. Returning
    ``None`` keeps plotting useful for smoke runs that did not run the costly
    real-model diagnostic.
    """

    source = Path(input_dir)
    if source.is_file() and source.name == "gsm8k_fd.json":
        return source
    root = source.parent if source.is_file() else source

    direct_candidates = (
        root / "gsm8k_fd.json",
        root / "diagnostics" / "gsm8k_fd.json",
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    recursive = sorted(path for path in root.rglob("gsm8k_fd.json") if path.is_file())
    if recursive:
        return recursive[0]

    candidate = root.parent / "diagnostics" / "gsm8k_fd.json"
    if candidate.is_file():
        return candidate

    # ``raw`` is a commonly supplied input subdirectory. In that case the
    # standard diagnostics sibling is one additional level above it.
    if root.name == "raw":
        candidate = root.parent.parent / "diagnostics" / "gsm8k_fd.json"
        if candidate.is_file():
            return candidate
    return None


def _finite_difference_diagnostic_rows(report: Mapping[str, Any]) -> pd.DataFrame:
    """Extract plotting columns from a GSM8K diagnostic JSON object."""

    finite_difference = report.get("finite_difference")
    if not isinstance(finite_difference, Mapping):
        raise TypeError("diagnostic JSON is missing finite_difference object")
    raw_results = finite_difference.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise TypeError("diagnostic JSON is missing finite_difference.results array")

    rows: list[dict[str, float]] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            continue
        agreement = raw_result.get("agreement_with_exact_projected_gradient")
        fisher = raw_result.get("fisher")
        if not isinstance(agreement, Mapping) or not isinstance(fisher, Mapping):
            continue
        row = {
            "mu": pd.to_numeric(raw_result.get("mu"), errors="coerce"),
            "cosine_similarity": pd.to_numeric(agreement.get("cosine_similarity"), errors="coerce"),
            "relative_l2_error": pd.to_numeric(agreement.get("relative_l2_error"), errors="coerce"),
            "legacy_to_correct_trace_ratio": pd.to_numeric(
                fisher.get("legacy_to_correct_trace_ratio"), errors="coerce"
            ),
        }
        if math.isfinite(float(row["mu"])) and float(row["mu"]) > 0.0:
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError("diagnostic JSON contains no plottable finite-difference results")
    return pd.DataFrame(rows).sort_values("mu").reset_index(drop=True)


def plot_finite_difference_diagnostics(
    diagnostic_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render finite-difference agreement and Fisher-curvature diagnostics.

    The left panel compares each central-difference radius against an exact,
    non-updating projected-gradient oracle. The right panel reports the
    legacy completion-outer Fisher trace divided by the token-local trace that
    matches the benchmark's length-normalized sequence KL.
    """

    diagnostic_path = Path(diagnostic_json)
    try:
        report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read diagnostic JSON {diagnostic_path}: {error}") from error
    if not isinstance(report, Mapping):
        raise TypeError("diagnostic JSON root must be an object")
    rows = _finite_difference_diagnostic_rows(report)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mus = rows["mu"].to_numpy(dtype=float)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15))

        agreement_specs = (
            ("cosine_similarity", "Cosine similarity", "#0072B2", "o", "-"),
            ("relative_l2_error", "Relative L2 error", "#D55E00", "s", "--"),
        )
        plotted_agreement = False
        agreement_values: list[float] = []
        for column, label, colour, marker, line_style in agreement_specs:
            values = rows[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            axes[0].plot(
                mus[finite],
                values[finite],
                color=colour,
                marker=marker,
                linestyle=line_style,
                markersize=5,
                label=label,
            )
            agreement_values.extend(values[finite])
            plotted_agreement = True
        axes[0].set_title("Projected-gradient fidelity", loc="left", fontweight="bold")
        axes[0].set_xlabel("Finite-difference radius, μ")
        axes[0].set_ylabel("Dimensionless agreement metric")
        if plotted_agreement:
            axes[0].legend(loc="best")
            lower = min(0.0, min(agreement_values) - 0.06)
            upper = max(1.02, max(agreement_values) + 0.06)
            axes[0].set_ylim(lower, upper)
        else:
            _empty_panel(axes[0], "No gradient-agreement metrics")

        ratios = rows["legacy_to_correct_trace_ratio"].to_numpy(dtype=float)
        finite_ratios = np.isfinite(ratios) & (ratios >= 0.0)
        axes[1].set_title("Fisher curvature mismatch", loc="left", fontweight="bold")
        axes[1].set_xlabel("Finite-difference radius, μ")
        axes[1].set_ylabel("Legacy completion trace / token-local trace")
        if finite_ratios.any():
            axes[1].plot(
                mus[finite_ratios],
                ratios[finite_ratios],
                color="#009E73",
                marker="D",
                markersize=5,
            )
            axes[1].fill_between(
                mus[finite_ratios],
                0.0,
                ratios[finite_ratios],
                color="#009E73",
                alpha=0.12,
                linewidth=0,
            )
            axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ratio_max = float(ratios[finite_ratios].max())
            axes[1].set_ylim(0.0, max(0.05, ratio_max * 1.28))
            axes[1].text(
                0.02,
                0.97,
                "100% would indicate equal trace",
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                color="#6B7280",
                fontsize=8,
            )
        else:
            _empty_panel(axes[1], "No Fisher-trace ratios")

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(mus)
            ax.set_xticklabels([f"{mu:g}" for mu in mus])
            ax.tick_params(axis="x", which="minor", labelbottom=False)
        fig.suptitle(
            "Forward-only estimator validation on fixed GSM8K rollouts",
            fontweight="bold",
        )
        return _save_figure(fig, output, "fd_diagnostics")


def plot_matched_gradient_oracle(
    diagnostic_json: str | Path,
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Render the exact matched-LoRA oracle bound into benchmark metadata."""

    path = Path(diagnostic_json)
    diagnostic = _read_json_mapping(path, description="matched-LoRA gradient oracle")
    bound = metadata.get("projected_gradient_oracle")
    if (
        diagnostic.get("schema") != MATCHED_GRADIENT_ORACLE_SCHEMA
        or not isinstance(bound, Mapping)
        or diagnostic != dict(bound)
    ):
        raise ValueError(
            "headline diagnostic must be the matched-LoRA gradient oracle exactly bound "
            "into benchmark metadata"
        )
    report = diagnostic.get("report")
    thresholds = diagnostic.get("thresholds")
    if not isinstance(report, Mapping) or not isinstance(thresholds, Mapping):
        raise TypeError("matched-LoRA oracle is missing report/threshold mappings")
    cosine = _as_float(report.get("cosine_similarity"))
    relative_error = _as_float(report.get("relative_l2_error"))
    minimum_cosine = _as_float(thresholds.get("minimum_cosine_similarity"))
    maximum_error = _as_float(thresholds.get("maximum_relative_l2_error"))
    exact_norm = _as_float(report.get("exact_projected_norm"))
    finite_difference_norm = _as_float(report.get("finite_difference_norm"))
    mu = _as_float(report.get("finite_difference_mu"))
    directions = report.get("directions")
    if (
        diagnostic.get("passed") is not True
        or diagnostic.get("adapter_parameter_count") != MATCHED_LORA_PARAMETER_COUNT
        or report.get("parameter_integrity") is not True
        or report.get("parameter_digest_before") != report.get("parameter_digest_after")
        or diagnostic.get("frozen_base_parameter_digest_before")
        != diagnostic.get("frozen_base_parameter_digest_after")
        or not all(
            math.isfinite(value)
            for value in (
                cosine,
                relative_error,
                minimum_cosine,
                maximum_error,
                exact_norm,
                finite_difference_norm,
                mu,
            )
        )
        or cosine < minimum_cosine
        or relative_error > maximum_error
        or min(exact_norm, finite_difference_norm, mu) <= 0.0
        or isinstance(directions, bool)
        or not isinstance(directions, int)
        or directions < 1
    ):
        raise ValueError("matched-LoRA gradient oracle is malformed or did not pass its gates")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
        axes[0].bar(["Observed"], [cosine], color="#0072B2", width=0.55)
        axes[0].axhline(
            minimum_cosine,
            color="#6B7280",
            linestyle="--",
            label=f"Required ≥ {minimum_cosine:.3f}",
        )
        axes[0].set_ylim(max(0.0, min(minimum_cosine, cosine) - 0.08), 1.01)
        axes[0].set_ylabel("Cosine similarity")
        axes[0].set_title("Projected-direction agreement", loc="left", fontweight="bold")
        axes[0].legend(loc="lower right")

        axes[1].bar(["Observed"], [relative_error], color="#D55E00", width=0.55)
        axes[1].axhline(
            maximum_error,
            color="#6B7280",
            linestyle="--",
            label=f"Required ≤ {maximum_error:.3f}",
        )
        axes[1].set_ylim(0.0, max(maximum_error, relative_error) * 1.25)
        axes[1].set_ylabel("Relative L2 error")
        axes[1].set_title("Finite-scale relative error", loc="left", fontweight="bold")
        axes[1].legend(loc="upper right")
        fig.suptitle(
            f"Matched-LoRA projected-gradient oracle · q={directions}, μ={mu:g}",
            fontweight="bold",
        )
        return _save_figure(fig, output, "projected_gradient_oracle")


def _plot_learning_curve(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    x_column: str,
    filename: str,
    title: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    aggregate = _curve_by_step(
        frame,
        metric="score",
        x_column=x_column,
        kind="evaluation",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    methods = _ordered_methods(aggregate["method"] if not aggregate.empty else [])
    styles = _method_styles(methods)
    all_values: list[float] = []
    finite_x = pd.to_numeric(aggregate.get(x_column), errors="coerce").to_numpy(dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    x_extent = (float(finite_x.min()), float(finite_x.max())) if finite_x.size else (0.0, 0.0)
    for method in methods:
        group = aggregate[aggregate["method"] == method].sort_values(x_column)
        x = group[x_column].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        colour, marker = styles[method]
        line_style = "-"
        if method == "base" and len(x) == 1 and x_extent[1] > x_extent[0]:
            # A frozen policy consumes no training samples. Draw its one
            # deterministic evaluation as a horizontal reference instead of
            # leaving an easy-to-miss point at the origin.
            x = np.asarray(x_extent)
            mean = np.repeat(mean, 2)
            low = np.repeat(low, 2)
            high = np.repeat(high, 2)
            line_style = "--"
        ax.plot(
            x,
            mean,
            color=colour,
            linestyle=line_style,
            marker=marker,
            markevery=max(1, len(x) // 6),
            markersize=4.5,
            label=_display_name(method),
        )
        if np.any(high > low):
            ax.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
        all_values.extend(low)
        all_values.extend(high)
    if not methods:
        _empty_panel(ax, "No evaluation records found")
    else:
        ax.legend(loc="best", ncols=2)
        _score_axis(ax, all_values, _primary_score_label(frame))
    ax.set_title(
        f"{title}\n{_ci_note(aggregate['count'] if not aggregate.empty else [])}",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    if x_column == "environment_samples":
        ax.set_xlabel("Environment samples")
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    else:
        ax.set_xlabel("Wall time (seconds)")
        ax.xaxis.set_major_formatter(EngFormatter(unit="s", sep=""))
    return _save_figure(fig, output_dir, filename)


def _plot_final_performance(
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    metrics: list[tuple[str, str]] = []
    for metric, label in (
        ("locked_test_accuracy", "Locked test · selected checkpoint"),
        (
            "selected_checkpoint_validation_accuracy",
            "Development · selected checkpoint",
        ),
        ("final_step_validation_accuracy", "Development · final optimizer step"),
    ):
        if f"{metric}_mean" in summary and summary[f"{metric}_mean"].notna().any():
            metrics.append((metric, label))
    if not metrics and (
        "final_expected_reward_mean" in summary
        and summary["final_expected_reward_mean"].notna().any()
    ):
        metrics.append(("final_expected_reward", "Expected reward"))
    if not metrics:
        metrics.append(("final_score", "Final score"))

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.6 * len(metrics), max(3.7, 1.2 + 0.55 * len(summary))),
        sharey=True,
        squeeze=False,
    )
    methods = _ordered_methods(summary["method"])
    styles = _method_styles(methods)
    positions = np.arange(len(methods))[::-1]
    indexed = summary.set_index("method")
    for index, (metric, label) in enumerate(metrics):
        ax = axes[0, index]
        values: list[float] = []
        for y, method in zip(positions, methods, strict=True):
            if method not in indexed.index:
                continue
            row = indexed.loc[method]
            mean, low, high = _plotting_estimate(row, metric)
            if not math.isfinite(mean):
                continue
            colour, marker = styles[method]
            x_error = np.array([[max(0.0, mean - low)], [max(0.0, high - mean)]])
            ax.errorbar(
                mean,
                y,
                xerr=x_error,
                fmt=marker,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7,
                capsize=3,
                elinewidth=1.7,
            )
            values.extend([low, high])
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("Exact-match accuracy")
        ax.set_yticks(positions, [_display_name(method) for method in methods])
        _x_score_axis(ax, values)
    fig.suptitle(
        f"Checkpoint-aware performance\n{_ci_note(summary['final_score_count'])}",
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "final_performance")


def _plot_compute_memory(
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.25), sharey=True)
    methods = _ordered_methods(summary["method"])
    styles = _method_styles(methods)
    indexed = summary.set_index("method")
    has_phase_timing = (
        "total_training_phase_seconds_mean" in summary
        and summary["total_training_phase_seconds_mean"].notna().any()
    )
    compute_specification = (
        (
            "total_training_phase_seconds_mean",
            "Total training phase (seconds)\npolicy sync + rollout/old score + optimizer",
            1.0,
        )
        if has_phase_timing
        else (
            "final_teacher_forced_examples_mean",
            "Teacher-forced examples (audit count; not FLOPs)",
            1.0,
        )
    )
    specifications = (
        compute_specification,
        (
            "peak_gpu_memory_allocated_bytes_mean",
            "Peak allocated GPU memory (GiB)\ncombined in-process HF + vLLM",
            2**30,
        ),
    )
    score_values: list[float] = []
    for ax, (x_column, x_label, divisor) in zip(axes, specifications, strict=True):
        x_values: list[float] = []
        plotted = False
        for method in methods:
            row = indexed.loc[method]
            x = float(row.get(x_column, float("nan"))) / divisor
            y, low, high = _plotting_estimate(row, "final_score")
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            colour, marker = styles[method]
            ax.errorbar(
                x,
                y,
                yerr=np.array([[max(0.0, y - low)], [max(0.0, high - y)]]),
                fmt=marker,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7,
                capsize=2.5,
                label=_display_name(method),
            )
            x_values.append(x)
            score_values.extend([low, high])
            plotted = True
        ax.set_xlabel(x_label)
        ax.set_title(
            "Training-phase efficiency"
            if "memory" not in x_column
            else "Combined-process memory footprint",
            loc="left",
            fontweight="bold",
        )
        if "memory" not in x_column:
            ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        if not plotted:
            _empty_panel(ax, f"No {x_label.lower()} telemetry")
        elif "memory" in x_column and x_values and max(x_values) == 0.0:
            ax.text(
                0.02,
                0.04,
                "0 GiB reported (CPU run or telemetry unavailable)",
                transform=ax.transAxes,
                color="#6B7280",
                fontsize=8,
            )
    _score_axis(axes[0], score_values, "Primary selected-checkpoint score")
    handles = [
        Line2D(
            [0],
            [0],
            color=styles[method][0],
            marker=styles[method][1],
            linestyle="none",
            label=_display_name(method),
        )
        for method in methods
    ]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncols=min(4, len(handles)))
    fig.suptitle(
        f"Performance–resource trade-offs\n{_ci_note(summary['final_score_count'])}",
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "compute_memory_tradeoffs")


def _plot_paired_comparisons(
    paired: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, max(3.8, 1.7 + 0.65 * len(paired))))
    if paired.empty:
        for ax in axes:
            _empty_panel(ax, "No seed-paired FO/BP runs")
        return _save_figure(fig, output_dir, "paired_method_comparisons")
    methods = _ordered_methods(paired["method"])
    styles = _method_styles(methods)
    indexed = paired.set_index("method")
    positions = np.arange(len(methods))[::-1]
    specifications = (
        (
            "primary_accuracy_delta",
            100.0,
            0.0,
            "Primary accuracy delta (percentage points)\nFO − BP-GRPO",
            "Accuracy",
        ),
        (
            "training_time_ratio",
            1.0,
            1.0,
            "Training-phase time / BP-GRPO (×)\npolicy sync + rollout/old score + optimizer",
            "Training phase",
        ),
        (
            "allocated_memory_delta_bytes",
            1.0 / 2**30,
            0.0,
            "Allocated-memory delta (GiB)\nFO − BP-GRPO; combined HF + vLLM",
            "Memory",
        ),
    )
    for ax, (metric, scale, reference, xlabel, title) in zip(axes, specifications, strict=True):
        plotted = False
        for y, method in zip(positions, methods, strict=True):
            row = indexed.loc[method]
            raw_mean, raw_low, raw_high = _plotting_estimate(row, metric)
            mean, low, high = raw_mean * scale, raw_low * scale, raw_high * scale
            if not all(math.isfinite(value) for value in (mean, low, high)):
                continue
            colour, marker = styles[method]
            ax.errorbar(
                mean,
                y,
                xerr=np.array([[max(0.0, mean - low)], [max(0.0, high - mean)]]),
                fmt=marker,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7,
                capsize=3,
                elinewidth=1.7,
            )
            plotted = True
        ax.axvline(reference, color="#6B7280", linestyle=":", linewidth=1.3)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_yticks(positions, [_display_name(method) for method in methods])
        if not plotted:
            _empty_panel(ax, f"No paired {title.lower()} telemetry")
    counts = paired.get("primary_accuracy_delta_count", pd.Series(dtype=float))
    fig.suptitle(
        f"Seed-paired forward-only comparison against BP-GRPO\n{_ci_note(counts)}",
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "paired_method_comparisons")


def _headline_methods_present(frame: pd.DataFrame) -> list[str]:
    if "method" not in frame:
        return []
    observed = {str(method) for method in frame["method"].dropna()}
    return [method for method in MATCHED_HEADLINE_METHODS if method in observed]


def _rolling_metric_within_runs(
    frame: pd.DataFrame,
    *,
    metric: str,
    output_metric: str,
    window: int,
) -> pd.DataFrame:
    """Return a trailing step-window mean computed separately within each seed."""

    training = frame[
        (frame["kind"] == "train_step") & frame["method"].isin(MATCHED_HEADLINE_METHODS)
    ].copy()
    training = _deduplicate_run_points(training, metric=metric, coordinates=("step",))
    if training.empty:
        training[output_metric] = pd.Series(dtype=float)
        return training
    parts: list[pd.DataFrame] = []
    for _, group in training.groupby("run_id", sort=True):
        group = group.sort_values(["step", "record_index"]).copy()
        values = pd.to_numeric(group[metric], errors="coerce")
        group[output_metric] = values.rolling(window=window, min_periods=1).mean()
        parts.append(group)
    return pd.concat(parts, ignore_index=True, sort=False)


def _plot_seed_mean_curve(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    *,
    styles: Mapping[str, tuple[str, str]],
    linewidth: float = 2.0,
) -> list[float]:
    values: list[float] = []
    for method in _headline_methods_present(aggregate):
        group = aggregate[aggregate["method"] == method].sort_values("step")
        if group.empty:
            continue
        x = group["step"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        colour, marker = styles[method]
        ax.plot(
            x,
            mean,
            color=colour,
            marker=marker,
            markevery=max(1, len(x) // 8),
            markersize=3.8,
            linewidth=linewidth,
            label=_display_name(method),
        )
        finite_interval = np.isfinite(low) & np.isfinite(high)
        if finite_interval.any():
            ax.fill_between(
                x,
                low,
                high,
                where=finite_interval,
                color=colour,
                alpha=0.13,
                linewidth=0,
            )
        values.extend(mean[np.isfinite(mean)])
        values.extend(low[np.isfinite(low)])
        values.extend(high[np.isfinite(high)])
    return values


def _selected_validation_points(frame: pd.DataFrame) -> pd.DataFrame:
    validation = _evaluation_records(frame, "validation")
    validation = validation[validation["method"].isin(MATCHED_HEADLINE_METHODS)]
    rows: list[dict[str, Any]] = []
    for run_id, group in validation.groupby("run_id", sort=True):
        group = group.sort_values(["step", "record_index"])
        selected_step = _last_finite(group, "selected_step")
        selected_accuracy = _last_finite(group, "selection_accuracy")
        if not (math.isfinite(selected_step) and math.isfinite(selected_accuracy)):
            accuracies = pd.to_numeric(group["accuracy"], errors="coerce")
            finite = group[np.isfinite(accuracies)]
            if finite.empty:
                continue
            best = float(pd.to_numeric(finite["accuracy"], errors="coerce").max())
            best_rows = finite[
                np.isclose(
                    pd.to_numeric(finite["accuracy"], errors="coerce"),
                    best,
                    rtol=0.0,
                    atol=0.0,
                )
            ]
            selected_step = float(pd.to_numeric(best_rows["step"], errors="raise").max())
            selected_accuracy = best
        rows.append(
            {
                "run_id": run_id,
                "method": str(group["method"].iloc[0]),
                "seed": group["seed"].iloc[0],
                "selected_step": selected_step,
                "selected_accuracy": selected_accuracy,
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_reward_and_dev_trajectories(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    """Plot matched train exact reward and development exact-match trajectories."""

    methods = _headline_methods_present(frame)
    styles = _method_styles(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.55))

    raw_training = frame[(frame["kind"] == "train_step") & frame["method"].isin(methods)].copy()
    raw_training = _deduplicate_run_points(
        raw_training,
        metric="rollout_exact_reward",
        coordinates=("step",),
    )
    for method in methods:
        group = raw_training[raw_training["method"] == method]
        colour, _ = styles[method]
        axes[0].scatter(
            group["step"],
            group["rollout_exact_reward"],
            color=colour,
            alpha=0.23,
            s=15,
            linewidths=0,
            zorder=2,
        )
    rolling_metric = "rollout_exact_reward_trailing_mean"
    rolling = _rolling_metric_within_runs(
        frame,
        metric="rollout_exact_reward",
        output_metric=rolling_metric,
        window=TRAIN_REWARD_ROLLING_WINDOW,
    )
    rolling_aggregate = aggregate_metrics(
        rolling,
        metric=rolling_metric,
        by="step",
        kind="train_step",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    reward_values = _plot_seed_mean_curve(axes[0], rolling_aggregate, styles=styles)
    axes[0].set_title("Training rollout exact reward", loc="left", fontweight="bold")
    axes[0].set_xlabel("Optimizer step")
    _score_axis(axes[0], reward_values, "Exact reward")
    if raw_training.empty:
        _empty_panel(axes[0], "No training exact-reward records")

    validation = _evaluation_records(frame, "validation")
    validation = validation[validation["method"].isin(methods)]
    dev_aggregate = aggregate_metrics(
        validation,
        metric="accuracy",
        by="step",
        kind="evaluation",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    dev_values = _plot_seed_mean_curve(axes[1], dev_aggregate, styles=styles)
    selected = _selected_validation_points(frame)
    for method in methods:
        group = selected[selected["method"] == method]
        colour, _ = styles[method]
        axes[1].scatter(
            group["selected_step"],
            group["selected_accuracy"],
            marker="*",
            s=92,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        dev_values.extend(pd.to_numeric(group["selected_accuracy"], errors="coerce").dropna())
    axes[1].set_title(
        "Development exact match · selected checkpoints",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_xlabel("Optimizer step")
    _score_axis(axes[1], dev_values, "Development exact-match accuracy")
    if validation.empty:
        _empty_panel(axes[1], "No development exact-match records")

    method_handles = [
        Line2D(
            [0],
            [0],
            color=styles[method][0],
            marker=styles[method][1],
            label=_display_name(method),
        )
        for method in methods
    ]
    encoding_handles = [
        Line2D(
            [0],
            [0],
            color="#6B7280",
            marker="o",
            linewidth=0,
            alpha=0.45,
            label="Raw seed-step reward",
        ),
        Line2D(
            [0],
            [0],
            color="#374151",
            linewidth=2,
            label=(f"Seed mean of within-seed\n{TRAIN_REWARD_ROLLING_WINDOW}-step trailing means"),
        ),
    ]
    if method_handles:
        method_legend = axes[0].legend(
            handles=method_handles,
            loc="lower left",
            title="Method",
        )
        axes[0].add_artist(method_legend)
        axes[0].legend(
            handles=encoding_handles,
            loc="lower right",
            title="Training-reward display",
        )
        axes[1].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#374151",
                    marker="*",
                    markersize=10,
                    linewidth=0,
                    label="Selected checkpoint (each seed)",
                )
            ],
            loc="lower left",
        )
    fig.suptitle(
        "Matched BP-GRPO vs FO-NPG · reward and development trajectories",
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "training_reward_and_dev_accuracy")


def _plot_grpo_surrogate_change(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    """Plot the common matched surrogate change without inventing a loss scalar."""

    methods = _headline_methods_present(frame)
    styles = _method_styles(methods)
    training = frame[(frame["kind"] == "train_step") & frame["method"].isin(methods)].copy()
    training = _deduplicate_run_points(
        training,
        metric="surrogate_improvement",
        coordinates=("step",),
    )
    aggregate = aggregate_metrics(
        training,
        metric="surrogate_improvement",
        by="step",
        kind="train_step",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.65))
    for method in methods:
        group = training[training["method"] == method]
        colour, _ = styles[method]
        ax.scatter(
            group["step"],
            group["surrogate_improvement"],
            color=colour,
            alpha=0.20,
            s=14,
            linewidths=0,
        )
    values = _plot_seed_mean_curve(ax, aggregate, styles=styles)
    ax.axhline(0.0, color="#6B7280", linestyle=":", linewidth=1.2)
    ax.set_title(
        "Fresh-rollout matched GRPO surrogate change",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Surrogate improvement (post-update − pre-update)")
    if not training.empty:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size and float(finite.max() - finite.min()) < 1e-3:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    else:
        _empty_panel(ax, "No matched surrogate-change records")
    handles = [
        Line2D(
            [0],
            [0],
            color=styles[method][0],
            marker=styles[method][1],
            label=_display_name(method),
        )
        for method in methods
    ]
    if handles:
        ax.legend(handles=handles, loc="best")
    fig.suptitle("Comparable optimizer objective trace · not an absolute loss", fontweight="bold")
    fig.text(
        0.5,
        -0.04,
        "The fresh-rollout GRPO objective is zero-centered. This is the common post-minus-pre "
        "surrogate change, not an absolute training loss; FO-NPG has no backprop loss scalar.",
        ha="center",
        va="top",
        color="#4B5563",
        fontsize=8.7,
    )
    return _save_figure(fig, output_dir, "grpo_surrogate_objective_change")


def _plot_diagnostic_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    styles: Mapping[str, tuple[str, str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    aggregate = _curve_by_step(
        frame,
        metric=metric,
        x_column="step",
        kind="train_step",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    values: list[float] = []
    for method in _ordered_methods(aggregate["method"] if not aggregate.empty else []):
        group = aggregate[aggregate["method"] == method].sort_values("step")
        x = group["step"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        colour, marker = styles[method]
        ax.plot(
            x,
            mean,
            color=colour,
            marker=marker,
            markevery=max(1, len(x) // 5),
            markersize=3.5,
            label=_display_name(method),
        )
        if np.any(high > low):
            ax.fill_between(x, low, high, color=colour, alpha=0.12, linewidth=0)
        values.extend(low)
        values.extend(high)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    if not values:
        _empty_panel(ax, f"No {title.lower()} records")
    elif metric in {"zero_advantage_fraction", "acceptance_rate"}:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(-0.03, 1.03)
    elif metric in {
        "projected_gradient_norm",
        "projected_gradient_norm_relative_to_first_nonzero",
        "empirical_kl",
        "step_norm",
    }:
        ax.set_ylim(bottom=0.0)


def _plot_call_diagnostics(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    styles: Mapping[str, tuple[str, str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    plotted = False
    for metric, line_style in (("forward_calls", "-"), ("backward_calls", "--")):
        aggregate = _curve_by_step(
            frame,
            metric=metric,
            x_column="step",
            kind="train_step",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for method in _ordered_methods(aggregate["method"] if not aggregate.empty else []):
            group = aggregate[aggregate["method"] == method].sort_values("step")
            colour, _ = styles[method]
            ax.plot(
                group["step"],
                group["mean"],
                color=colour,
                linestyle=line_style,
            )
            plotted = True
    ax.set_title("Invocation audit · not a FLOP estimate", loc="left", fontweight="bold")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cumulative audit invocations")
    ax.yaxis.set_major_formatter(EngFormatter(sep=""))
    if plotted:
        call_legend = ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#374151",
                    linestyle="-",
                    label="Forward invocation",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#374151",
                    linestyle="--",
                    label="Backward invocation",
                ),
            ],
            loc="upper left",
        )
        ax.add_artist(call_legend)
    else:
        _empty_panel(ax, "No model-call records")


def _normalise_projected_gradient_norms(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize each run's gradient norm without comparing coordinate spaces."""

    output_metric = "projected_gradient_norm_relative_to_first_nonzero"
    normalised = frame.copy()
    normalised[output_metric] = float("nan")
    training = normalised[(normalised["kind"] == "train_step") & (normalised["method"] != "base")]
    for _, group in training.groupby("run_id", sort=True):
        group = group.sort_values(["step", "record_index"])
        values = pd.to_numeric(group["projected_gradient_norm"], errors="coerce")
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.empty:
            continue
        normalised.loc[group.index, output_metric] = values / float(positive.iloc[0])
    return normalised


def _plot_optimizer_diagnostics(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.25))
    methods = _ordered_methods(frame["method"])
    styles = _method_styles(methods)
    gradient_frame = _normalise_projected_gradient_norms(frame)
    _plot_diagnostic_metric(
        axes[0, 0],
        gradient_frame,
        metric="projected_gradient_norm_relative_to_first_nonzero",
        title="Gradient-norm trace · within-seed normalized",
        ylabel="Norm / first nonzero norm",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    axes[0, 0].text(
        0.02,
        0.03,
        GRADIENT_NORM_SCOPE_NOTE,
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        color="#4B5563",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )
    _plot_diagnostic_metric(
        axes[0, 1],
        frame,
        metric="empirical_kl",
        title="Empirical policy KL",
        ylabel="KL divergence",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    _plot_diagnostic_metric(
        axes[1, 0],
        frame,
        metric="step_norm",
        title="Applied policy-step norm",
        ylabel="LoRA-coordinate L2 norm",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    _plot_diagnostic_metric(
        axes[1, 1],
        frame,
        metric="acceptance_rate",
        title="Update acceptance",
        ylabel="Acceptance rate",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            color=styles[method][0],
            marker=styles[method][1],
            label=_display_name(method),
        )
        for method in methods
        if method != "base"
    ]
    if method_handles:
        fig.legend(
            handles=method_handles,
            loc="outside lower center",
            ncols=min(4, len(method_handles)),
        )
    fig.suptitle(
        "Optimizer health · matched BP-GRPO vs FO-NPG",
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "optimizer_diagnostics")


def plot_results(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    locked_test_results: str | Path | None = None,
    evaluation_manifest: str | Path | None = None,
    locked_source_receipt: str | Path | None = None,
    finite_difference_diagnostic: str | Path | None = None,
) -> dict[str, Path]:
    """Render one validated corrected matched-LoRA benchmark artifact.

    Each figure is emitted as both a high-resolution PNG for quick viewing and
    a vector PDF for the report.  Returned keys include the extension (for
    example ``"learning_curves_samples_png"``).  The answer-sealed test receipt
    and estimator diagnostic are exact-path opt-ins; neither is discovered by
    walking sibling artifact trees.
    """

    frame = load_matched_results(input_dir)
    if locked_test_results is not None:
        if evaluation_manifest is None or locked_source_receipt is None:
            raise ValueError(
                "locked-test plotting requires exact evaluation-manifest and v2 source-receipt paths"
            )
        frame = attach_locked_test_results(
            frame,
            locked_test_results,
            evaluation_manifest_path=evaluation_manifest,
            source_index_receipt_path=locked_source_receipt,
        )
    elif evaluation_manifest is not None or locked_source_receipt is not None:
        raise ValueError("locked manifest/source paths require --locked-test-results")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(
        frame,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired = paired_method_comparisons(
        frame,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8g")
    paired_path = output / "paired_comparisons.csv"
    paired.to_csv(paired_path, index=False, float_format="%.8g")

    artifacts: dict[str, Path] = {
        "summary_csv": summary_path,
        "paired_comparisons_csv": paired_path,
    }
    with plt.rc_context(PLOT_STYLE):
        figures = {
            "learning_curves_samples": _plot_learning_curve(
                frame,
                output,
                x_column="environment_samples",
                filename="learning_curves_samples",
                title="Learning curves · matched environment budget",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "learning_curves_wall_time": _plot_learning_curve(
                frame,
                output,
                x_column="wall_time_seconds",
                filename="learning_curves_wall_time",
                title="Learning curves · elapsed wall time",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "final_performance": _plot_final_performance(summary, output),
            "compute_memory_tradeoffs": _plot_compute_memory(summary, output),
            "paired_method_comparisons": _plot_paired_comparisons(paired, output),
            "training_reward_and_dev_accuracy": _plot_reward_and_dev_trajectories(
                frame,
                output,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "grpo_surrogate_objective_change": _plot_grpo_surrogate_change(
                frame,
                output,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "optimizer_diagnostics": _plot_optimizer_diagnostics(
                frame,
                output,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    for stem, paths in figures.items():
        for extension, path in paths.items():
            artifacts[f"{stem}_{extension}"] = path

    if finite_difference_diagnostic is not None:
        diagnostic_paths = plot_matched_gradient_oracle(
            finite_difference_diagnostic,
            output,
            metadata=frame.attrs["matched_metadata"],
        )
        for extension, diagnostic_path in diagnostic_paths.items():
            artifacts[f"projected_gradient_oracle_{extension}"] = diagnostic_path
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render figures from one validated corrected matched-LoRA benchmark."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="matched benchmark artifact root (or its raw directory)",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for figures and summary.csv")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"bootstrap resamples per estimate (default: {DEFAULT_BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"deterministic bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED})",
    )
    parser.add_argument(
        "--locked-test-results",
        type=Path,
        help="exact path to the one-shot locked-test results.json receipt",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="exact committed evaluation-split manifest required with locked results",
    )
    parser.add_argument(
        "--locked-source-receipt",
        type=Path,
        help="exact committed v2 locked source-index receipt required with locked results",
    )
    parser.add_argument(
        "--finite-difference-diagnostic",
        type=Path,
        help="exact diagnostic JSON to render (never auto-discovered)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m rl_no_backward.plotting``."""

    arguments = _build_parser().parse_args(argv)
    artifacts = plot_results(
        arguments.input_dir,
        arguments.output_dir,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.bootstrap_seed,
        locked_test_results=arguments.locked_test_results,
        evaluation_manifest=arguments.evaluation_manifest,
        locked_source_receipt=arguments.locked_source_receipt,
        finite_difference_diagnostic=arguments.finite_difference_diagnostic,
    )
    print(f"wrote {len(artifacts)} artifacts to {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
