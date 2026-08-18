"""Publication-quality plots and summaries for RL benchmark JSONL files.

The experiment runner deliberately writes append-only JSONL rather than a
plotting-specific table.  This module is the small compatibility layer between
those raw records and the final report.  It accepts both the controlled-task
metric names used by :mod:`rl_no_backward.experiment` and common GSM8K/W&B
spellings (including nested dictionaries and slash-separated names).

Examples
--------
Generate all report figures from a benchmark directory::

    python -m rl_no_backward.plotting runs/pilot artifacts/pilot

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

if not os.environ.get("DISPLAY"):
    plt.switch_backend("Agg")


DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 2026

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
    "base": "Base model",
    "bp_grpo": "Backprop GRPO",
    "grpo": "Backprop GRPO",
    "standard_grpo": "Backprop GRPO",
    "fo_pg": "Forward-only PG",
    "fo_npg": "Forward-only NPG",
    "focus_npg": "History-subspace NPG",
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
    "focus_npg": 4,
    "es": 5,
}

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
    state.  A singleton has a zero-width interval, which is honest and keeps
    smoke-test plots usable without pretending that one run measures variance.
    """

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return BootstrapEstimate(float("nan"), float("nan"), float("nan"), 0)
    mean = float(array.mean())
    if array.size == 1:
        return BootstrapEstimate(mean, mean, mean, 1)
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
            "forward_calls": _find_number(
                flat,
                (
                    "cumulative_forward_calls",
                    "total_forward_calls",
                    "forward_calls",
                    "model_forward_calls",
                ),
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
                    "peak_gpu_memory_bytes",
                    "max_gpu_memory_bytes",
                    "peak_memory_bytes",
                    "cuda_peak_memory_bytes",
                ),
            ),
            "accuracy": accuracy,
            "expected_reward": expected_reward,
            "score": accuracy if math.isfinite(accuracy) else expected_reward,
            "score_source": accuracy_source or reward_source,
            "empirical_kl": _find_number(
                flat,
                ("empirical_kl", "approx_kl", "policy_kl", "observed_kl", "kl"),
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


def load_results(input_dir: str | Path) -> pd.DataFrame:
    """Recursively load and canonicalise JSONL run records.

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
        raise ValueError(f"no valid JSON object records found beneath {root}")

    frame = pd.DataFrame.from_records(records)
    for column in (
        "step",
        "environment_samples",
        "wall_time_seconds",
        "forward_calls",
        "backward_calls",
        "teacher_forced_examples",
        "peak_gpu_memory_bytes",
        "accuracy",
        "expected_reward",
        "score",
        "empirical_kl",
        "zero_advantage_fraction",
        "acceptance_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(
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
        final_evaluations = (
            test_evaluations
            if not test_evaluations.empty
            else validation_evaluations
            if not validation_evaluations.empty
            else evaluations
        )
        training = group[group["kind"] == "train_step"]
        score_sources = [
            str(value)
            for value in final_evaluations.get("score_source", pd.Series(dtype=object)).dropna()
        ]
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": str(group["method"].iloc[0]),
            "seed": group["seed"].iloc[0],
            "score_source": Counter(score_sources).most_common(1)[0][0] if score_sources else "",
            "final_score": _last_finite(final_evaluations, "score"),
            "final_accuracy": _last_finite(final_evaluations, "accuracy"),
            "final_expected_reward": _last_finite(final_evaluations, "expected_reward"),
            "final_environment_samples": _last_finite(group, "environment_samples"),
            "final_wall_time_seconds": _last_finite(group, "wall_time_seconds"),
            "final_forward_calls": _last_finite(group, "forward_calls"),
            "final_backward_calls": _last_finite(group, "backward_calls"),
            "final_teacher_forced_examples": _last_finite(group, "teacher_forced_examples"),
            "peak_gpu_memory_bytes": float(
                pd.to_numeric(group["peak_gpu_memory_bytes"], errors="coerce").max()
            ),
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
        "final_expected_reward",
        "auc_environment_samples",
        "normalised_auc_environment_samples",
        "auc_wall_time_seconds",
        "normalised_auc_wall_time_seconds",
        "final_environment_samples",
        "final_wall_time_seconds",
        "final_forward_calls",
        "final_backward_calls",
        "final_teacher_forced_examples",
        "peak_gpu_memory_bytes",
        "acceptance_rate",
    )
    for method, group in run_summary.groupby("method", sort=False):
        sources = [source for source in group["score_source"] if source]
        row: dict[str, Any] = {
            "method": method,
            "display_name": _display_name(method),
            "runs": len(group),
            "seeds": int(group["seed"].nunique()),
            "primary_metric_sources": ";".join(sorted(set(sources))),
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


def _ordered_methods(methods: Iterable[str]) -> list[str]:
    unique = {str(method) for method in methods}
    return sorted(unique, key=lambda method: (METHOD_PRIORITY.get(method, 100), method))


def _display_name(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _method_styles(methods: Iterable[str]) -> dict[str, tuple[str, str]]:
    ordered = _ordered_methods(methods)
    styles: dict[str, tuple[str, str]] = {}
    for index, method in enumerate(ordered):
        if method == "base":
            styles[method] = ("#6B7280", "o")
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
    ax.set_title(title, loc="left", fontweight="bold")
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
    if "final_accuracy_mean" in summary and summary["final_accuracy_mean"].notna().any():
        metrics.append(("final_accuracy", "Exact-match accuracy"))
    elif (
        "final_expected_reward_mean" in summary
        and summary["final_expected_reward_mean"].notna().any()
    ):
        metrics.append(("final_expected_reward", "Expected reward"))
    if not metrics:
        metrics.append(("final_score", "Final score"))

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.2 * len(metrics), max(3.4, 1.0 + 0.55 * len(summary))),
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
            mean = float(row[f"{metric}_mean"])
            low = float(row[f"{metric}_ci_low"])
            high = float(row[f"{metric}_ci_high"])
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
        ax.set_xlabel("Final evaluation")
        ax.set_yticks(positions, [_display_name(method) for method in methods])
        _x_score_axis(ax, values)
    fig.suptitle("Final performance · mean and 95% bootstrap CI", fontweight="bold")
    return _save_figure(fig, output_dir, "final_performance")


def _plot_compute_memory(
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.25), sharey=True)
    methods = _ordered_methods(summary["method"])
    styles = _method_styles(methods)
    indexed = summary.set_index("method")
    specifications = (
        (
            "final_forward_calls_mean",
            "Teacher-forced microbatch calls\n(rollout/eval excluded)",
            1.0,
        ),
        ("peak_gpu_memory_bytes_mean", "Peak allocated GPU memory (GiB)", 2**30),
    )
    score_values: list[float] = []
    for ax, (x_column, x_label, divisor) in zip(axes, specifications, strict=True):
        x_values: list[float] = []
        plotted = False
        for method in methods:
            row = indexed.loc[method]
            x = float(row.get(x_column, float("nan"))) / divisor
            y = float(row.get("final_score_mean", float("nan")))
            low = float(row.get("final_score_ci_low", y))
            high = float(row.get("final_score_ci_high", y))
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
            "Compute efficiency" if "forward" in x_column else "Memory efficiency",
            loc="left",
            fontweight="bold",
        )
        if "forward" in x_column:
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
    _score_axis(axes[0], score_values, "Final evaluation score")
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
    fig.suptitle("Performance–resource trade-offs", fontweight="bold")
    return _save_figure(fig, output_dir, "compute_memory_tradeoffs")


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
    ax.set_title("Model-call accounting", loc="left", fontweight="bold")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cumulative calls")
    ax.yaxis.set_major_formatter(EngFormatter(sep=""))
    if plotted:
        call_legend = ax.legend(
            handles=[
                Line2D([0], [0], color="#374151", linestyle="-", label="Forward"),
                Line2D([0], [0], color="#374151", linestyle="--", label="Backward"),
            ],
            loc="upper left",
        )
        ax.add_artist(call_legend)
    else:
        _empty_panel(ax, "No model-call records")


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
    _plot_diagnostic_metric(
        axes[0, 0],
        frame,
        metric="empirical_kl",
        title="Empirical policy KL",
        ylabel="KL divergence",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    _plot_diagnostic_metric(
        axes[0, 1],
        frame,
        metric="zero_advantage_fraction",
        title="Zero-advantage groups",
        ylabel="Fraction of groups",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    _plot_diagnostic_metric(
        axes[1, 0],
        frame,
        metric="acceptance_rate",
        title="Update acceptance",
        ylabel="Acceptance rate",
        styles=styles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    _plot_call_diagnostics(
        axes[1, 1],
        frame,
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
    fig.suptitle("Optimizer health and backward-pass audit", fontweight="bold")
    return _save_figure(fig, output_dir, "optimizer_diagnostics")


def plot_results(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Load raw runs, write ``summary.csv``, and render every report figure.

    Each figure is emitted as both a high-resolution PNG for quick viewing and
    a vector PDF for the report.  Returned keys include the extension (for
    example ``"learning_curves_samples_png"``).
    """

    frame = load_results(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(
        frame,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8g")

    artifacts: dict[str, Path] = {"summary_csv": summary_path}
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

    diagnostic_path = find_gsm8k_fd_diagnostic(input_dir)
    if diagnostic_path is not None:
        try:
            diagnostic_paths = plot_finite_difference_diagnostics(
                diagnostic_path,
                output,
            )
        except (TypeError, ValueError) as error:
            warnings.warn(
                f"skipping invalid finite-difference diagnostic: {error}",
                UserWarning,
                stacklevel=2,
            )
        else:
            for extension, path in diagnostic_paths.items():
                artifacts[f"fd_diagnostics_{extension}"] = path
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render RL benchmark figures from recursive JSONL run files."
    )
    parser.add_argument("input_dir", type=Path, help="Run file or directory containing JSONL")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m rl_no_backward.plotting``."""

    arguments = _build_parser().parse_args(argv)
    artifacts = plot_results(
        arguments.input_dir,
        arguments.output_dir,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(f"wrote {len(artifacts)} artifacts to {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
