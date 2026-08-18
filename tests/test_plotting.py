from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rl_no_backward.plotting import (
    aggregate_metrics,
    bootstrap_mean_ci,
    find_gsm8k_fd_diagnostic,
    load_results,
    main,
    plot_finite_difference_diagnostics,
    plot_results,
    summarize_results,
)


def _write_synthetic_runs(root: Path) -> None:
    for method_index, method in enumerate(("bp_grpo", "fo_npg")):
        for seed in range(3):
            path = root / "raw" / method / f"{method}_seed{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, object]] = []
            for step in range(3):
                samples = step * 128
                wall_time = step * (3.0 + 1.5 * method_index) + seed * 0.05
                accuracy = 0.22 + 0.12 * step - 0.025 * method_index + 0.01 * seed
                expected_reward = accuracy + 0.08
                if step:
                    records.append(
                        {
                            "kind": "train_step",
                            "method": method,
                            "seed": seed,
                            "step": step,
                            # These are deliberately per-step values; the loader
                            # must prefer the cumulative fields below.
                            "environment_samples": 128,
                            "forward_calls": 7 + 4 * method_index,
                            "backward_calls": 2 if method == "bp_grpo" else 0,
                            "cumulative_environment_samples": samples,
                            "cumulative_forward_calls": step * (7 + 4 * method_index),
                            "cumulative_backward_calls": step * (2 if method == "bp_grpo" else 0),
                            "cumulative_teacher_forced_examples": step * (400 + 120 * method_index),
                            "wall_time_seconds": wall_time,
                            "peak_gpu_memory_bytes": (3 + method_index) * 2**30,
                            "empirical_kl": 0.005 * step * (method_index + 1),
                            "zero_advantage_fraction": 0.25 / step,
                            "accepted": not (method == "fo_npg" and seed == 2 and step == 1),
                        }
                    )
                if method == "bp_grpo":
                    evaluation: dict[str, object] = {
                        "kind": "evaluation",
                        "method": method,
                        "seed": seed,
                        "step": step,
                        "environment_samples": samples,
                        "wall_time_seconds": wall_time,
                        "forward_calls": step * 7,
                        "backward_calls": step * 2,
                        "teacher_forced_examples": step * 400,
                        "peak_gpu_memory_bytes": 3 * 2**30,
                        "test_accuracy": accuracy,
                        "test_expected_reward": expected_reward,
                    }
                else:
                    # Nested future-facing GSM8K/W&B-style names exercise the
                    # normalisation of slashes and namespaces.
                    evaluation = {
                        "record_type": "eval",
                        "config": {"method": method, "seed": seed},
                        "optimizer_step": step,
                        "progress": {
                            "environment_samples": samples,
                            "wall_time_seconds": wall_time,
                            "forward_calls": step * 11,
                            "backward_calls": 0,
                            "teacher_forced_examples": step * 520,
                            "peak_gpu_memory_bytes": 4 * 2**30,
                        },
                        "eval": {
                            "gsm8k": {"exact_match": accuracy},
                            "expected_reward": expected_reward,
                        },
                    }
                records.append(evaluation)
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")


def _write_validation_and_test_run(root: Path) -> None:
    path = root / "raw" / "bp_grpo_seed0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 0,
            "environment_samples": 0,
            "wall_time_seconds": 0.0,
            "val_accuracy": 0.2,
            "val_exact_reward": 0.2,
            "val_shaped_reward": 0.3,
        },
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 10,
            "environment_samples": 100,
            "wall_time_seconds": 5.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "val_accuracy": 0.4,
            "val_exact_reward": 0.4,
            "val_shaped_reward": 0.5,
        },
        {
            "kind": "evaluation",
            "split": "test",
            "method": "bp_grpo",
            "seed": 0,
            "step": 10,
            "environment_samples": 100,
            "wall_time_seconds": 5.0,
            "test_accuracy": 0.9,
            "test_exact_reward": 0.9,
            "test_shaped_reward": 0.95,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_bootstrap_is_deterministic_and_ignores_nonfinite_values() -> None:
    first = bootstrap_mean_ci([0.1, 0.3, 0.5, np.nan], samples=500, seed=17)
    second = bootstrap_mean_ci([0.1, 0.3, 0.5, np.inf], samples=500, seed=17)

    assert first == second
    assert first.mean == pytest.approx(0.3)
    assert first.ci_low <= first.mean <= first.ci_high
    assert first.count == 3


def test_recursive_loader_normalises_controlled_and_gsm8k_metrics(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    # An interrupted final append is tolerated without discarding prior rows.
    damaged = tmp_path / "raw" / "fo_npg" / "fo_npg_seed0.jsonl"
    with damaged.open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":')

    with pytest.warns(UserWarning, match="skipping invalid JSON"):
        frame = load_results(tmp_path)

    assert set(frame["method"]) == {"bp_grpo", "fo_npg"}
    evaluations = frame[frame["kind"] == "evaluation"]
    assert len(evaluations) == 18
    assert evaluations["accuracy"].notna().all()
    assert evaluations["expected_reward"].notna().all()
    assert any("gsm8k_exact_match" in source for source in evaluations["score_source"])

    final_forward = frame[
        (frame["method"] == "fo_npg") & (frame["kind"] == "train_step") & (frame["step"] == 2)
    ]
    assert set(final_forward["environment_samples"]) == {256.0}
    assert set(final_forward["forward_calls"]) == {22.0}


def test_loader_and_summary_preserve_phase_timing_and_allocator_peaks(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "fo_pg_seed0.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "kind": "train_step",
            "method": "fo_pg",
            "seed": 0,
            "step": 1,
            "environment_samples": 8,
            "wall_time_seconds": 5.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "rollout_and_old_score_seconds": 1.25,
            "optimizer_seconds": 2.5,
            "peak_gpu_memory_allocated_bytes": 3 * 2**30,
            "peak_gpu_memory_reserved_bytes": 4 * 2**30,
        },
        {
            "kind": "evaluation",
            "method": "fo_pg",
            "seed": 0,
            "step": 1,
            "environment_samples": 8,
            "wall_time_seconds": 7.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "evaluation_seconds": 2.0,
            "peak_gpu_memory_allocated_bytes": 3 * 2**30,
            "peak_gpu_memory_reserved_bytes": 4 * 2**30,
            "val_accuracy": 0.5,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    frame = load_results(tmp_path)
    train = frame[frame["kind"] == "train_step"].iloc[0]
    assert train["rollout_and_old_score_seconds"] == pytest.approx(1.25)
    assert train["optimizer_seconds"] == pytest.approx(2.5)
    assert train["full_prefix_calls"] == 1
    assert train["suffix_calls"] == 4
    assert train["peak_gpu_memory_bytes"] == 3 * 2**30
    assert train["peak_gpu_memory_allocated_bytes"] == 3 * 2**30
    assert train["peak_gpu_memory_reserved_bytes"] == 4 * 2**30

    summary = summarize_results(frame, bootstrap_samples=10)
    assert summary["total_rollout_and_old_score_seconds_mean"].item() == pytest.approx(1.25)
    assert summary["total_optimizer_seconds_mean"].item() == pytest.approx(2.5)
    assert summary["total_evaluation_seconds_mean"].item() == pytest.approx(2.0)
    assert summary["final_full_prefix_calls_mean"].item() == 1
    assert summary["final_suffix_calls_mean"].item() == 4
    assert summary["peak_gpu_memory_allocated_bytes_mean"].item() == 3 * 2**30
    assert summary["peak_gpu_memory_reserved_bytes_mean"].item() == 4 * 2**30


def test_aggregation_and_summary_have_bootstrap_ci_and_auc(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    frame = load_results(tmp_path)
    aggregate = aggregate_metrics(
        frame,
        metric="accuracy",
        by="environment_samples",
        bootstrap_samples=300,
        bootstrap_seed=9,
    )
    final = aggregate[aggregate["environment_samples"] == 256]

    assert set(final["count"]) == {3}
    assert (final["ci_low"] <= final["mean"]).all()
    assert (final["mean"] <= final["ci_high"]).all()

    summary = summarize_results(frame, bootstrap_samples=300, bootstrap_seed=9)
    assert list(summary["method"]) == ["bp_grpo", "fo_npg"]
    assert set(summary["runs"]) == {3}
    assert summary["final_accuracy_mean"].notna().all()
    assert summary["normalised_auc_environment_samples_mean"].between(0, 1).all()
    assert summary.loc[summary["method"] == "fo_npg", "final_backward_calls_mean"].item() == 0


def test_validation_curve_excludes_same_step_test_but_summary_prefers_test(
    tmp_path: Path,
) -> None:
    _write_validation_and_test_run(tmp_path)
    frame = load_results(tmp_path)

    evaluations = frame[frame["kind"] == "evaluation"]
    assert list(evaluations["evaluation_split"]) == ["validation", "validation", "test"]

    curve = aggregate_metrics(
        frame,
        metric="accuracy",
        by="environment_samples",
        bootstrap_samples=100,
    )
    final_curve = curve[curve["environment_samples"] == 100]
    assert len(final_curve) == 1
    assert final_curve["mean"].item() == pytest.approx(0.4)

    summary = summarize_results(frame, bootstrap_samples=100)
    assert summary["final_accuracy_mean"].item() == pytest.approx(0.9)
    assert summary["final_expected_reward_mean"].item() == pytest.approx(0.95)
    assert summary["primary_metric_sources"].item() == "test_accuracy"
    assert summary["normalised_auc_environment_samples_mean"].item() == pytest.approx(0.3)


def test_plot_results_writes_readable_png_pdf_and_summary(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    output = tmp_path / "report"
    artifacts = plot_results(
        tmp_path / "raw",
        output,
        bootstrap_samples=200,
        bootstrap_seed=42,
    )

    expected_stems = {
        "learning_curves_samples",
        "learning_curves_wall_time",
        "final_performance",
        "compute_memory_tradeoffs",
        "optimizer_diagnostics",
    }
    assert artifacts["summary_csv"] == output / "summary.csv"
    for stem in expected_stems:
        for extension in ("png", "pdf"):
            path = artifacts[f"{stem}_{extension}"]
            assert path.exists()
            assert path.stat().st_size > 1_000
    image = plt.imread(artifacts["learning_curves_samples_png"])
    assert image.ndim == 3
    assert min(image.shape[:2]) > 400

    summary = pd.read_csv(output / "summary.csv")
    assert list(summary["method"]) == ["bp_grpo", "fo_npg"]
    assert "auc_environment_samples_mean" in summary
    assert "peak_gpu_memory_bytes_mean" in summary
    assert "peak_gpu_memory_allocated_bytes_mean" in summary
    assert "peak_gpu_memory_reserved_bytes_mean" in summary


def test_cli_main_accepts_bootstrap_controls(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    output = tmp_path / "cli-report"

    assert (
        main(
            [
                str(tmp_path / "raw"),
                str(output),
                "--bootstrap-samples",
                "50",
                "--bootstrap-seed",
                "3",
            ]
        )
        == 0
    )
    assert (output / "summary.csv").exists()


def _write_fd_diagnostic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic": "real_model_gsm8k_projected_finite_difference",
        "finite_difference": {
            "results": [
                {
                    "mu": mu,
                    "agreement_with_exact_projected_gradient": {
                        "cosine_similarity": cosine,
                        "relative_l2_error": relative_error,
                    },
                    "fisher": {"legacy_to_correct_trace_ratio": ratio},
                }
                for mu, cosine, relative_error, ratio in (
                    (0.25, 0.979, 0.236, 0.019),
                    (0.5, 0.886, 0.504, 0.021),
                    (1.0, 0.992, 0.131, 0.023),
                    (2.0, 0.999, 0.036, 0.022),
                )
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finite_difference_diagnostic_plot_is_public_and_readable(tmp_path: Path) -> None:
    diagnostic = tmp_path / "gsm8k_fd.json"
    _write_fd_diagnostic(diagnostic)

    paths = plot_finite_difference_diagnostics(diagnostic, tmp_path / "figures")

    assert set(paths) == {"png", "pdf"}
    assert all(path.exists() and path.stat().st_size > 1_000 for path in paths.values())
    image = plt.imread(paths["png"])
    assert image.ndim == 3
    assert min(image.shape[:2]) > 400


def test_plot_results_discovers_sibling_gsm8k_diagnostic(tmp_path: Path) -> None:
    final_run = tmp_path / "artifacts" / "final_gsm8k"
    _write_synthetic_runs(final_run)
    diagnostic = tmp_path / "artifacts" / "diagnostics" / "gsm8k_fd.json"
    _write_fd_diagnostic(diagnostic)

    assert find_gsm8k_fd_diagnostic(final_run) == diagnostic
    artifacts = plot_results(
        final_run,
        tmp_path / "report",
        bootstrap_samples=30,
        bootstrap_seed=4,
    )

    assert artifacts["fd_diagnostics_png"].exists()
    assert artifacts["fd_diagnostics_pdf"].exists()
