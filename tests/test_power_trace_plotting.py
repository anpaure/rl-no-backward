from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rl_no_backward.power_trace_plotting import (
    load_power_trace_pair,
    load_sidecapture_optimizer_trace,
    main,
    plot_power_trace_comparison,
    summarize_power_trace_pair,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_array(root: Path, relative: str, values: np.ndarray) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    return {
        "path": relative,
        "dtype": str(values.dtype),
        "shape": list(values.shape),
    }


def _write_sidecapture_store(
    root: Path,
    *,
    method: str,
    start_sample: int,
    end_sample: int,
    nvml_power_w: float | None,
    nvml_pre_seconds: float = 1.1,
) -> tuple[np.ndarray, int, int]:
    root.mkdir(parents=True)
    rate = 100_000.0
    values = (
        np.sin(np.linspace(0.0, 600.0, 2_000_000, dtype=np.float64))
        * (0.25 if method == "bp_grpo" else 0.40)
    ).astype(np.float32)
    primary = _write_array(
        root,
        "channels/power/shard_000000/capture_000000000.npy",
        values,
    )
    primary.update(sample_rate_hz=rate, unit="normalized_adc", metadata={"calibrated": False})
    host_start_ns = 10_000_000_000
    host_end_ns = host_start_ns + round((end_sample - start_sample) / rate * 1e9)
    channels: dict[str, object] = {"power": primary}
    if nvml_power_w is not None:
        timestamps = np.arange(
            host_start_ns - round(nvml_pre_seconds * 1e9),
            host_end_ns + 100_000_001,
            50_000_000,
            dtype=np.int64,
        )
        power = np.full(timestamps.shape, nvml_power_w, dtype=np.float32)
        power_descriptor = _write_array(
            root,
            "channels/nvml.power_w/shard_000000/capture_000000000.npy",
            power,
        )
        timestamp_descriptor = _write_array(
            root,
            "timestamps/nvml.power_w/shard_000000/capture_000000000.npy",
            timestamps,
        )
        power_descriptor.update(
            sample_rate_hz=20.0,
            unit="W",
            metadata={"clock_domain": "host_monotonic"},
            timestamps_path=timestamp_descriptor["path"],
            timestamp_clock="host_monotonic_ns",
        )
        channels["nvml.power_w"] = power_descriptor

    annotations = [
        {
            "name": f"{method}.optimizer.host",
            "kind": "interval",
            "clock_domain": "host_monotonic",
            "start": host_start_ns,
            "end": host_end_ns,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "sync": "none",
            "metadata": {},
            "mapping": {
                "method": "host_monotonic_trigger_delta",
                "trigger_source": "chipwhisperer.force",
            },
        },
        {
            "name": f"{method}.optimizer.cuda",
            "kind": "interval",
            "clock_domain": "cuda",
            "start": 0.0,
            "end": (host_end_ns - host_start_ns) / 1e6,
            "start_sample": start_sample + 1,
            "end_sample": end_sample - 1,
            "sync": "both",
            "metadata": {},
            "mapping": {
                "method": "host_trigger_plus_cuda_origin_enqueue_plus_cuda_elapsed",
                "trigger_source": "chipwhisperer.force",
            },
        },
    ]
    annotation_path = "annotations/shard_000000/capture_000000000.json"
    _write_json(root / annotation_path, annotations)
    resolved = {
        "mode": "stream",
        "sample_rate_hz": rate,
        "samples": values.size,
        "duration_s": values.size / rate,
        "pretrigger_samples": 0,
        "raw_sample_rate_hz": 150_000_000.0,
        "decimation": 1_500,
        "bits_per_sample": 12,
        "warnings": [],
        "details": {
            "trigger": "force",
            "usb_read_mode": "auto",
            "gain_db": 10.0,
            "max_stream_rate_hz": 10_000_000.0,
            "stream_segment_size": 65_536,
            "stream_fast_fifo": False,
            "stream_arm_settle_s": 0.001,
        },
    }
    sampler_metadata = {
        "backend": "composite",
        "primary": {
            "backend": "chipwhisperer",
            "resolved": resolved,
            "streaming_controls": {
                "requested_mode": "stream",
                "resolved_mode": "stream",
                "adc_stream_mode_readback": True,
                "max_stream_rate_hz_configured": 10_000_000.0,
                "stream_segment_size_configured": 65_536,
                "stream_segment_size_readback": 65_536,
                "stream_fast_fifo_requested": False,
                "stream_arm_settle_s_requested": 0.001,
                "requested_trigger_mode": "auto",
                "actual_trigger_mode_readback": "force",
            },
        },
        "auxiliaries": ([{"backend": "nvml"}] if nvml_power_w is not None else []),
    }
    record = {
        "schema_version": "sidecapture.dataset/v1",
        "index": 0,
        "attempt": 1,
        "primary_channel": "power",
        "channels": channels,
        "annotations": {"path": annotation_path, "count": len(annotations)},
        "labels": {"method": method, "seed": 0, "step": 1},
        "health": {"ok": True, "issues": [], "metrics": {}},
        "trigger": {
            "host_monotonic_ns": 0,
            "source": "chipwhisperer.force",
            "metadata": {"mode": "force"},
        },
        "batch_metadata": {
            "primary": {
                "adc_errors": 0,
                "scope_info": resolved,
                "stream_fast_fifo": False,
                "stream_arm_settle_s": 0.001,
                "normal_fifo_enforced_before_trigger": True,
                "slow_fifo_trigger_executed": True,
            },
            "auxiliaries": {},
        },
        "sampler_metadata": sampler_metadata,
    }
    _write_json(root / "records/shard_000000/capture_000000000.json", record)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "sidecapture.dataset/v1",
            "experiment": {
                "request": {
                    "duration_s": values.size / rate,
                    "sample_rate_hz": rate,
                    "pretrigger_s": 0.0,
                    "mode": "stream",
                    "bits_per_sample": 12,
                    "gain_db": 10.0,
                    "channel": "power",
                },
                "resolved": resolved,
                "sampler": sampler_metadata,
            },
            "identity_hash": method,
        },
    )
    return values[start_sample:end_sample], host_start_ns, host_end_ns


def test_loads_exact_host_crop_and_computes_cw_and_nvml_metrics(tmp_path: Path) -> None:
    bp_expected, _, _ = _write_sidecapture_store(
        tmp_path / "bp",
        method="bp_grpo",
        start_sample=100_000,
        end_sample=130_000,
        nvml_power_w=100.0,
    )
    _write_sidecapture_store(
        tmp_path / "fo",
        method="fo_npg",
        start_sample=100_000,
        end_sample=160_000,
        nvml_power_w=200.0,
    )

    bp, fo = load_power_trace_pair(tmp_path / "bp", tmp_path / "fo")
    summary = summarize_power_trace_pair(tmp_path / "bp", tmp_path / "fo").set_index("method")

    assert np.array_equal(bp.adc_values, bp_expected)
    assert bp.adc_baseline_values is not None
    assert bp.adc_baseline_values.size == 100_000
    assert bp.adc_baseline_duration_seconds == pytest.approx(1.0)
    assert bp.host_duration_seconds == pytest.approx(0.3)
    assert fo.host_duration_seconds == pytest.approx(0.6)
    assert summary.loc["bp_grpo", "cw_rms"] == pytest.approx(
        np.sqrt(np.mean(np.square(bp_expected.astype(np.float64))))
    )
    assert bp.nvml is not None
    assert bp.nvml.mean_power_w == pytest.approx(100.0)
    assert bp.nvml.peak_power_w == pytest.approx(100.0)
    assert bp.nvml.energy_j == pytest.approx(30.0)
    assert bp.nvml.baseline_mean_power_w == pytest.approx(100.0)
    assert bp.nvml.baseline_adjusted_dynamic_energy_j == pytest.approx(0.0)
    assert bp.nvml.relative_time_seconds[0] == pytest.approx(0.0)
    assert bp.nvml.relative_time_seconds[-1] == pytest.approx(0.3)
    assert np.allclose(bp.nvml.power_w, 100.0)
    assert fo.nvml is not None
    assert fo.nvml.energy_j == pytest.approx(120.0)
    assert summary.loc["bp_grpo", "cw_preoptimizer_baseline_available"]
    assert summary.loc["bp_grpo", "cw_preoptimizer_baseline_duration_seconds"] == pytest.approx(1.0)
    assert np.isfinite(summary.loc["bp_grpo", "cw_optimizer_to_prebaseline_ac_rms_ratio"])
    assert summary.loc["bp_grpo", "nvml_baseline_adjusted_dynamic_energy_j"] == pytest.approx(0.0)
    assert set(summary["uncertainty_status"]) == {"single_capture_no_interval"}


def test_plot_writes_honest_n1_png_pdf_and_csv(tmp_path: Path) -> None:
    _write_sidecapture_store(
        tmp_path / "bp",
        method="bp_grpo",
        start_sample=100_000,
        end_sample=130_000,
        nvml_power_w=100.0,
    )
    _write_sidecapture_store(
        tmp_path / "fo",
        method="fo_npg",
        start_sample=100_000,
        end_sample=160_000,
        nvml_power_w=200.0,
    )

    artifacts = plot_power_trace_comparison(tmp_path / "bp", tmp_path / "fo", tmp_path / "report")

    assert set(artifacts) == {"summary_csv", "comparison_png", "comparison_pdf"}
    assert artifacts["summary_csv"].exists() and artifacts["summary_csv"].stat().st_size > 100
    assert artifacts["comparison_png"].stat().st_size > 1_000
    assert artifacts["comparison_pdf"].stat().st_size > 1_000
    image = plt.imread(artifacts["comparison_png"])
    assert image.ndim == 3
    assert min(image.shape[:2]) > 500
    summary = pd.read_csv(artifacts["summary_csv"])
    assert list(summary["method"]) == ["bp_grpo", "fo_npg"]
    assert summary.filter(regex="(?i)ci|confidence").empty
    assert set(summary["uncertainty_status"]) == {"single_capture_no_interval"}
    assert set(summary["cw_unit"]) == {"normalized_adc"}
    assert set(summary["cw_capture_mode"]) == {"stream"}
    assert summary["cw_preoptimizer_baseline_available"].all()
    assert set(summary["cw_raw_summary_scope"]) == {
        "exact host optimizer interval; raw normalized_adc"
    }


def test_pair_without_nvml_reports_no_absolute_power_or_energy(tmp_path: Path) -> None:
    _write_sidecapture_store(
        tmp_path / "bp",
        method="bp_grpo",
        start_sample=100_000,
        end_sample=130_000,
        nvml_power_w=None,
    )
    _write_sidecapture_store(
        tmp_path / "fo",
        method="fo_npg",
        start_sample=100_000,
        end_sample=160_000,
        nvml_power_w=None,
    )

    summary = summarize_power_trace_pair(tmp_path / "bp", tmp_path / "fo")

    assert summary["nvml_integrated_energy_j"].isna().all()
    assert summary["nvml_mean_power_w"].isna().all()
    assert set(summary["nvml_sample_count"]) == {0}


def test_nvml_baseline_is_optional_and_never_extrapolated(tmp_path: Path) -> None:
    for method, end_sample, watts in (
        ("bp_grpo", 130_000, 100.0),
        ("fo_npg", 160_000, 200.0),
    ):
        _write_sidecapture_store(
            tmp_path / method,
            method=method,
            start_sample=100_000,
            end_sample=end_sample,
            nvml_power_w=watts,
            nvml_pre_seconds=0.1,
        )

    bp, fo = load_power_trace_pair(tmp_path / "bp_grpo", tmp_path / "fo_npg")
    summary = pd.DataFrame.from_records([bp.summary_row(), fo.summary_row()])

    assert bp.nvml is not None and bp.nvml.energy_j == pytest.approx(30.0)
    assert fo.nvml is not None and fo.nvml.energy_j == pytest.approx(120.0)
    assert bp.nvml.baseline_mean_power_w is None
    assert bp.nvml.baseline_adjusted_dynamic_energy_j is None
    assert summary["nvml_preoptimizer_baseline_mean_power_w"].isna().all()
    assert summary["nvml_baseline_adjusted_dynamic_energy_j"].isna().all()


def test_fail_closed_on_wrong_schema_annotation_or_asymmetric_nvml(tmp_path: Path) -> None:
    _write_sidecapture_store(
        tmp_path / "bp",
        method="bp_grpo",
        start_sample=100_000,
        end_sample=130_000,
        nvml_power_w=100.0,
    )
    _write_sidecapture_store(
        tmp_path / "fo",
        method="fo_npg",
        start_sample=100_000,
        end_sample=160_000,
        nvml_power_w=None,
    )
    with pytest.raises(ValueError, match="present for both captures or neither"):
        load_power_trace_pair(tmp_path / "bp", tmp_path / "fo")

    manifest_path = tmp_path / "bp" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "sidecapture.dataset/v0"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="sidecapture.dataset/v1"):
        load_sidecapture_optimizer_trace(tmp_path / "bp", expected_method="bp_grpo")

    manifest["schema_version"] = "sidecapture.dataset/v1"
    manifest["experiment"]["request"]["mode"] = "burst"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="locked 20 s/100 kHz stream plan"):
        load_sidecapture_optimizer_trace(tmp_path / "bp", expected_method="bp_grpo")

    manifest["experiment"]["request"]["mode"] = "stream"
    _write_json(manifest_path, manifest)
    annotation_path = tmp_path / "bp" / "annotations/shard_000000/capture_000000000.json"
    annotations = json.loads(annotation_path.read_text())
    annotations[0]["name"] = "bp_grpo.optimizer.wrong"
    _write_json(annotation_path, annotations)
    with pytest.raises(ValueError, match="bp_grpo.optimizer.host"):
        load_sidecapture_optimizer_trace(tmp_path / "bp", expected_method="bp_grpo")


def test_cli_accepts_exact_two_stores(tmp_path: Path) -> None:
    _write_sidecapture_store(
        tmp_path / "bp",
        method="bp_grpo",
        start_sample=100_000,
        end_sample=130_000,
        nvml_power_w=None,
    )
    _write_sidecapture_store(
        tmp_path / "fo",
        method="fo_npg",
        start_sample=100_000,
        end_sample=160_000,
        nvml_power_w=None,
    )

    assert (
        main(
            [
                "--bp-store",
                str(tmp_path / "bp"),
                "--fo-store",
                str(tmp_path / "fo"),
                "--output",
                str(tmp_path / "report"),
            ]
        )
        == 0
    )
    assert (tmp_path / "report" / "power_trace_summary.csv").exists()
