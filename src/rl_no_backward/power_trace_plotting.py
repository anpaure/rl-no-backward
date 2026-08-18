"""Fail-closed analysis for one BP and one FO SideCapture optimizer trace.

The ChipWhisperer channel is an uncalibrated ``normalized_adc`` signal.  This
module therefore reports amplitude statistics only for that channel.  Absolute
power and energy are computed only when the store also contains timestamped
``nvml.power_w`` telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if not os.environ.get("DISPLAY"):
    plt.switch_backend("Agg")


SIDECAPTURE_SCHEMA = "sidecapture.dataset/v1"
METHODS = ("bp_grpo", "fo_npg")
METHOD_LABELS = {"bp_grpo": "BP-GRPO", "fo_npg": "FO-NPG"}
METHOD_COLOURS = {"bp_grpo": "#D55E00", "fo_npg": "#009E73"}
UNCERTAINTY_STATUS = "single_capture_no_interval"
SLIDING_RMS_WINDOW_SECONDS = 0.050
PRE_OPTIMIZER_BASELINE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class NVMLSummary:
    sample_count: int
    mean_power_w: float
    peak_power_w: float
    energy_j: float
    relative_time_seconds: np.ndarray
    power_w: np.ndarray
    baseline_mean_power_w: float | None
    baseline_adjusted_dynamic_energy_j: float | None


@dataclass(frozen=True, slots=True)
class OptimizerPowerTrace:
    method: str
    store: Path
    adc_values: np.ndarray
    adc_baseline_values: np.ndarray | None
    adc_sample_rate_hz: float
    host_start_ns: int
    host_end_ns: int
    host_duration_seconds: float
    cuda_start_sample: int
    cuda_end_sample: int
    nvml: NVMLSummary | None

    @property
    def adc_duration_seconds(self) -> float:
        return self.adc_values.size / self.adc_sample_rate_hz

    @property
    def adc_baseline_duration_seconds(self) -> float:
        if self.adc_baseline_values is None:
            return 0.0
        return self.adc_baseline_values.size / self.adc_sample_rate_hz

    def _cw_ac_metrics(self) -> tuple[float, float, float]:
        """Return reference mean, optimizer AC RMS, and baseline-relative ratio."""

        reference = self.adc_values
        if self.adc_baseline_values is not None:
            reference = self.adc_baseline_values
        reference_mean = float(np.mean(reference, dtype=np.float64))
        optimizer_ac_rms = float(
            np.sqrt(
                np.mean(
                    np.square(self.adc_values - reference_mean, dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )
        if self.adc_baseline_values is None:
            return reference_mean, optimizer_ac_rms, float("nan")
        baseline_ac_rms = float(
            np.sqrt(
                np.mean(
                    np.square(self.adc_baseline_values - reference_mean, dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )
        ratio = optimizer_ac_rms / baseline_ac_rms if baseline_ac_rms > 0.0 else float("nan")
        return reference_mean, optimizer_ac_rms, ratio

    def summary_row(self) -> dict[str, Any]:
        quantiles = np.quantile(self.adc_values, [0.05, 0.25, 0.50, 0.75, 0.95])
        nvml = self.nvml
        reference_mean, optimizer_ac_rms, baseline_ratio = self._cw_ac_metrics()
        baseline_ac_rms = float("nan")
        if self.adc_baseline_values is not None:
            baseline_ac_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            self.adc_baseline_values - reference_mean,
                            dtype=np.float64,
                        ),
                        dtype=np.float64,
                    )
                )
            )
        return {
            "method": self.method,
            "display_name": METHOD_LABELS[self.method],
            "capture_count": 1,
            "uncertainty_status": UNCERTAINTY_STATUS,
            "uncertainty_description": (
                "descriptive point only (n=1 capture); no repeat-based uncertainty interval"
            ),
            "optimizer_host_duration_seconds": self.host_duration_seconds,
            "optimizer_adc_duration_seconds": self.adc_duration_seconds,
            "cw_sample_count": int(self.adc_values.size),
            "cw_sample_rate_hz": self.adc_sample_rate_hz,
            "cw_unit": "normalized_adc",
            "cw_raw_summary_scope": "exact host optimizer interval; raw normalized_adc",
            "cw_rms": float(np.sqrt(np.mean(np.square(self.adc_values, dtype=np.float64)))),
            "cw_q05": float(quantiles[0]),
            "cw_q25": float(quantiles[1]),
            "cw_median": float(quantiles[2]),
            "cw_q75": float(quantiles[3]),
            "cw_q95": float(quantiles[4]),
            "cw_preoptimizer_baseline_available": self.adc_baseline_values is not None,
            "cw_preoptimizer_baseline_duration_seconds": self.adc_baseline_duration_seconds,
            "cw_ac_reference_mean": reference_mean,
            "cw_preoptimizer_baseline_ac_rms": baseline_ac_rms,
            "cw_optimizer_ac_rms": optimizer_ac_rms,
            "cw_optimizer_to_prebaseline_ac_rms_ratio": baseline_ratio,
            "nvml_sample_count": 0 if nvml is None else nvml.sample_count,
            "nvml_mean_power_w": float("nan") if nvml is None else nvml.mean_power_w,
            "nvml_peak_power_w": float("nan") if nvml is None else nvml.peak_power_w,
            "nvml_integrated_energy_j": float("nan") if nvml is None else nvml.energy_j,
            "nvml_preoptimizer_baseline_mean_power_w": (
                float("nan")
                if nvml is None or nvml.baseline_mean_power_w is None
                else nvml.baseline_mean_power_w
            ),
            "nvml_baseline_adjusted_dynamic_energy_j": (
                float("nan")
                if nvml is None or nvml.baseline_adjusted_dynamic_energy_j is None
                else nvml.baseline_adjusted_dynamic_energy_j
            ),
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} does not exist: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object: {path}")
    return dict(value)


def _contained_file(root: Path, relative: Any, *, description: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"{description} must be a safe relative path")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} escapes the SideCapture store") from error
    if not target.is_file():
        raise FileNotFoundError(f"{description} is missing: {target}")
    return target


def _load_array(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    description: str,
) -> np.ndarray:
    path = _contained_file(root, descriptor.get("path"), description=description)
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} is not a valid non-pickled NumPy array: {path}") from error
    expected_shape = descriptor.get("shape")
    if (
        values.ndim != 1
        or expected_shape != list(values.shape)
        or descriptor.get("dtype") != str(values.dtype)
        or values.size < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{description} differs from its descriptor or contains non-finite values")
    return np.asarray(values)


def _load_annotations(root: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptor = record.get("annotations")
    if not isinstance(descriptor, Mapping):
        raise TypeError("SideCapture record has no annotation descriptor")
    path = _contained_file(root, descriptor.get("path"), description="annotation payload")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"annotation payload is not valid UTF-8 JSON: {path}") from error
    if (
        not isinstance(value, list)
        or descriptor.get("count") != len(value)
        or any(not isinstance(annotation, Mapping) for annotation in value)
    ):
        raise ValueError("annotation payload differs from its record descriptor")
    return [dict(annotation) for annotation in value]


def _exact_annotation(
    annotations: Sequence[Mapping[str, Any]],
    *,
    name: str,
    clock_domain: str,
) -> dict[str, Any]:
    matches = [dict(annotation) for annotation in annotations if annotation.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} annotation; found {len(matches)}")
    annotation = matches[0]
    if annotation.get("kind") != "interval" or annotation.get("clock_domain") != clock_domain:
        raise ValueError(f"annotation {name!r} is not a {clock_domain} interval")
    return annotation


def _load_nvml_summary(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    start_ns: int,
    end_ns: int,
) -> NVMLSummary:
    if descriptor.get("unit") != "W" or descriptor.get("timestamp_clock") != "host_monotonic_ns":
        raise ValueError("nvml.power_w must use watts and host_monotonic_ns timestamps")
    power = np.asarray(_load_array(root, descriptor, description="NVML power channel"), dtype=float)
    timestamp_path = _contained_file(
        root,
        descriptor.get("timestamps_path"),
        description="NVML timestamp channel",
    )
    try:
        timestamps = np.load(timestamp_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("NVML timestamps are not a valid non-pickled NumPy array") from error
    if (
        timestamps.ndim != 1
        or timestamps.shape != power.shape
        or timestamps.dtype.kind not in {"i", "u"}
        or np.any(np.diff(timestamps.astype(np.int64)) <= 0)
    ):
        raise ValueError("NVML timestamps must be a strictly increasing integer vector")
    timestamps = timestamps.astype(np.int64)
    if timestamps[0] > start_ns or timestamps[-1] < end_ns:
        raise ValueError("NVML telemetry does not bracket the host optimizer interval")

    def integrate_window(
        window_start_ns: int, window_end_ns: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        internal = (timestamps > window_start_ns) & (timestamps < window_end_ns)
        integration_ns = np.concatenate(
            ([window_start_ns], timestamps[internal], [window_end_ns])
        ).astype(np.int64)
        integration_power = np.interp(integration_ns, timestamps, power)
        elapsed_seconds = (integration_ns - window_start_ns).astype(np.float64) / 1e9
        energy_j = float(np.trapezoid(integration_power, elapsed_seconds))
        return elapsed_seconds, integration_power, energy_j

    elapsed_seconds, integration_power, energy = integrate_window(start_ns, end_ns)
    duration = (end_ns - start_ns) / 1e9
    baseline_mean_power_w: float | None = None
    dynamic_energy_j: float | None = None
    baseline_start_ns = start_ns - round(PRE_OPTIMIZER_BASELINE_SECONDS * 1e9)
    baseline_raw_samples = int(
        np.count_nonzero((timestamps >= baseline_start_ns) & (timestamps <= start_ns))
    )
    # Never extrapolate a baseline. Requiring at least two actual samples in the
    # exact pre-window also avoids calling a single long interpolation span a
    # measured idle reference.
    if timestamps[0] <= baseline_start_ns and baseline_raw_samples >= 2:
        _, _, baseline_energy = integrate_window(baseline_start_ns, start_ns)
        baseline_mean_power_w = baseline_energy / PRE_OPTIMIZER_BASELINE_SECONDS
        dynamic_energy_j = energy - baseline_mean_power_w * duration
    return NVMLSummary(
        sample_count=int(integration_power.size),
        mean_power_w=energy / duration,
        peak_power_w=float(integration_power.max()),
        energy_j=energy,
        relative_time_seconds=elapsed_seconds,
        power_w=integration_power,
        baseline_mean_power_w=baseline_mean_power_w,
        baseline_adjusted_dynamic_energy_j=dynamic_energy_j,
    )


def load_sidecapture_optimizer_trace(
    store: str | Path,
    *,
    expected_method: str,
) -> OptimizerPowerTrace:
    """Load and crop one committed SideCapture v1 optimizer record."""

    if expected_method not in METHODS:
        raise ValueError(f"expected_method must be one of {METHODS}")
    root = Path(store).resolve()
    manifest = _read_mapping(root / "manifest.json", description="SideCapture manifest")
    if manifest.get("schema_version") != SIDECAPTURE_SCHEMA:
        raise ValueError(f"SideCapture manifest must use {SIDECAPTURE_SCHEMA}")
    record_paths = sorted((root / "records").rglob("capture_*.json"))
    if len(record_paths) != 1:
        raise ValueError(
            f"SideCapture store must contain exactly one committed record; found {len(record_paths)}"
        )
    record = _read_mapping(record_paths[0], description="SideCapture record")
    labels = record.get("labels")
    health = record.get("health")
    channels = record.get("channels")
    if (
        record.get("schema_version") != SIDECAPTURE_SCHEMA
        or record.get("index") != 0
        or not isinstance(labels, Mapping)
        or labels.get("method") != expected_method
        or not isinstance(health, Mapping)
        or health.get("ok") is not True
        or not isinstance(channels, Mapping)
    ):
        raise ValueError("SideCapture record identity, health, or schema is invalid")
    primary_name = record.get("primary_channel")
    primary = channels.get(primary_name)
    if not isinstance(primary_name, str) or not isinstance(primary, Mapping):
        raise TypeError("SideCapture record has no primary channel")
    sample_rate = primary.get("sample_rate_hz")
    primary_metadata = primary.get("metadata")
    if (
        primary.get("unit") != "normalized_adc"
        or not isinstance(primary_metadata, Mapping)
        or primary_metadata.get("calibrated") is not False
        or isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(float(sample_rate))
        or float(sample_rate) <= 0.0
    ):
        raise ValueError(
            "primary ChipWhisperer channel must be finite-rate, uncalibrated normalized_adc"
        )
    full_adc = np.asarray(
        _load_array(root, primary, description="primary ChipWhisperer channel"), dtype=float
    )
    annotations = _load_annotations(root, record)
    host = _exact_annotation(
        annotations,
        name=f"{expected_method}.optimizer.host",
        clock_domain="host_monotonic",
    )
    cuda = _exact_annotation(
        annotations,
        name=f"{expected_method}.optimizer.cuda",
        clock_domain="cuda",
    )
    mapping = host.get("mapping")
    start_ns, end_ns = host.get("start"), host.get("end")
    start_sample, end_sample = host.get("start_sample"), host.get("end_sample")
    if (
        not isinstance(mapping, Mapping)
        or mapping.get("method") != "host_monotonic_trigger_delta"
        or isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or end_ns <= start_ns
        or isinstance(start_sample, bool)
        or not isinstance(start_sample, int)
        or isinstance(end_sample, bool)
        or not isinstance(end_sample, int)
        or not 0 <= start_sample < end_sample <= full_adc.size
    ):
        raise ValueError("host optimizer annotation has invalid time/sample bounds or mapping")
    cuda_start, cuda_end = cuda.get("start_sample"), cuda.get("end_sample")
    if (
        isinstance(cuda_start, bool)
        or not isinstance(cuda_start, int)
        or isinstance(cuda_end, bool)
        or not isinstance(cuda_end, int)
        or not 0 <= cuda_start < cuda_end <= full_adc.size
    ):
        raise ValueError("CUDA optimizer annotation has invalid projected sample bounds")
    adc_values = full_adc[start_sample:end_sample]
    baseline_sample_count = round(float(sample_rate) * PRE_OPTIMIZER_BASELINE_SECONDS)
    adc_baseline_values: np.ndarray | None = None
    if baseline_sample_count >= 1 and start_sample >= baseline_sample_count:
        adc_baseline_values = full_adc[start_sample - baseline_sample_count : start_sample]
    host_duration = (end_ns - start_ns) / 1e9
    adc_duration = adc_values.size / float(sample_rate)
    if abs(host_duration - adc_duration) > 2.0 / float(sample_rate):
        raise ValueError("host optimizer duration and mapped ChipWhisperer crop disagree")

    nvml_descriptor = channels.get("nvml.power_w")
    nvml = None
    if nvml_descriptor is not None:
        if not isinstance(nvml_descriptor, Mapping):
            raise TypeError("nvml.power_w descriptor must be an object")
        nvml = _load_nvml_summary(
            root,
            nvml_descriptor,
            start_ns=start_ns,
            end_ns=end_ns,
        )
    return OptimizerPowerTrace(
        method=expected_method,
        store=root,
        adc_values=adc_values,
        adc_baseline_values=adc_baseline_values,
        adc_sample_rate_hz=float(sample_rate),
        host_start_ns=start_ns,
        host_end_ns=end_ns,
        host_duration_seconds=host_duration,
        cuda_start_sample=cuda_start,
        cuda_end_sample=cuda_end,
        nvml=nvml,
    )


def load_power_trace_pair(
    bp_store: str | Path,
    fo_store: str | Path,
) -> tuple[OptimizerPowerTrace, OptimizerPowerTrace]:
    """Load the exact BP/FO pair and enforce symmetric capture configuration."""

    bp = load_sidecapture_optimizer_trace(bp_store, expected_method="bp_grpo")
    fo = load_sidecapture_optimizer_trace(fo_store, expected_method="fo_npg")
    if not math.isclose(bp.adc_sample_rate_hz, fo.adc_sample_rate_hz, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("BP and FO ChipWhisperer sample rates differ")
    if (bp.nvml is None) != (fo.nvml is None):
        raise ValueError("NVML telemetry must be present for both captures or neither capture")
    if (bp.adc_baseline_values is None) != (fo.adc_baseline_values is None):
        raise ValueError(
            "a full 1 s pre-optimizer ChipWhisperer baseline must be present for both captures "
            "or neither capture"
        )
    return bp, fo


def summarize_power_trace_pair(
    bp_store: str | Path,
    fo_store: str | Path,
) -> pd.DataFrame:
    bp, fo = load_power_trace_pair(bp_store, fo_store)
    return pd.DataFrame.from_records([bp.summary_row(), fo.summary_row()])


def _display_envelope(
    values: np.ndarray, sample_rate: float, max_bins: int = 4_000
) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_bins:
        return np.arange(values.size, dtype=float) / sample_rate, values
    width = math.ceil(values.size / max_bins)
    x: list[float] = []
    y: list[float] = []
    for start in range(0, values.size, width):
        stop = min(start + width, values.size)
        chunk = values[start:stop]
        low = start + int(np.argmin(chunk))
        high = start + int(np.argmax(chunk))
        for index in sorted((low, high)):
            x.append(index / sample_rate)
            y.append(float(values[index]))
    return np.asarray(x), np.asarray(y)


def _sliding_rms(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    window_seconds: float = SLIDING_RMS_WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, round(sample_rate_hz * window_seconds))
    if window > values.size:
        raise ValueError("optimizer crop is shorter than the locked sliding-RMS window")
    squared = np.square(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    mean_square = (prefix[window:] - prefix[:-window]) / window
    mean_square = np.maximum(mean_square, 0.0)
    center_samples = np.arange(mean_square.size, dtype=float) + (window - 1) / 2.0
    return center_samples / sample_rate_hz, np.sqrt(mean_square)


def _ac_envelope(trace: OptimizerPowerTrace) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Build an aligned AC envelope and return its measured baseline RMS, if usable."""

    baseline = trace.adc_baseline_values
    reference = trace.adc_values if baseline is None else baseline
    reference_mean = float(np.mean(reference, dtype=np.float64))
    if baseline is None:
        values = trace.adc_values - reference_mean
        time_offset = 0.0
        baseline_rms = None
    else:
        values = np.concatenate((baseline, trace.adc_values)) - reference_mean
        time_offset = trace.adc_baseline_duration_seconds
        baseline_rms = float(
            np.sqrt(
                np.mean(
                    np.square(baseline - reference_mean, dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )
        if baseline_rms <= 0.0:
            baseline_rms = None
    envelope_time, envelope = _sliding_rms(values, trace.adc_sample_rate_hz)
    return envelope_time - time_offset, envelope, baseline_rms


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(
            path, bbox_inches="tight", facecolor="white", dpi=220 if extension == "png" else None
        )
        paths[extension] = path
    plt.close(fig)
    return paths


def plot_power_trace_comparison(
    bp_store: str | Path,
    fo_store: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render the honest n=1 BP/FO comparison and write its metric table."""

    bp, fo = load_power_trace_pair(bp_store, fo_store)
    traces = (bp, fo)
    summary = pd.DataFrame.from_records([trace.summary_row() for trace in traces])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "power_trace_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.9g", na_rep="")

    style = {
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.constrained_layout.use": True,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "grid.alpha": 0.20,
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(11.4, 9.2))
        grid = fig.add_gridspec(3, 2, height_ratios=(1.35, 0.78, 0.90))
        rms_ax = fig.add_subplot(grid[0, :])
        trace_axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
        amplitude_ax = fig.add_subplot(grid[2, 0])
        nvml_ax = fig.add_subplot(grid[2, 1])

        combined = np.concatenate([trace.adc_values for trace in traces])
        lower, upper = float(combined.min()), float(combined.max())
        padding = max((upper - lower) * 0.05, 1e-9)
        max_duration = max(trace.adc_duration_seconds for trace in traces)
        envelope_data = {trace.method: _ac_envelope(trace) for trace in traces}
        use_baseline_ratio = all(envelope_data[trace.method][2] is not None for trace in traces)
        for trace in traces:
            rms_time, rms, baseline_rms = envelope_data[trace.method]
            if use_baseline_ratio:
                assert baseline_rms is not None
                rms = rms / baseline_rms
            rms_ax.plot(
                rms_time,
                rms,
                color=METHOD_COLOURS[trace.method],
                linewidth=1.5,
                label=f"{METHOD_LABELS[trace.method]} ({trace.host_duration_seconds:.3f} s)",
            )
            rms_ax.axvline(
                trace.host_duration_seconds,
                color=METHOD_COLOURS[trace.method],
                linestyle=":",
                linewidth=1,
            )
        pre_duration = max(trace.adc_baseline_duration_seconds for trace in traces)
        if pre_duration > 0.0:
            rms_ax.axvspan(-pre_duration, 0.0, color="#E5E7EB", alpha=0.55, zorder=-1)
            rms_ax.axvline(0.0, color="#374151", linewidth=1.0)
            rms_ax.text(
                -pre_duration / 2.0,
                0.98,
                "1 s pre-optimizer baseline",
                transform=rms_ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color="#4B5563",
                fontsize=8,
            )
        rms_ax.set_xlim(-pre_duration, max_duration * 1.01)
        rms_ax.set_xlabel("Seconds from optimizer start · aligned t=0; no time warping")
        if use_baseline_ratio:
            rms_ax.axhline(1.0, color="#6B7280", linewidth=0.9, linestyle="--")
            rms_ax.set_ylabel("50 ms AC RMS / own 1 s pre-baseline RMS (×)")
            rms_title = "Aligned AC-coupled power-proxy envelope · relative to own baseline"
        else:
            rms_ax.set_ylabel("50 ms AC RMS (normalized ADC)")
            rms_title = "Aligned AC-coupled power-proxy envelope · no usable baseline ratio"
        rms_ax.set_title(rms_title, loc="left", fontweight="bold")
        rms_ax.legend(loc="best")
        for ax, trace in zip(trace_axes, traces, strict=True):
            x, y = _display_envelope(trace.adc_values, trace.adc_sample_rate_hz)
            ax.plot(x, y, color=METHOD_COLOURS[trace.method], linewidth=0.8)
            ax.axvline(trace.host_duration_seconds, color="#374151", linestyle=":", linewidth=1)
            ax.set_xlim(0.0, max_duration * 1.01)
            ax.set_ylim(lower - padding, upper + padding)
            ax.set_title(
                f"Raw CW · {METHOD_LABELS[trace.method]} · {trace.host_duration_seconds:.3f} s",
                loc="left",
                fontweight="bold",
            )
            ax.set_xlabel("Seconds from optimizer start")
            ax.set_ylabel("ChipWhisperer amplitude (normalized ADC)")

        positions = {"bp_grpo": 1.0, "fo_npg": 0.0}
        for trace in traces:
            row = trace.summary_row()
            y = positions[trace.method]
            colour = METHOD_COLOURS[trace.method]
            amplitude_ax.hlines(y, row["cw_q05"], row["cw_q95"], color=colour, linewidth=2)
            amplitude_ax.hlines(y, row["cw_q25"], row["cw_q75"], color=colour, linewidth=7)
            amplitude_ax.plot(row["cw_median"], y, "o", color="white", markeredgecolor=colour)
            amplitude_ax.plot(row["cw_rms"], y, ">", color=colour, markersize=7)
        amplitude_ax.set_yticks([1.0, 0.0], ["BP-GRPO", "FO-NPG"])
        amplitude_ax.set_xlabel("normalized_adc (amplitude only; not watts)")
        amplitude_ax.set_title("ADC distribution and RMS", loc="left", fontweight="bold")
        amplitude_ax.legend(
            handles=[
                Line2D([0], [0], color="#374151", linewidth=2, label="5th–95th percentile"),
                Line2D([0], [0], color="#374151", linewidth=7, label="25th–75th percentile"),
                Line2D([0], [0], marker="o", color="#374151", linewidth=0, label="Median"),
                Line2D([0], [0], marker=">", color="#374151", linewidth=0, label="RMS"),
            ],
            loc="center",
            fontsize=7.5,
            ncols=2,
        )

        if all(trace.nvml is not None for trace in traces):
            summary_lines: list[str] = []
            for trace in traces:
                assert trace.nvml is not None
                nvml_ax.plot(
                    trace.nvml.relative_time_seconds,
                    trace.nvml.power_w,
                    color=METHOD_COLOURS[trace.method],
                    linewidth=1.4,
                    marker="o",
                    markersize=2.5,
                    label=METHOD_LABELS[trace.method],
                )
                summary_lines.append(
                    f"{METHOD_LABELS[trace.method]}: raw {trace.nvml.energy_j:.1f} J · "
                    f"mean {trace.nvml.mean_power_w:.1f} W · peak {trace.nvml.peak_power_w:.1f} W"
                )
                if trace.nvml.baseline_mean_power_w is not None:
                    assert trace.nvml.baseline_adjusted_dynamic_energy_j is not None
                    nvml_ax.axhline(
                        trace.nvml.baseline_mean_power_w,
                        color=METHOD_COLOURS[trace.method],
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.65,
                    )
                    summary_lines.append(
                        f"  1 s pre mean {trace.nvml.baseline_mean_power_w:.1f} W · "
                        f"baseline-adjusted net {trace.nvml.baseline_adjusted_dynamic_energy_j:+.1f} J"
                    )
            nvml_ax.set_xlim(0.0, max_duration * 1.01)
            nvml_ax.set_xlabel("Seconds from host optimizer start")
            nvml_ax.set_ylabel("NVML power (W)")
            nvml_ax.set_title(
                "Timestamped NVML watts · raw energy is primary",
                loc="left",
                fontweight="bold",
            )
            nvml_ax.legend(loc="upper right", fontsize=8)
            nvml_ax.text(
                0.02,
                0.03,
                "\n".join(summary_lines),
                transform=nvml_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        else:
            nvml_ax.grid(False)
            nvml_ax.set_xticks([])
            nvml_ax.set_yticks([])
            nvml_ax.text(
                0.5,
                0.5,
                "NVML absent\nNo watts or joules computed",
                transform=nvml_ax.transAxes,
                ha="center",
                va="center",
                color="#6B7280",
            )
            nvml_ax.set_title("Timestamped NVML context", loc="left", fontweight="bold")

        fig.suptitle(
            "Optimizer power traces · one capture per method\n"
            "Descriptive n=1 only; no repeat-based uncertainty interval",
            fontweight="bold",
        )
        figure_paths = _save_figure(fig, output, "power_trace_comparison")
    return {
        "summary_csv": summary_path,
        "comparison_png": figure_paths["png"],
        "comparison_pdf": figure_paths["pdf"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one BP and one FO SideCapture optimizer trace (descriptive n=1)."
    )
    parser.add_argument("--bp-store", type=Path, required=True)
    parser.add_argument("--fo-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifacts = plot_power_trace_comparison(args.bp_store, args.fo_store, args.output)
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NVMLSummary",
    "OptimizerPowerTrace",
    "load_power_trace_pair",
    "load_sidecapture_optimizer_trace",
    "main",
    "plot_power_trace_comparison",
    "summarize_power_trace_pair",
]
