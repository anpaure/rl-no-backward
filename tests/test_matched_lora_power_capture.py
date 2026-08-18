from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import rl_no_backward.matched_lora_power_capture as power_module
from rl_no_backward.matched_lora_power_capture import (
    BACKPROP_MODULE,
    CAPTURE_METHODS,
    CHIPWHISPERER_COMMIT,
    CHIPWHISPERER_REF,
    CHIPWHISPERER_SERIAL,
    CHIPWHISPERER_VERSION,
    POWER_PAIR_SCHEMA,
    SIDECAPTURE_COMMIT,
    SIDECAPTURE_VERSION,
    PowerCaptureConfig,
    _capture_optimizer_with_sidecapture,
    _load_sidecapture_api,
    _PreparedOptimizerCapture,
    _result_mapping,
    _safe_relative_file,
    _trace_file_receipts,
    _validate_matched_pair,
    _validate_trace_record,
    load_power_capture_config,
    run_power_capture_pair,
)

CONFIG_PATH = Path("configs/gsm8k_matched_lora_power_trace.yaml")
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _request() -> dict[str, object]:
    return {
        "duration_s": 30.0,
        "sample_rate_hz": 5_000.0,
        "pretrigger_s": 0.0,
        "mode": "burst",
        "bits_per_sample": 12,
        "gain_db": 10.0,
        "channel": "power",
    }


def _resolved() -> dict[str, object]:
    return {
        "mode": "burst",
        "sample_rate_hz": 5_000.0,
        "samples": 150_000,
        "duration_s": 30.0,
        "pretrigger_samples": 0,
        "raw_sample_rate_hz": 150_000_000.0,
        "decimation": 30_000,
        "bits_per_sample": 12,
        "warnings": [],
        "details": {},
    }


def _sampler_metadata(*, nvml: bool = True) -> dict[str, object]:
    primary = {
        "backend": "chipwhisperer",
        "chipwhisperer_version": CHIPWHISPERER_VERSION,
        "serial_number": CHIPWHISPERER_SERIAL,
        "product_id": "0xace6",
        "clock_hz": 150_000_000.0,
        "trigger": "auto",
        "usb_read_mode": "auto",
        "safe_memory_fraction": 0.65,
        "resolved": _resolved(),
        "adc_error_controls": {
            "lo_gain_errors_disabled_required": True,
            "lo_gain_errors_disabled_readback": True,
            "clip_errors_disabled_supported": True,
            "clip_errors_disabled_requested": False,
            "clip_errors_disabled_readback": False,
            "sidecapture_adc_clipping_validator_enabled": True,
        },
    }
    if not nvml:
        return primary
    return {
        "backend": "composite",
        "primary": primary,
        "auxiliaries": [
            {
                "backend": "nvml",
                "gpu_index": 0,
                "interval_s": 0.01,
                "clock_domain": "host_monotonic",
            }
        ],
    }


def _make_trace(root: Path, method: str = "bp_grpo") -> dict[str, object]:
    sampler = _sampler_metadata()
    _write_json(
        root / "manifest.json",
        {
            "schema_version": power_module.SIDECAPTURE_DATASET_SCHEMA,
            "experiment": {
                "request": _request(),
                "resolved": _resolved(),
                "sampler": sampler,
            },
        },
    )
    power_path = root / "channels/power/shard_000000/capture_000000000.npy"
    power_path.parent.mkdir(parents=True)
    power_path.write_bytes(b"NUMPY")
    annotations = [
        {
            "name": f"{method}.optimizer.ready",
            "clock_domain": "host_monotonic",
            "sync": "none",
        },
        {
            "name": f"{method}.optimizer.host",
            "clock_domain": "host_monotonic",
            "sync": "none",
        },
        {
            "name": f"{method}.optimizer.cuda",
            "clock_domain": "cuda",
            "sync": "both",
        },
        {
            "name": f"{method}.optimizer.complete",
            "clock_domain": "host_monotonic",
            "sync": "none",
        },
    ]
    annotation_path = root / "annotations/shard_000000/capture_000000000.json"
    _write_json(annotation_path, annotations)
    for name in ("matched_candidate", "optimizer_result"):
        _write_json(root / f"artifacts/shard_000000/capture_000000000.{name}.json", {"ok": True})
    record = {
        "schema_version": power_module.SIDECAPTURE_DATASET_SCHEMA,
        "index": 0,
        "attempt": 1,
        "primary_channel": "power",
        "health": {"ok": True, "issues": [], "metrics": {}},
        "labels": {"method": method},
        "channels": {
            "power": {
                "path": power_path.relative_to(root).as_posix(),
                "dtype": "float32",
                "shape": [150_000],
                "sample_rate_hz": 5_000.0,
                "unit": "normalized_adc",
                "metadata": {
                    "calibrated": False,
                    "gain_db": 10.0,
                    "bits_per_sample": 12,
                },
            }
        },
        "annotations": {
            "path": annotation_path.relative_to(root).as_posix(),
            "count": 4,
        },
        "artifacts": {
            name: {
                "path": (f"artifacts/shard_000000/capture_000000000.{name}.json"),
                "format": "json",
            }
            for name in ("matched_candidate", "optimizer_result")
        },
        "sampler_metadata": sampler,
    }
    _write_json(root / "records/shard_000000/capture_000000000.json", record)
    return record


def _pair_child(method: str) -> dict[str, object]:
    digests = {name: f"digest-{name}" for name in power_module._MATCHED_DIGEST_FIELDS}
    candidate = {"step": 1, "rollout_seed": 20_001, "example_ids": ["example-0"]}
    digests["candidate_digest"] = power_module._json_digest(candidate)
    digests["prompt_example_ids_digest"] = power_module._json_digest(candidate["example_ids"])
    digests["initialization_file_sha256"] = power_module._sha256_file(CONFIG_PATH)
    backward_calls = 8 if method == "bp_grpo" else 0
    return {
        "method": method,
        "source_commit": SOURCE_COMMIT,
        "config_sha256": "config-digest",
        **digests,
        "candidate": candidate,
        "initialization_path": str(CONFIG_PATH.resolve()),
        "optimizer_result": {
            "backward_calls": backward_calls,
            "forward_calls": 16,
            "policy_evaluations": 2,
        },
        "fo_backprop_module_imported": False if method == "fo_npg" else None,
        "frozen_base_parameter_digest_before": "base-digest",
        "frozen_base_parameter_digest_after": "base-digest",
        "result_receipt_sha256": f"receipt-{method}",
    }


def test_module_import_is_sidecapture_and_runner_lazy() -> None:
    code = (
        "import sys; import rl_no_backward.matched_lora_power_capture; "
        "assert 'sidecapture' not in sys.modules; "
        "assert 'rl_no_backward.matched_lora_runner' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def test_locked_config_is_exact_and_has_no_eval_or_wandb() -> None:
    matched, power = load_power_capture_config(CONFIG_PATH)
    assert matched.seeds == [0]
    assert matched.steps == 1
    assert matched.methods == ["base", "bp_grpo", "fo_npg"]
    assert matched.wandb_mode == "disabled"
    assert matched.run_test_evaluation is False
    assert matched.test_size == 0
    assert power.duration == "30s"
    assert power.sample_rate == "5kHz"
    assert power.mode == "burst"
    assert power.serial_number == CHIPWHISPERER_SERIAL
    assert power.product_id == 0xACE6
    assert power.clock_hz == 150_000_000.0
    assert power.safe_memory_fraction == 0.65
    assert power.capture_count == power.max_attempts == 1
    assert power.trigger_delay_s == 1.0
    assert power.nvml_auxiliary is True
    assert power.nvml_interval_s == 0.01
    assert power.sidecapture_commit == SIDECAPTURE_COMMIT
    assert power.chipwhisperer_commit == CHIPWHISPERER_COMMIT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", "31s"),
        ("sample_rate", "10kHz"),
        ("mode", "stream"),
        ("max_attempts", 2),
        ("capture_count", 2),
        ("serial_number", "wrong"),
        ("product_id", 1),
        ("clock_hz", 1.0),
        ("safe_memory_fraction", 0.5),
        ("cuda_annotation_sync", "none"),
        ("sidecapture_commit", "wrong"),
        ("chipwhisperer_commit", "wrong"),
    ],
)
def test_power_config_rejects_comparability_drift(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        PowerCaptureConfig.from_mapping({field: value})


def test_sidecapture_and_chipwhisperer_vcs_installations_are_hard_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sidecapture = ModuleType("sidecapture")
    fake_sidecapture.__version__ = SIDECAPTURE_VERSION
    monkeypatch.setitem(sys.modules, "sidecapture", fake_sidecapture)

    def provenance(name: str) -> dict[str, object]:
        if name == "sidecapture":
            return {
                "version": SIDECAPTURE_VERSION,
                "direct_url": {"vcs_info": {"commit_id": SIDECAPTURE_COMMIT}},
            }
        return {
            "version": CHIPWHISPERER_VERSION,
            "direct_url": {
                "vcs_info": {
                    "commit_id": CHIPWHISPERER_COMMIT,
                    "requested_revision": CHIPWHISPERER_REF,
                }
            },
        }

    monkeypatch.setattr(power_module, "_distribution_provenance", provenance)
    assert _load_sidecapture_api(PowerCaptureConfig()) is fake_sidecapture

    def wrong_provenance(name: str) -> dict[str, object]:
        value = provenance(name)
        if name == "chipwhisperer":
            value["direct_url"] = {"vcs_info": {"commit_id": "wrong"}}
        return value

    monkeypatch.setattr(power_module, "_distribution_provenance", wrong_provenance)
    with pytest.raises(RuntimeError, match="pinned ISL fork"):
        _load_sidecapture_api(PowerCaptureConfig())


def test_optimizer_result_must_be_json_finite() -> None:
    @dataclass
    class Result:
        backward_calls: int
        score: float

    assert _result_mapping(Result(0, 1.5)) == {"backward_calls": 0, "score": 1.5}
    with pytest.raises(ValueError, match="Out of range float"):
        _result_mapping(Result(0, float("nan")))


def test_one_shot_capture_uses_locked_husky_controls_and_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}

    class FakeRequest:
        @classmethod
        def create(cls, **kwargs: object) -> dict[str, object]:
            state["request"] = kwargs
            return dict(kwargs)

    class FakeChipWhispererSampler:
        def __init__(self, request: object, **kwargs: object) -> None:
            self.request = request
            self.kwargs = kwargs
            self.scope = SimpleNamespace(
                adc=SimpleNamespace(
                    lo_gain_errors_disabled=False,
                    clip_errors_disabled=True,
                )
            )

        def plan(self) -> object:
            class Resolved:
                def to_dict(self) -> dict[str, object]:
                    return _resolved()

            return Resolved()

        def metadata(self) -> dict[str, object]:
            return {"backend": "chipwhisperer", **self.kwargs}

    class FakeNVMLSampler:
        def __init__(self, **kwargs: object) -> None:
            state["nvml"] = kwargs

    class FakeCompositeSampler:
        def __init__(self, primary: object, *auxiliaries: object) -> None:
            self.primary = primary
            self.auxiliaries = auxiliaries

    class FakeDirectoryStore:
        def __init__(self, root: Path, **kwargs: object) -> None:
            state["store"] = {"root": root, **kwargs}

    class FakeRetryPolicy:
        def __init__(self, **kwargs: object) -> None:
            state["retry"] = kwargs

    class FakeWorkload:
        pass

    class FakeContext:
        def __init__(self) -> None:
            self.annotations: list[tuple[str, str, str | None]] = []
            self.artifacts: dict[str, object] = {}

        def mark(self, name: str, **metadata: object) -> None:
            del metadata
            self.annotations.append((name, "host", None))

        @contextmanager
        def region(self, name: str, **metadata: object):
            del metadata
            self.annotations.append((name, "host", None))
            yield

        @contextmanager
        def cuda_region(self, name: str, *, sync: str, **metadata: object):
            del metadata
            self.annotations.append((name, "cuda", sync))
            yield

        def add_artifact(self, name: str, value: object) -> None:
            self.artifacts[name] = value

    class FakeExperiment:
        def __init__(self, **kwargs: object) -> None:
            state["experiment"] = kwargs
            self.workload = kwargs["workload"]
            self.sampler = kwargs["sampler"]

        def __enter__(self):
            primary = getattr(self.sampler, "primary", self.sampler)
            primary.plan()
            state["primary_metadata"] = primary.metadata()
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def run(self, count: int, *, labels: dict[str, object]):
            state["run"] = {"count": count, "labels": labels}
            context = FakeContext()
            self.workload.run(context)
            state["context"] = context
            return [{"index": 0, "attempt": 1}]

    fake_api = SimpleNamespace(
        CaptureRequest=FakeRequest,
        ChipWhispererSampler=FakeChipWhispererSampler,
        NVMLSampler=FakeNVMLSampler,
        CompositeSampler=FakeCompositeSampler,
        DirectoryStore=FakeDirectoryStore,
        RetryPolicy=FakeRetryPolicy,
        Workload=FakeWorkload,
        Experiment=FakeExperiment,
    )
    monkeypatch.setattr(power_module, "_load_sidecapture_api", lambda _: fake_api)
    validations: list[dict[str, object]] = []
    prepared = _PreparedOptimizerCapture(
        method="fo_npg",
        optimizer_step=lambda: {
            "backward_calls": 0,
            "forward_calls": 20,
            "policy_evaluations": 20,
        },
        lora_digest=lambda: "lora",
        frozen_base_digest=lambda: "base",
        validate_after=lambda result: validations.append(dict(result)),
        provenance={"candidate_digest": "candidate"},
        runtime={},
    )
    record, result = _capture_optimizer_with_sidecapture(
        prepared,
        power=PowerCaptureConfig(),
        trace_root=tmp_path / "trace",
    )
    assert record == {"index": 0, "attempt": 1}
    assert result["backward_calls"] == 0
    assert len(validations) == 1
    assert state["request"] == {
        "duration": "30s",
        "sample_rate": "5kHz",
        "mode": "burst",
        "bits_per_sample": 12,
        "gain_db": 10.0,
        "channel": "power",
    }
    experiment = state["experiment"]
    assert isinstance(experiment, dict)
    assert experiment["warmup"] == 0
    assert experiment["trigger_delay_s"] == 1.0
    assert experiment["workload_sync"] == "none"
    assert state["retry"] == {
        "max_attempts": 1,
        "backoff_s": 0.0,
        "recover_sampler": False,
    }
    assert state["run"]["count"] == 1  # type: ignore[index]
    primary = state["primary_metadata"]
    assert primary["product_id"] == 0xACE6  # type: ignore[index]
    assert primary["clock_hz"] == 150_000_000.0  # type: ignore[index]
    controls = primary["adc_error_controls"]  # type: ignore[index]
    assert controls["lo_gain_errors_disabled_readback"] is True
    assert controls["clip_errors_disabled_readback"] is False
    context = state["context"]
    assert context.annotations == [
        ("fo_npg.optimizer.ready", "host", None),
        ("fo_npg.optimizer.host", "host", None),
        ("fo_npg.optimizer.cuda", "cuda", "both"),
        ("fo_npg.optimizer.complete", "host", None),
    ]
    assert set(context.artifacts) == {"matched_candidate", "optimizer_result"}


def test_trace_validator_binds_exact_hardware_plan_and_method_markers(tmp_path: Path) -> None:
    record = _make_trace(tmp_path, "bp_grpo")
    receipt = _validate_trace_record(
        tmp_path,
        record,
        method="bp_grpo",
        power=PowerCaptureConfig(),
    )
    assert receipt["resolved_capture"]["samples"] == 150_000
    assert receipt["primary_channel"]["unit"] == "normalized_adc"
    assert receipt["annotation_count"] == 4


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("manifest", "sample_rate_hz", 4_999.0, "request sample_rate_hz"),
        ("resolved", "decimation", 29_999, "resolved plan decimation"),
        ("sampler", "product_id", "0x1234", "primary sampler metadata"),
    ],
)
def test_trace_validator_rejects_hardware_plan_drift(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    record = _make_trace(tmp_path)
    if target == "manifest":
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["experiment"]["request"][field] = value
        _write_json(tmp_path / "manifest.json", manifest)
    elif target == "resolved":
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["experiment"]["resolved"][field] = value
        _write_json(tmp_path / "manifest.json", manifest)
    else:
        record["sampler_metadata"]["primary"][field] = value  # type: ignore[index]
        _write_json(tmp_path / "records/shard_000000/capture_000000000.json", record)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["experiment"]["sampler"] = record["sampler_metadata"]
        _write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match=message):
        _validate_trace_record(
            tmp_path,
            record,
            method="bp_grpo",
            power=PowerCaptureConfig(),
        )


def test_artifact_paths_reject_parent_and_symlink_escapes(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes"):
        _safe_relative_file(root, "../outside.bin")
    (root / "linked.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _safe_relative_file(root, "linked.bin")
    with pytest.raises(ValueError, match="symlink"):
        _trace_file_receipts(root)


def test_pair_validator_requires_identical_inputs_and_strict_fo_boundary() -> None:
    bp = _pair_child("bp_grpo")
    fo = _pair_child("fo_npg")
    matched = _validate_matched_pair(bp, fo, source_commit=SOURCE_COMMIT)
    assert set(matched) == set(power_module._MATCHED_DIGEST_FIELDS)

    fo["reward_digest"] = "different"
    with pytest.raises(RuntimeError, match="reward_digest"):
        _validate_matched_pair(bp, fo, source_commit=SOURCE_COMMIT)
    fo = _pair_child("fo_npg")
    fo["hf_old_logprob_digest"] = "different"
    with pytest.raises(RuntimeError, match="hf_old_logprob_digest"):
        _validate_matched_pair(bp, fo, source_commit=SOURCE_COMMIT)
    fo = _pair_child("fo_npg")
    fo["optimizer_result"]["backward_calls"] = 1  # type: ignore[index]
    with pytest.raises(RuntimeError, match="no-backward"):
        _validate_matched_pair(bp, fo, source_commit=SOURCE_COMMIT)


def test_parent_launches_exactly_one_child_per_method_in_fixed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("locked: true\n", encoding="utf-8")
    output = tmp_path / "outside" / "pair"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    launched: list[list[str]] = []

    fake_matched = SimpleNamespace()
    monkeypatch.setattr(
        power_module,
        "load_power_capture_config",
        lambda _: (fake_matched, PowerCaptureConfig()),
    )
    monkeypatch.setattr(
        power_module,
        "_verify_clean_source",
        lambda: (SOURCE_COMMIT, worktree),
    )
    monkeypatch.setattr(power_module, "_require_output_outside_worktree", lambda *_: None)
    monkeypatch.setattr(power_module, "_verify_committed_config", lambda *_: None)

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs == {"cwd": worktree, "check": True}
        launched.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(power_module.subprocess, "run", run)
    monkeypatch.setattr(
        power_module,
        "_verify_result_receipt",
        lambda path, *, method, source_commit: _pair_child(method),
    )
    monkeypatch.setattr(
        power_module,
        "_sha256_file",
        lambda path: f"sha-{Path(path).name}",
    )
    destination = run_power_capture_pair(config, output)
    assert destination == output / "power_pair.json"
    assert len(launched) == 2
    assert [command[command.index("--child-method") + 1] for command in launched] == list(
        CAPTURE_METHODS
    )
    assert all(command[command.index("--seed") + 1] == "0" for command in launched)
    assert all(command[command.index("--step") + 1] == "1" for command in launched)
    pair = json.loads(destination.read_text())
    assert pair["schema"] == POWER_PAIR_SCHEMA
    assert pair["capture_order"] == ["bp_grpo", "fo_npg"]
    assert pair["sidecapture_experiment_run_calls"] == 2
    assert pair["total_capture_count"] == 2
    assert pair["comparison_valid"] is True
    assert pair["results"]["bp_grpo"]["sidecapture_store"] == "bp_grpo/sidecapture"
    assert pair["results"]["fo_npg"]["sidecapture_store"] == "fo_npg/sidecapture"


def test_parent_refuses_overwrite_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        power_module,
        "load_power_capture_config",
        lambda _: (SimpleNamespace(), PowerCaptureConfig()),
    )
    monkeypatch.setattr(
        power_module,
        "_verify_clean_source",
        lambda: (SOURCE_COMMIT, tmp_path / "worktree"),
    )
    monkeypatch.setattr(power_module, "_require_output_outside_worktree", lambda *_: None)
    monkeypatch.setattr(power_module, "_verify_committed_config", lambda *_: None)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_power_capture_pair(CONFIG_PATH, output)


def test_forward_module_constant_does_not_alias_runner_forward_path() -> None:
    assert BACKPROP_MODULE.endswith("matched_lora_backprop")
    assert "matched_lora_forward_only" not in BACKPROP_MODULE
