"""One-shot matched-LoRA ChipWhisperer captures for BP-GRPO versus FO-NPG.

The public mode is a small parent orchestrator.  It starts exactly one fresh
Python child for each trained method, then verifies that both children used the
same initialization and frozen rollout.  A child performs all model and vLLM
setup before arming SideCapture; only the optimizer/update callback is inside
the hardware trace window.

SideCapture is intentionally imported only inside the child capture function.
This keeps ordinary repository imports and CPU-only tests independent of the
hardware package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

CAPTURE_METHODS = ("bp_grpo", "fo_npg")
POWER_CAPTURE_SCHEMA = "rl-no-backward-matched-lora-power-capture-v1"
POWER_PAIR_SCHEMA = "rl-no-backward-matched-lora-power-pair-v1"
SIDECAPTURE_DATASET_SCHEMA = "sidecapture.dataset/v1"
SIDECAPTURE_COMMIT = "c1ef46862a72b8557af03d781200d226fcda738e"
SIDECAPTURE_VERSION = "0.1.1"
CHIPWHISPERER_REF = "v6.0.0-isl.1"
CHIPWHISPERER_VERSION = "6.0.0+isl.1"
CHIPWHISPERER_COMMIT = "ed0ed33efbc55421f4e1d995816fd50a5b2e5d01"
CHIPWHISPERER_SERIAL = "50203220325531583230313235303038"
BACKPROP_MODULE = "rl_no_backward.matched_lora_backprop"

_MATCHED_DIGEST_FIELDS = (
    "candidate_digest",
    "initialization_digest",
    "initialization_file_sha256",
    "initial_lora_digest",
    "behavior_policy_digest",
    "rollout_token_digest",
    "behavior_logprob_digest",
    "hf_old_logprob_digest",
    "reward_digest",
    "prompt_example_ids_digest",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON mapping")
    return dict(value)


def _require_exact_int(value: Any, *, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return value


def _require_exact_number(value: Any, *, name: str, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return result


@dataclass(frozen=True, slots=True)
class PowerCaptureConfig:
    """Locked settings for the two one-shot physical captures."""

    duration: str = "16s"
    sample_rate: str = "18.75kHz"
    mode: str = "burst"
    bits_per_sample: int = 12
    gain_db: float = 10.0
    channel: str = "power"
    serial_number: str = CHIPWHISPERER_SERIAL
    product_id: int = 0xACE6
    clock_hz: float = 150_000_000.0
    safe_memory_fraction: float = 1.0
    trigger: str = "auto"
    usb_read_mode: str = "auto"
    trigger_delay_s: float = 1.0
    capture_count: int = 1
    max_attempts: int = 1
    warmup: int = 0
    workload_sync: str = "none"
    cuda_annotation_sync: str = "both"
    trace_dtype: str = "float32"
    nvml_auxiliary: bool = True
    nvml_gpu_index: int = 0
    nvml_interval_s: float = 0.01
    sidecapture_version: str = SIDECAPTURE_VERSION
    sidecapture_commit: str = SIDECAPTURE_COMMIT
    chipwhisperer_ref: str = CHIPWHISPERER_REF
    chipwhisperer_version: str = CHIPWHISPERER_VERSION
    chipwhisperer_commit: str = CHIPWHISPERER_COMMIT

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> PowerCaptureConfig:
        payload = dict(mapping or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown power_capture config keys: {unknown}")
        return cls(**payload)

    def __post_init__(self) -> None:
        exact_strings = {
            "duration": "16s",
            "sample_rate": "18.75kHz",
            "mode": "burst",
            "channel": "power",
            "serial_number": CHIPWHISPERER_SERIAL,
            "trigger": "auto",
            "usb_read_mode": "auto",
            "workload_sync": "none",
            "cuda_annotation_sync": "both",
            "trace_dtype": "float32",
            "sidecapture_version": SIDECAPTURE_VERSION,
            "sidecapture_commit": SIDECAPTURE_COMMIT,
            "chipwhisperer_ref": CHIPWHISPERER_REF,
            "chipwhisperer_version": CHIPWHISPERER_VERSION,
            "chipwhisperer_commit": CHIPWHISPERER_COMMIT,
        }
        for name, expected in exact_strings.items():
            if getattr(self, name) != expected:
                raise ValueError(f"power_capture.{name} must be exactly {expected!r}")
        for name, expected in {
            "bits_per_sample": 12,
            "capture_count": 1,
            "max_attempts": 1,
            "warmup": 0,
            "nvml_gpu_index": 0,
            "product_id": 0xACE6,
        }.items():
            _require_exact_int(getattr(self, name), name=f"power_capture.{name}", expected=expected)
        _require_exact_number(self.gain_db, name="power_capture.gain_db", expected=10.0)
        _require_exact_number(
            self.clock_hz,
            name="power_capture.clock_hz",
            expected=150_000_000.0,
        )
        _require_exact_number(
            self.safe_memory_fraction,
            name="power_capture.safe_memory_fraction",
            expected=1.0,
        )
        _require_exact_number(
            self.trigger_delay_s,
            name="power_capture.trigger_delay_s",
            expected=1.0,
        )
        _require_exact_number(
            self.nvml_interval_s,
            name="power_capture.nvml_interval_s",
            expected=0.01,
        )
        if not isinstance(self.nvml_auxiliary, bool):
            raise TypeError("power_capture.nvml_auxiliary must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _PreparedOptimizerCapture:
    method: str
    optimizer_step: Callable[[], Any]
    lora_digest: Callable[[], str]
    frozen_base_digest: Callable[[], str]
    validate_after: Callable[[Mapping[str, Any]], None]
    provenance: dict[str, Any]
    runtime: dict[str, Any]


def load_power_capture_config(path: str | Path) -> tuple[Any, PowerCaptureConfig]:
    """Load the matched config plus the isolated ``power_capture`` extension."""

    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("power capture config must contain a YAML mapping")
    payload = dict(value)
    power_value = payload.pop("power_capture", None)
    if not isinstance(power_value, Mapping):
        raise TypeError("power capture config requires a power_capture mapping")
    # Lazy to keep module import independent of torch, vLLM, and runner internals.
    from .matched_lora_runner import MatchedLoRAExperimentConfig

    matched = MatchedLoRAExperimentConfig.from_mapping(payload)
    power = PowerCaptureConfig.from_mapping(power_value)
    if matched.seeds != [0] or matched.steps != 1:
        raise ValueError("power capture matched config must lock seeds=[0] and steps=1")
    if matched.methods != ["base", "bp_grpo", "fo_npg"]:
        raise ValueError("power capture matched methods must be [base, bp_grpo, fo_npg]")
    if matched.wandb_mode != "disabled":
        raise ValueError("power capture config must disable W&B")
    if matched.run_test_evaluation or matched.test_size != 0:
        raise ValueError("power capture must not evaluate or access locked-test rows")
    return matched, power


def _git_output(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _verify_clean_source(expected_commit: str | None = None) -> tuple[str, Path]:
    worktree_value = _git_output("rev-parse", "--show-toplevel")
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    if worktree_value is None or commit is None or len(commit) != 40:
        raise RuntimeError("power capture requires a Git worktree with a 40-character HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("power capture child source commit differs from its parent")
    if status is None or status:
        raise RuntimeError("power capture requires an exactly clean Git worktree")
    return commit, Path(worktree_value).resolve()


def _require_output_outside_worktree(output: Path, worktree: Path) -> None:
    resolved = output.resolve()
    try:
        resolved.relative_to(worktree.resolve())
    except ValueError:
        return
    raise ValueError("power capture output must be outside the Git worktree")


def _verify_committed_config(config: Path, worktree: Path, source_commit: str) -> None:
    resolved = config.resolve(strict=True)
    try:
        relative = resolved.relative_to(worktree.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("power capture config must be a committed worktree file") from error
    tracked = _git_output("ls-files", "--error-unmatch", "--", relative, cwd=worktree)
    if tracked != relative:
        raise RuntimeError("power capture config is not tracked at the launch commit")
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=worktree,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("power capture config is absent from the launch commit") from error
    if hashlib.sha256(committed).hexdigest() != _sha256_file(resolved):
        raise RuntimeError("power capture config bytes differ from the launch commit")


def _prepare_new_directory(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing power capture output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _distribution_provenance(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "version": None, "direct_url": None}
    direct_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_text) if direct_text else None
    return {"name": name, "version": distribution.version, "direct_url": direct_url}


def _load_sidecapture_api(power: PowerCaptureConfig) -> ModuleType:
    """Import and pin SideCapture only at authorized child invocation time."""

    try:
        import sidecapture as sc
    except ImportError as error:  # pragma: no cover - only exercised on the H100 host
        raise RuntimeError(
            "SideCapture is missing; install the pinned anpaure/sidecapture commit first"
        ) from error
    if getattr(sc, "__version__", None) != power.sidecapture_version:
        raise RuntimeError(
            "SideCapture version mismatch: "
            f"expected {power.sidecapture_version}, got {getattr(sc, '__version__', None)}"
        )
    provenance = _distribution_provenance("sidecapture")
    direct = provenance.get("direct_url")
    vcs = direct.get("vcs_info") if isinstance(direct, Mapping) else None
    installed_commit = vcs.get("commit_id") if isinstance(vcs, Mapping) else None
    if installed_commit != power.sidecapture_commit:
        raise RuntimeError(
            "SideCapture installation is not bound to the pinned Git commit: "
            f"expected {power.sidecapture_commit}, got {installed_commit!r}"
        )
    chipwhisperer = _distribution_provenance("chipwhisperer")
    chipwhisperer_direct = chipwhisperer.get("direct_url")
    chipwhisperer_vcs = (
        chipwhisperer_direct.get("vcs_info") if isinstance(chipwhisperer_direct, Mapping) else None
    )
    chipwhisperer_commit = (
        chipwhisperer_vcs.get("commit_id") if isinstance(chipwhisperer_vcs, Mapping) else None
    )
    chipwhisperer_ref = (
        chipwhisperer_vcs.get("requested_revision")
        if isinstance(chipwhisperer_vcs, Mapping)
        else None
    )
    if (
        chipwhisperer.get("version") != power.chipwhisperer_version
        or chipwhisperer_commit != power.chipwhisperer_commit
        or chipwhisperer_ref != power.chipwhisperer_ref
    ):
        raise RuntimeError(
            "ChipWhisperer installation is not the pinned ISL fork: "
            f"version={chipwhisperer.get('version')!r}, "
            f"commit={chipwhisperer_commit!r}, ref={chipwhisperer_ref!r}"
        )
    return sc


def _best_effort_nvidia_smi() -> dict[str, Any] | None:
    try:
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return {"query": "index,uuid,pci.bus_id,name,driver_version", "rows": row.splitlines()}


def _runtime_metadata(torch: Any, *, resolved_model_snapshot: str) -> dict[str, Any]:
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_index": device_index,
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "nvidia_smi": _best_effort_nvidia_smi(),
        "resolved_model_snapshot": resolved_model_snapshot,
        "packages": {
            name: _distribution_provenance(name)
            for name in ("sidecapture", "chipwhisperer", "torch", "vllm", "flash-attn")
        },
    }


def _prepare_optimizer_capture(
    matched: Any,
    *,
    method: str,
    seed: int,
    step: int,
    output: Path,
) -> _PreparedOptimizerCapture:
    """Prepare one immutable rollout, leaving only its update callable uncaptured."""

    if method not in CAPTURE_METHODS or seed != 0 or step != 1:
        raise ValueError("power capture child is locked to bp_grpo/fo_npg, seed=0, step=1")
    if method == "fo_npg" and BACKPROP_MODULE in sys.modules:
        raise RuntimeError("forward-only child imported the reverse-mode implementation")

    import torch

    from . import matched_lora_runner as runner

    initialization_path = Path(matched.shared_initialization_path_template.format(seed=seed))
    official_train = runner.load_gsm8k_split("train", revision=matched.dataset_revision)
    train = runner.select_seeded_subset(
        official_train,
        matched.train_size,
        seed=matched.subset_seed,
        namespace="matched-standard-lora-train-v1",
    )
    schedule = runner.build_prompt_schedule(train, matched, seed=seed)
    if len(schedule) != 1 or int(schedule[0]["step"]) != step:
        raise RuntimeError("power capture did not reconstruct the single locked schedule entry")
    entry = dict(schedule[0])
    train_by_id = {example.example_id: example for example in train}
    examples = tuple(train_by_id[str(example_id)] for example_id in entry["example_ids"])

    bundle, snapshot, initialization_digest = runner._load_bundle(matched, seed=seed)
    if not initialization_path.is_file():
        raise FileNotFoundError("model loader did not materialize the shared LoRA initialization")
    initialization_file_sha256 = _sha256_file(initialization_path)
    loaded_digest = runner.load_shared_lora_initialization(
        initialization_path,
        bundle.model,
        matched.lora,
    )
    initial_lora_digest = runner.lora_state_digest(bundle.model)
    if loaded_digest != initialization_digest or initial_lora_digest != initialization_digest:
        raise RuntimeError("child did not start from the immutable shared LoRA initialization")
    frozen_base_before = runner.frozen_base_parameter_digest(bundle.model)

    if method == "fo_npg":
        runner.set_adapter_grad_enabled(bundle, False)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        runner.assert_lora_frozen(bundle.model)

    engine = runner.create_standard_lora_vllm_engine(
        matched.model_name,
        revision=matched.model_revision,
        dtype=matched.dtype,
        max_model_len=matched.max_prompt_tokens + matched.max_new_tokens,
        max_lora_rank=matched.lora.rank,
        kv_cache_memory_bytes=matched.vllm_kv_cache_memory_bytes,
        enforce_eager=matched.vllm_enforce_eager,
        flash_attn_version=matched.vllm_flash_attn_version,
        max_num_seqs=matched.responses_per_step,
        seed=seed,
    )
    generator = runner.ReloadableLoRAGenerator(engine)
    reload_receipt = generator.sync(
        bundle.model,
        output / "vllm_lora_exports",
        version=1,
        policy_version=f"power-capture/{method}/seed={seed}/step={step}/frozen-rollout",
    )
    runner._synchronize_cuda()
    before_rollout_digest = runner.lora_state_digest(bundle.model)
    if generator.state_digest != before_rollout_digest:
        raise RuntimeError("vLLM behavior policy digest differs from live HF initialization")
    rollout = runner.build_matched_rollout(
        bundle,
        generator,
        examples,
        matched,
        rollout_seed=int(entry["rollout_seed"]),
    )
    runner._synchronize_cuda()
    if runner.lora_state_digest(bundle.model) != before_rollout_digest:
        raise RuntimeError("frozen rollout mutated the HF LoRA policy")
    if rollout.behavior_policy_digest != before_rollout_digest:
        raise RuntimeError("rollout behavior policy is not the shared initialization")

    if method == "bp_grpo":
        from .matched_lora_backprop import (
            make_matched_lora_optimizer,
            make_matched_lr_scheduler,
            matched_backprop_grpo_step,
        )

        optimizer = make_matched_lora_optimizer(bundle, matched.backprop)
        scheduler = make_matched_lr_scheduler(
            optimizer,
            matched.backprop,
            total_steps=matched.steps,
        )

        def optimizer_step() -> Any:
            return matched_backprop_grpo_step(
                bundle,
                rollout.rollout,
                rollout.sampler_token_log_probs,
                optimizer,
                matched.objective,
                matched.backprop,
                scheduler,
            )

    else:
        direction_generator = torch.Generator(device=bundle.device).manual_seed(seed + 70_000)

        def optimizer_step() -> Any:
            return runner.matched_forward_npg_step(
                bundle,
                rollout.rollout,
                rollout.sampler_token_log_probs,
                direction_generator,
                matched.objective,
                matched.forward,
            )

    def current_lora_digest() -> str:
        return runner.lora_state_digest(bundle.model)

    def current_frozen_base_digest() -> str:
        return runner.frozen_base_parameter_digest(bundle.model)

    def validate_after(result: Mapping[str, Any]) -> None:
        backward_calls = result.get("backward_calls")
        if isinstance(backward_calls, bool) or not isinstance(backward_calls, int):
            raise TypeError("optimizer result has invalid backward_calls")
        if method == "fo_npg":
            if backward_calls != 0 or BACKPROP_MODULE in sys.modules:
                raise RuntimeError("forward-only capture crossed the no-backward boundary")
            runner.assert_lora_frozen(bundle.model)
            if any(
                parameter.requires_grad or parameter.grad is not None
                for parameter in bundle.trainable_parameters
            ):
                raise RuntimeError("forward-only adapter acquired gradients")
        elif backward_calls <= 0:
            raise RuntimeError("BP-GRPO capture did not execute a backward call")

    prompt_ids = [str(value) for value in entry["example_ids"]]
    candidate_payload = {
        "step": step,
        "rollout_seed": int(entry["rollout_seed"]),
        "example_ids": prompt_ids,
    }
    provenance = {
        "candidate_digest": _json_digest(candidate_payload),
        "candidate": candidate_payload,
        "prompt_example_ids_digest": _json_digest(prompt_ids),
        "initialization_digest": initialization_digest,
        "initialization_path": str(initialization_path.resolve()),
        "initialization_file_sha256": initialization_file_sha256,
        "initial_lora_digest": initial_lora_digest,
        "frozen_base_parameter_digest_before": frozen_base_before,
        "behavior_policy_digest": rollout.behavior_policy_digest,
        "behavior_policy_version": rollout.behavior_policy_version,
        "rollout_token_digest": str(rollout.provenance["rollout_token_digest"]),
        "behavior_logprob_digest": str(rollout.provenance["behavior_logprob_digest"]),
        "hf_old_logprob_digest": str(rollout.provenance["hf_old_logprob_digest"]),
        "reward_digest": str(rollout.provenance["reward_digest"]),
        "vllm_hf_parity": dict(rollout.parity),
        "vllm_reload_receipt": asdict(reload_receipt),
        "responses": int(rollout.rollout.environment_samples),
        "valid_response_tokens": int(rollout.rollout.valid_response_tokens),
        "old_score_forward_calls": int(rollout.old_score_forward_calls),
        "resolved_model_snapshot": str(snapshot),
    }
    runtime = _runtime_metadata(torch, resolved_model_snapshot=str(snapshot))
    return _PreparedOptimizerCapture(
        method=method,
        optimizer_step=optimizer_step,
        lora_digest=current_lora_digest,
        frozen_base_digest=current_frozen_base_digest,
        validate_after=validate_after,
        provenance=provenance,
        runtime=runtime,
    )


def _result_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("captured optimizer step must return a dataclass or mapping")
    # This is both a JSONability/finite-number assertion and a defensive copy.
    decoded = json.loads(_canonical_json_bytes(payload))
    if not isinstance(decoded, dict):  # pragma: no cover - payload was a mapping
        raise TypeError("optimizer result did not serialize to a mapping")
    return decoded


def _capture_optimizer_with_sidecapture(
    prepared: _PreparedOptimizerCapture,
    *,
    power: PowerCaptureConfig,
    trace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exactly one non-replayable optimizer callback in one experiment."""

    sc = _load_sidecapture_api(power)

    class LockedHealthChipWhispererSampler(sc.ChipWhispererSampler):
        """Lock Husky nuisance/clipping controls and expose their readback."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.adc_error_controls: dict[str, Any] | None = None

        def plan(self) -> Any:
            resolved = super().plan()
            resolved_payload = resolved.to_dict()
            expected_plan = {
                "mode": "burst",
                "sample_rate_hz": 18_750.0,
                "samples": 300_000,
                "duration_s": 16.0,
                "pretrigger_samples": 0,
                "raw_sample_rate_hz": 150_000_000.0,
                "decimation": 8_000,
                "bits_per_sample": 12,
            }
            for name, expected in expected_plan.items():
                if resolved_payload.get(name) != expected:
                    raise RuntimeError(
                        f"Husky resolved plan {name} drifted before capture: "
                        f"expected {expected!r}, got {resolved_payload.get(name)!r}"
                    )
            adc = getattr(self.scope, "adc", None)
            if adc is None or not hasattr(adc, "lo_gain_errors_disabled"):
                raise RuntimeError("connected Husky lacks required lo_gain_errors_disabled control")
            adc.lo_gain_errors_disabled = True
            low_gain_readback = bool(adc.lo_gain_errors_disabled)
            if low_gain_readback is not True:
                raise RuntimeError("Husky rejected lo_gain_errors_disabled=True")
            clip_supported = hasattr(adc, "clip_errors_disabled")
            clip_readback: bool | None = None
            if clip_supported:
                # False leaves hardware clipping detection enabled.  SideCapture's
                # normalized-ADC rail/span validator remains enabled as a second gate.
                adc.clip_errors_disabled = False
                clip_readback = bool(adc.clip_errors_disabled)
                if clip_readback is not False:
                    raise RuntimeError("Husky rejected clip_errors_disabled=False")
            self.adc_error_controls = {
                "lo_gain_errors_disabled_required": True,
                "lo_gain_errors_disabled_readback": low_gain_readback,
                "clip_errors_disabled_supported": clip_supported,
                "clip_errors_disabled_requested": False if clip_supported else None,
                "clip_errors_disabled_readback": clip_readback,
                "sidecapture_adc_clipping_validator_enabled": True,
            }
            return resolved

        def metadata(self) -> dict[str, Any]:
            payload = dict(super().metadata())
            payload["adc_error_controls"] = self.adc_error_controls
            return payload

    request = sc.CaptureRequest.create(
        duration=power.duration,
        sample_rate=power.sample_rate,
        mode=power.mode,
        bits_per_sample=power.bits_per_sample,
        gain_db=power.gain_db,
        channel=power.channel,
    )
    primary = LockedHealthChipWhispererSampler(
        request,
        serial_number=power.serial_number,
        product_id=power.product_id,
        clock_hz=power.clock_hz,
        trigger=power.trigger,
        usb_read_mode=power.usb_read_mode,
        safe_memory_fraction=power.safe_memory_fraction,
    )
    sampler = primary
    if power.nvml_auxiliary:
        sampler = sc.CompositeSampler(
            primary,
            sc.NVMLSampler(
                gpu_index=power.nvml_gpu_index,
                interval_s=power.nvml_interval_s,
            ),
        )

    class OneShotOptimizerWorkload(sc.Workload):
        replay_safe = False

        def __init__(self) -> None:
            self.calls = 0
            self.result: dict[str, Any] | None = None

        def run(self, context: Any) -> dict[str, Any]:
            if self.calls != 0:
                raise RuntimeError("one-shot optimizer workload was invoked more than once")
            self.calls += 1
            metadata = {
                "method": prepared.method,
                "seed": 0,
                "step": 1,
                "candidate_digest": prepared.provenance["candidate_digest"],
            }
            context.mark(f"{prepared.method}.optimizer.ready", **metadata)
            started = time.perf_counter()
            with (
                context.region(f"{prepared.method}.optimizer.host", **metadata),
                context.cuda_region(
                    f"{prepared.method}.optimizer.cuda",
                    sync=power.cuda_annotation_sync,
                    **metadata,
                ),
            ):
                raw_result = prepared.optimizer_step()
            elapsed = time.perf_counter() - started
            result = _result_mapping(raw_result)
            result["optimizer_wall_time_seconds"] = elapsed
            context.mark(f"{prepared.method}.optimizer.complete", **metadata)
            # Keep only lightweight no-backward/counter assertions inside the
            # fixed hardware window.  The multi-gigabyte frozen-base digest is
            # intentionally recomputed once, after SideCapture has finished.
            prepared.validate_after(result)
            context.add_artifact("optimizer_result", result)
            context.add_artifact("matched_candidate", prepared.provenance)
            self.result = result
            return result

        def metadata(self) -> dict[str, Any]:
            return {
                "type": "matched_lora_one_shot_optimizer",
                "method": prepared.method,
                "seed": 0,
                "step": 1,
                "replay_safe": False,
                "scope": "optimizer_update_only",
            }

    workload = OneShotOptimizerWorkload()
    experiment = sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(trace_root, trace_dtype=power.trace_dtype, resume=False),
        retry=sc.RetryPolicy(
            max_attempts=power.max_attempts,
            backoff_s=0.0,
            recover_sampler=False,
        ),
        warmup=power.warmup,
        trigger_delay_s=power.trigger_delay_s,
        workload_sync=power.workload_sync,
    )
    labels = {
        "experiment": "matched_lora_bp_vs_fo_power",
        "method": prepared.method,
        "seed": 0,
        "step": 1,
        "scope": "optimizer_update_only",
        "candidate_digest": prepared.provenance["candidate_digest"],
    }
    with experiment:
        records = experiment.run(1, labels=labels)
    if workload.calls != 1 or len(records) != 1 or workload.result is None:
        raise RuntimeError("SideCapture did not produce exactly one accepted optimizer trace")
    return dict(records[0]), workload.result


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"artifact path escapes trace root: {relative!r}")
    components = [
        root / Path(*candidate.parts[:index]) for index in range(1, len(candidate.parts) + 1)
    ]
    if any(part.is_symlink() for part in components):
        raise ValueError(f"artifact path traverses a symlink: {relative!r}")
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"artifact path escapes trace root: {relative!r}") from error
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {relative!r}")
    return resolved


def _trace_file_receipts(trace_root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in sorted(trace_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"trace artifact must not be a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(trace_root).as_posix()
        safe = _safe_relative_file(trace_root, relative)
        receipts.append(
            {
                "path": relative,
                "size_bytes": safe.stat().st_size,
                "sha256": _sha256_file(safe),
            }
        )
    return receipts


def _validate_trace_record(
    trace_root: Path,
    record: Mapping[str, Any],
    *,
    method: str,
    power: PowerCaptureConfig,
) -> dict[str, Any]:
    manifest_path = _safe_relative_file(trace_root, "manifest.json")
    manifest = _load_json_mapping(manifest_path)
    if manifest.get("schema_version") != SIDECAPTURE_DATASET_SCHEMA:
        raise RuntimeError("SideCapture manifest uses an unsupported dataset schema")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, Mapping):
        raise TypeError("SideCapture manifest lacks experiment metadata")
    request = experiment.get("request")
    resolved = experiment.get("resolved")
    if not isinstance(request, Mapping) or not isinstance(resolved, Mapping):
        raise TypeError("SideCapture manifest lacks request/resolved capture plans")
    expected_request = {
        "duration_s": 16.0,
        "sample_rate_hz": 18_750.0,
        "pretrigger_s": 0.0,
        "mode": "burst",
        "bits_per_sample": 12,
        "gain_db": 10.0,
        "channel": "power",
    }
    for name, expected in expected_request.items():
        if request.get(name) != expected:
            raise RuntimeError(f"SideCapture request {name} is not locked: {request.get(name)!r}")
    expected_resolved = {
        "mode": "burst",
        "sample_rate_hz": 18_750.0,
        "samples": 300_000,
        "duration_s": 16.0,
        "pretrigger_samples": 0,
        "raw_sample_rate_hz": 150_000_000.0,
        "decimation": 8_000,
        "bits_per_sample": 12,
    }
    for name, expected in expected_resolved.items():
        if resolved.get(name) != expected:
            raise RuntimeError(
                f"SideCapture resolved plan {name} is not locked: {resolved.get(name)!r}"
            )

    record_paths = sorted(trace_root.glob("records/shard_*/capture_*.json"))
    if len(record_paths) != 1:
        raise RuntimeError("SideCapture trace must contain exactly one committed record")
    disk_record = _load_json_mapping(record_paths[0])
    if disk_record.get("schema_version") != SIDECAPTURE_DATASET_SCHEMA:
        raise RuntimeError("SideCapture record uses an unsupported dataset schema")
    if disk_record.get("index") != 0 or disk_record.get("attempt") != 1:
        raise RuntimeError("SideCapture record must be capture index 0, attempt 1")
    if disk_record.get("primary_channel") != power.channel:
        raise RuntimeError("ChipWhisperer must remain the primary power channel")
    if disk_record.get("health", {}).get("ok") is not True:
        raise RuntimeError("SideCapture committed record is not healthy")
    labels = disk_record.get("labels")
    if not isinstance(labels, Mapping) or labels.get("method") != method:
        raise RuntimeError("SideCapture labels do not identify the requested method")
    channels = disk_record.get("channels")
    if not isinstance(channels, Mapping) or power.channel not in channels:
        raise RuntimeError("SideCapture record has no ChipWhisperer power channel")
    nvml_host_span_seconds: float | None = None
    if power.nvml_auxiliary:
        nvml_power = channels.get("nvml.power_w")
        if not isinstance(nvml_power, Mapping):
            raise RuntimeError("SideCapture record has no timestamped NVML power channel")
        timestamp_path = _safe_relative_file(
            trace_root,
            str(nvml_power.get("timestamps_path")),
        )
        import numpy as np

        timestamps = np.load(timestamp_path, allow_pickle=False)
        if (
            timestamps.ndim != 1
            or timestamps.size < 2
            or timestamps.dtype.kind not in {"i", "u"}
            or np.any(np.diff(timestamps.astype(np.int64)) <= 0)
        ):
            raise RuntimeError("NVML timestamp channel is malformed or non-monotonic")
        nvml_host_span_seconds = float((int(timestamps[-1]) - int(timestamps[0])) / 1e9)
        # SideCapture 0.1.1 trusts the ChipWhisperer decimation write.  This
        # independent host-clock gate catches a silently wrapped/clamped ADC
        # divider before an apparently healthy but time-compressed trace can
        # be accepted.
        if nvml_host_span_seconds < 15.5:
            raise RuntimeError(
                "ChipWhisperer capture completed too early for the locked 16-second plan: "
                f"host span was {nvml_host_span_seconds:.6f} seconds"
            )
    primary = channels[power.channel]
    if not isinstance(primary, Mapping):
        raise TypeError("SideCapture primary channel descriptor is invalid")
    if (
        primary.get("unit") != "normalized_adc"
        or primary.get("metadata", {}).get("calibrated") is not False
    ):
        raise RuntimeError("ChipWhisperer channel must remain explicitly uncalibrated ADC data")
    if (
        primary.get("shape") != [300_000]
        or primary.get("sample_rate_hz") != 18_750.0
        or primary.get("metadata", {}).get("gain_db") != 10.0
        or primary.get("metadata", {}).get("bits_per_sample") != 12
    ):
        raise RuntimeError("stored ChipWhisperer channel differs from the locked 16s/18.75kHz plan")

    sampler_metadata = disk_record.get("sampler_metadata")
    if not isinstance(sampler_metadata, Mapping):
        raise TypeError("SideCapture record lacks sampler metadata")
    if power.nvml_auxiliary:
        if sampler_metadata.get("backend") != "composite":
            raise RuntimeError("NVML-enabled trace must use a CompositeSampler")
        primary_sampler = sampler_metadata.get("primary")
        auxiliaries = sampler_metadata.get("auxiliaries")
        if (
            not isinstance(auxiliaries, list)
            or len(auxiliaries) != 1
            or auxiliaries[0].get("backend") != "nvml"
            or auxiliaries[0].get("gpu_index") != 0
            or auxiliaries[0].get("interval_s") != 0.01
        ):
            raise RuntimeError("CompositeSampler does not contain the locked NVML auxiliary")
    else:
        primary_sampler = sampler_metadata
    if not isinstance(primary_sampler, Mapping):
        raise TypeError("SideCapture record lacks primary ChipWhisperer metadata")
    if (
        primary_sampler.get("backend") != "chipwhisperer"
        or primary_sampler.get("serial_number") != power.serial_number
        or primary_sampler.get("product_id") != "0xace6"
        or primary_sampler.get("clock_hz") != 150_000_000.0
        or primary_sampler.get("safe_memory_fraction") != 1.0
        or primary_sampler.get("resolved") != resolved
    ):
        raise RuntimeError("primary sampler metadata differs from the locked Husky plan")
    error_controls = primary_sampler.get("adc_error_controls")
    if not isinstance(error_controls, Mapping) or (
        error_controls.get("lo_gain_errors_disabled_readback") is not True
        or error_controls.get("sidecapture_adc_clipping_validator_enabled") is not True
    ):
        raise RuntimeError("Husky ADC error-control readback is missing or unsafe")
    if error_controls.get("clip_errors_disabled_supported") is True and (
        error_controls.get("clip_errors_disabled_readback") is not False
    ):
        raise RuntimeError("Husky hardware clipping errors were disabled")

    manifest_sampler = experiment.get("sampler")
    if manifest_sampler != sampler_metadata:
        raise RuntimeError("manifest and record sampler metadata differ")
    primary_path = _safe_relative_file(trace_root, str(primary["path"]))
    annotations_descriptor = disk_record.get("annotations")
    if not isinstance(annotations_descriptor, Mapping):
        raise TypeError("SideCapture annotations descriptor is invalid")
    annotations_path = _safe_relative_file(
        trace_root,
        str(annotations_descriptor["path"]),
    )
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise TypeError("SideCapture annotations artifact must contain a list")
    annotation_pairs = {(row.get("name"), row.get("clock_domain")) for row in annotations}
    required = {
        (f"{method}.optimizer.ready", "host_monotonic"),
        (f"{method}.optimizer.host", "host_monotonic"),
        (f"{method}.optimizer.cuda", "cuda"),
        (f"{method}.optimizer.complete", "host_monotonic"),
    }
    if not required.issubset(annotation_pairs):
        raise RuntimeError("SideCapture record lacks synchronized host/CUDA optimizer annotations")
    cuda_rows = [row for row in annotations if row.get("name") == f"{method}.optimizer.cuda"]
    if len(cuda_rows) != 1 or cuda_rows[0].get("sync") != "both":
        raise RuntimeError("optimizer CUDA annotation must synchronize both boundaries")
    artifacts = disk_record.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "matched_candidate",
        "optimizer_result",
    }:
        raise RuntimeError("SideCapture record lacks the two bound result artifacts")
    for descriptor in artifacts.values():
        if not isinstance(descriptor, Mapping):
            raise TypeError("SideCapture artifact descriptor is invalid")
        _safe_relative_file(trace_root, str(descriptor["path"]))
    # The returned in-memory record is also checked where scalar comparison is stable.
    if record.get("index") != 0 or record.get("attempt") != 1:
        raise RuntimeError("in-memory SideCapture record differs from its commit marker")
    return {
        "manifest_path": "manifest.json",
        "manifest_sha256": _sha256_file(manifest_path),
        "requested_capture": dict(request),
        "resolved_capture": dict(resolved),
        "sampler_metadata": dict(sampler_metadata),
        "record_path": record_paths[0].relative_to(trace_root).as_posix(),
        "record_sha256": _sha256_file(record_paths[0]),
        "primary_channel_path": primary_path.relative_to(trace_root.resolve()).as_posix(),
        "primary_channel_sha256": _sha256_file(primary_path),
        "primary_channel": dict(primary),
        "annotations_path": annotations_path.relative_to(trace_root.resolve()).as_posix(),
        "annotations_sha256": _sha256_file(annotations_path),
        "annotation_count": len(annotations),
        "health": dict(disk_record["health"]),
        "nvml_capture_host_span_seconds": nvml_host_span_seconds,
    }


def _capture_child(
    config_path: Path,
    output: Path,
    *,
    method: str,
    seed: int,
    step: int,
    expected_source_commit: str,
) -> Path:
    if method not in CAPTURE_METHODS or seed != 0 or step != 1:
        raise ValueError("capture child is locked to bp_grpo/fo_npg, seed=0, step=1")
    source_commit, worktree = _verify_clean_source(expected_source_commit)
    _verify_committed_config(config_path, worktree, source_commit)
    _require_output_outside_worktree(output, worktree)
    _prepare_new_directory(output)
    matched, power = load_power_capture_config(config_path)
    config_sha256 = _sha256_file(config_path)
    prepared = _prepare_optimizer_capture(
        matched,
        method=method,
        seed=seed,
        step=step,
        output=output,
    )
    if prepared.provenance["initial_lora_digest"] != prepared.lora_digest():
        raise RuntimeError("prepared policy changed before SideCapture was armed")

    trace_root = output / "sidecapture"
    record, optimizer_result = _capture_optimizer_with_sidecapture(
        prepared,
        power=power,
        trace_root=trace_root,
    )
    trace = _validate_trace_record(
        trace_root,
        record,
        method=method,
        power=power,
    )
    files = _trace_file_receipts(trace_root)
    if not files:
        raise RuntimeError("SideCapture emitted no durable trace artifacts")
    lora_after = prepared.lora_digest()
    frozen_base_after = prepared.frozen_base_digest()
    if frozen_base_after != prepared.provenance["frozen_base_parameter_digest_before"]:
        raise RuntimeError("frozen base parameters changed during captured optimizer step")
    result = {
        "schema": POWER_CAPTURE_SCHEMA,
        "status": "complete",
        "method": method,
        "seed": seed,
        "step": step,
        "source_commit": source_commit,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "matched_config": matched.as_dict(),
        "power_capture_config": power.as_dict(),
        **prepared.provenance,
        "lora_digest_after": lora_after,
        "frozen_base_parameter_digest_after": frozen_base_after,
        "optimizer_result": optimizer_result,
        "backward_calls": optimizer_result["backward_calls"],
        "forward_calls": optimizer_result["forward_calls"],
        "policy_evaluations": optimizer_result["policy_evaluations"],
        "fo_backprop_module_imported": (
            BACKPROP_MODULE in sys.modules if method == "fo_npg" else None
        ),
        "process_isolation": "fresh_python_process_per_method",
        "runtime": prepared.runtime,
        "sidecapture": {
            "capture_count": 1,
            "accepted_count": 1,
            "attempt": 1,
            "trace_root": "sidecapture",
            **trace,
            "files": files,
        },
        "measurement_scope": {
            "captured_phase": "optimizer_update_only",
            "rollout_captured": False,
            "evaluation_run": False,
            "wandb_initialized": False,
            "chipwhisperer_quantity": "uncalibrated normalized ADC proxy",
            "chipwhisperer_total_gpu_watts_claimed": False,
            "nvml_auxiliary": power.nvml_auxiliary,
            "timing_alignment": (
                "projected host-trigger/CUDA-event annotations; no shared hardware clock"
            ),
        },
    }
    result["result_receipt_sha256"] = _json_digest(result)
    destination = output / "result.json"
    _write_json_exclusive(destination, result)
    return destination


def _verify_result_receipt(path: Path, *, method: str, source_commit: str) -> dict[str, Any]:
    result = _load_json_mapping(path)
    serialized_digest = result.pop("result_receipt_sha256", None)
    if not isinstance(serialized_digest, str) or _json_digest(result) != serialized_digest:
        raise RuntimeError(f"invalid child result receipt digest: {path}")
    result["result_receipt_sha256"] = serialized_digest
    if (
        result.get("schema") != POWER_CAPTURE_SCHEMA
        or result.get("status") != "complete"
        or result.get("method") != method
        or result.get("seed") != 0
        or result.get("step") != 1
        or result.get("source_commit") != source_commit
    ):
        raise RuntimeError(f"child result identity is invalid: {path}")
    sidecapture = result.get("sidecapture")
    if not isinstance(sidecapture, Mapping):
        raise TypeError("child result lacks SideCapture provenance")
    if (
        sidecapture.get("capture_count") != 1
        or sidecapture.get("accepted_count") != 1
        or sidecapture.get("attempt") != 1
    ):
        raise RuntimeError("child did not produce exactly one one-attempt trace")
    trace_root = path.parent / str(sidecapture.get("trace_root"))
    power_value = result.get("power_capture_config")
    if not isinstance(power_value, Mapping):
        raise TypeError("child result lacks the locked power capture config")
    power = PowerCaptureConfig.from_mapping(power_value)
    record_path = _safe_relative_file(trace_root, str(sidecapture.get("record_path")))
    disk_record = _load_json_mapping(record_path)
    independently_verified = _validate_trace_record(
        trace_root,
        disk_record,
        method=method,
        power=power,
    )
    for field in (
        "manifest_sha256",
        "record_sha256",
        "primary_channel_sha256",
        "annotations_sha256",
        "annotation_count",
    ):
        if independently_verified[field] != sidecapture.get(field):
            raise RuntimeError(f"parent hardware-plan verification differs: {field}")
    artifacts = disk_record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("SideCapture record lacks bound artifacts")
    candidate_descriptor = artifacts.get("matched_candidate")
    optimizer_descriptor = artifacts.get("optimizer_result")
    if not isinstance(candidate_descriptor, Mapping) or not isinstance(
        optimizer_descriptor,
        Mapping,
    ):
        raise TypeError("SideCapture record has invalid bound artifact descriptors")
    candidate_artifact = _load_json_mapping(
        _safe_relative_file(trace_root, str(candidate_descriptor.get("path")))
    )
    optimizer_artifact = _load_json_mapping(
        _safe_relative_file(trace_root, str(optimizer_descriptor.get("path")))
    )
    for field in _MATCHED_DIGEST_FIELDS:
        if candidate_artifact.get(field) != result.get(field):
            raise RuntimeError(f"candidate artifact differs from child result: {field}")
    if optimizer_artifact != result.get("optimizer_result"):
        raise RuntimeError("optimizer artifact differs from child result")
    receipts = sidecapture.get("files")
    if not isinstance(receipts, list) or not receipts:
        raise RuntimeError("child result has no trace file receipts")
    actual_paths = []
    for row in receipts:
        if not isinstance(row, Mapping):
            raise TypeError("trace file receipt is invalid")
        artifact = _safe_relative_file(trace_root, str(row.get("path")))
        if artifact.stat().st_size != row.get("size_bytes") or _sha256_file(artifact) != row.get(
            "sha256"
        ):
            raise RuntimeError(f"trace artifact digest mismatch: {artifact}")
        actual_paths.append(str(row["path"]))
    current_paths = [row["path"] for row in _trace_file_receipts(trace_root)]
    if actual_paths != current_paths:
        raise RuntimeError("trace file set differs from the sealed child receipt")
    return result


def _validate_matched_pair(
    bp: Mapping[str, Any],
    fo: Mapping[str, Any],
    *,
    source_commit: str,
) -> dict[str, str]:
    if bp.get("method") != "bp_grpo" or fo.get("method") != "fo_npg":
        raise RuntimeError("power pair must contain BP-GRPO then FO-NPG")
    if bp.get("source_commit") != source_commit or fo.get("source_commit") != source_commit:
        raise RuntimeError("power pair source commits differ")
    matched: dict[str, str] = {}
    for field in _MATCHED_DIGEST_FIELDS:
        left = bp.get(field)
        right = fo.get(field)
        if not isinstance(left, str) or left != right:
            raise RuntimeError(f"BP/FO matched-input digest differs: {field}")
        matched[field] = left
    if bp.get("config_sha256") != fo.get("config_sha256"):
        raise RuntimeError("BP/FO capture configs differ")
    for child in (bp, fo):
        candidate = child.get("candidate")
        if not isinstance(candidate, Mapping) or _json_digest(candidate) != child.get(
            "candidate_digest"
        ):
            raise RuntimeError("child candidate digest is internally inconsistent")
        example_ids = candidate.get("example_ids")
        if not isinstance(example_ids, list) or _json_digest(example_ids) != child.get(
            "prompt_example_ids_digest"
        ):
            raise RuntimeError("child prompt-example digest is internally inconsistent")
    if bp.get("initialization_path") != fo.get("initialization_path"):
        raise RuntimeError("BP/FO initialization artifact paths differ")
    initialization_path = Path(str(bp.get("initialization_path")))
    if not initialization_path.is_file() or _sha256_file(initialization_path) != bp.get(
        "initialization_file_sha256"
    ):
        raise RuntimeError("shared initialization file differs from the matched receipt")
    bp_backward = bp.get("optimizer_result", {}).get("backward_calls")
    fo_backward = fo.get("optimizer_result", {}).get("backward_calls")
    if isinstance(bp_backward, bool) or not isinstance(bp_backward, int) or bp_backward <= 0:
        raise RuntimeError("BP-GRPO result did not execute backward")
    if fo_backward != 0 or fo.get("fo_backprop_module_imported") is not False:
        raise RuntimeError("FO-NPG result violated strict no-backward isolation")
    if bp.get("frozen_base_parameter_digest_before") != bp.get(
        "frozen_base_parameter_digest_after"
    ) or fo.get("frozen_base_parameter_digest_before") != fo.get(
        "frozen_base_parameter_digest_after"
    ):
        raise RuntimeError("a captured update modified frozen base parameters")
    return matched


def _child_command(
    *,
    config_path: Path,
    output: Path,
    method: str,
    source_commit: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rl_no_backward.matched_lora_power_capture",
        "--config",
        str(config_path.resolve()),
        "--output",
        str(output.resolve()),
        "--child-method",
        method,
        "--seed",
        "0",
        "--step",
        "1",
        "--expected-source-commit",
        source_commit,
    ]


def run_power_capture_pair(config_path: str | Path, output_dir: str | Path) -> Path:
    """Launch the two fresh children and seal their matched-input comparison."""

    config = Path(config_path).resolve()
    # Validate before creating any output, without importing SideCapture.
    matched, power = load_power_capture_config(config)
    del matched
    source_commit, worktree = _verify_clean_source()
    _verify_committed_config(config, worktree, source_commit)
    output = Path(output_dir)
    _require_output_outside_worktree(output, worktree)
    _prepare_new_directory(output)
    copied_config = output / "power_capture_config.yaml"
    with copied_config.open("x", encoding="utf-8") as handle:
        handle.write(config.read_text(encoding="utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    plan = {
        "schema": "rl-no-backward-matched-lora-power-launch-plan-v1",
        "source_commit": source_commit,
        "methods": list(CAPTURE_METHODS),
        "capture_order": list(CAPTURE_METHODS),
        "seed": 0,
        "step": 1,
        "child_process_count": 2,
        "capture_count_per_child": 1,
        "max_attempts_per_child": 1,
        "config_sha256": _sha256_file(config),
        "power_capture_config": power.as_dict(),
    }
    plan["plan_receipt_sha256"] = _json_digest(plan)
    _write_json_exclusive(output / "launch_plan.json", plan)

    result_paths: dict[str, Path] = {}
    for method in CAPTURE_METHODS:
        child_output = output / method
        subprocess.run(
            _child_command(
                config_path=config,
                output=child_output,
                method=method,
                source_commit=source_commit,
            ),
            cwd=worktree,
            check=True,
        )
        result_paths[method] = child_output / "result.json"

    bp = _verify_result_receipt(
        result_paths["bp_grpo"],
        method="bp_grpo",
        source_commit=source_commit,
    )
    fo = _verify_result_receipt(
        result_paths["fo_npg"],
        method="fo_npg",
        source_commit=source_commit,
    )
    matched_digests = _validate_matched_pair(bp, fo, source_commit=source_commit)
    pair = {
        "schema": POWER_PAIR_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "methods": list(CAPTURE_METHODS),
        "capture_order": list(CAPTURE_METHODS),
        "seed": 0,
        "step": 1,
        "child_process_count": 2,
        "sidecapture_experiment_run_calls": 2,
        "total_capture_count": 2,
        "max_attempts_per_capture": 1,
        "matched_digests": matched_digests,
        "results": {
            method: {
                "path": result_paths[method].relative_to(output).as_posix(),
                "sidecapture_store": f"{method}/sidecapture",
                "sha256": _sha256_file(result_paths[method]),
                "result_receipt_sha256": (bp if method == "bp_grpo" else fo)[
                    "result_receipt_sha256"
                ],
            }
            for method in CAPTURE_METHODS
        },
        "comparison_valid": True,
        "measurement_claims": {
            "chipwhisperer_primary": True,
            "chipwhisperer_calibrated": False,
            "chipwhisperer_total_gpu_watts_claimed": False,
            "nvml_is_auxiliary": power.nvml_auxiliary,
        },
    }
    pair["pair_receipt_sha256"] = _json_digest(pair)
    destination = output / "power_pair.json"
    _write_json_exclusive(destination, pair)
    return destination


def _print_plan(config_path: Path) -> None:
    matched, power = load_power_capture_config(config_path)
    print(
        json.dumps(
            {
                "methods": list(CAPTURE_METHODS),
                "seed": 0,
                "step": 1,
                "model": matched.model_name,
                "model_revision": matched.model_revision,
                "responses": matched.responses_per_step,
                "sidecapture_experiment_run_calls": 2,
                "capture_count_per_child": 1,
                "max_attempts": power.max_attempts,
                "capture": power.as_dict(),
                "captured_phase": "optimizer_update_only",
                "evaluation": False,
                "wandb": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--child-method", choices=CAPTURE_METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args(argv)
    if args.print_plan:
        if any(
            value is not None
            for value in (
                args.output,
                args.child_method,
                args.seed,
                args.step,
                args.expected_source_commit,
            )
        ):
            parser.error("--print-plan cannot be combined with capture arguments")
        _print_plan(args.config)
        return
    if args.output is None:
        parser.error("capture requires --output")
    if args.child_method is not None:
        if args.seed is None or args.step is None or args.expected_source_commit is None:
            parser.error("--child-method requires --seed, --step, and --expected-source-commit")
        destination = _capture_child(
            args.config.resolve(),
            args.output,
            method=args.child_method,
            seed=args.seed,
            step=args.step,
            expected_source_commit=args.expected_source_commit,
        )
    else:
        if any(value is not None for value in (args.seed, args.step, args.expected_source_commit)):
            parser.error("seed/step/source-commit are internal child arguments")
        destination = run_power_capture_pair(args.config, args.output)
    print(destination)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "CAPTURE_METHODS",
    "CHIPWHISPERER_SERIAL",
    "POWER_CAPTURE_SCHEMA",
    "POWER_PAIR_SCHEMA",
    "PowerCaptureConfig",
    "load_power_capture_config",
    "main",
    "run_power_capture_pair",
]
