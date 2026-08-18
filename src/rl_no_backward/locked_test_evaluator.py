"""One-shot, answer-sealed GSM8K evaluation for selected matched-LoRA policies.

This module is deliberately separate from the learning runner.  Its planning
path reads only manifests, source indices, selection receipts, and checkpoint
metadata.  The pinned GSM8K rows (and therefore locked answers) are selected
only after an explicit authorization flag, a content-addressed plan check, and
exclusive creation of both an output directory and a benchmark-local
consumption ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .evaluation_manifest import (
    EvaluationSplitManifest,
    load_evaluation_split_manifest,
)
from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8KExample,
    exact_match_reward,
    extract_model_answer,
    format_prompt,
)
from .locked_source_sealing import (
    FINAL_PRIOR_METADATA_SOURCES,
    FINAL_PRIOR_QUESTION_SOURCES,
    LOCKED_SOURCE_INDEX_SCHEMA,
    PINNED_GSM8K_REVISION,
    LockedSourceIndexEntry,
    LockedSourceIndexReceipt,
    gsm8k_question_id,
    inspect_prior_question_evidence,
    load_locked_source_index_receipt,
    validate_locked_source_audit,
    write_locked_source_index_receipt,
)
from .standard_lora import (
    StandardLoRAConfig,
    assert_lora_frozen,
    attach_standard_lora,
    load_lora_state_dict,
    lora_parameter_layout,
    lora_state_digest,
)
from .vllm_lora_rollout import ReloadableLoRAGenerator, create_standard_lora_vllm_engine

LOCKED_EVALUATION_PLAN_SCHEMA = "rl-no-backward-locked-evaluation-plan-v1"
LOCKED_EVALUATION_RESULT_SCHEMA = "rl-no-backward-locked-evaluation-result-v1"
LOCKED_CONSUMPTION_SCHEMA = "rl-no-backward-locked-consumption-v1"
EXPECTED_LOCKED_TEST_COUNT = 679
EXPECTED_OFFICIAL_TEST_COUNT = 1_319
EXPECTED_EXCLUDED_TEST_COUNT = 384
EXPECTED_DEV_COUNT = 256
EXPECTED_METHODS = ("base", "bp_grpo", "fo_npg")
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_POLICY_PAIRS = (
    ("base", 0),
    ("bp_grpo", 0),
    ("fo_npg", 0),
    ("bp_grpo", 1),
    ("fo_npg", 1),
    ("bp_grpo", 2),
    ("fo_npg", 2),
)
EXPECTED_DATASET_REVISION = PINNED_GSM8K_REVISION
_HEX_DIGITS = frozenset("0123456789abcdef")


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


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{target} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{target} must contain a JSON object")
    return dict(value)


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


def _git_output(*args: str, worktree: Path | None = None) -> str | None:
    command = ["git"]
    if worktree is not None:
        command.extend(["-C", str(worktree)])
    command.extend(args)
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _reject_symlinked_input(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {cursor}")
    return absolute


def _selection_and_checkpoint_receipt(
    benchmark_output: Path,
    *,
    method: str,
    seed: int,
    expected_parameter_count: int,
) -> dict[str, Any]:
    label = f"gsm8k_{method}_seed{seed}"
    selection_path = benchmark_output / "selection" / f"{label}.json"
    checkpoint_path = benchmark_output / "checkpoints" / f"{label}.pt"
    raw_path = benchmark_output / "raw" / f"{label}.jsonl"
    learning_gate_path = benchmark_output / "learning_gate" / f"{label}.json"
    for artifact_name, artifact_path in (
        ("selection receipt", selection_path),
        ("selected checkpoint", checkpoint_path),
        ("raw training log", raw_path),
        ("learning-gate receipt", learning_gate_path),
    ):
        _require_regular_file_within(artifact_path, benchmark_output, name=artifact_name)
    selection = _load_json_mapping(selection_path)
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
    if set(selection) != expected_selection_keys:
        raise ValueError(f"{label} selection receipt has missing or unknown fields")
    if (
        selection["schema"] != "rl-no-backward-validation-selection-v1"
        or selection["method"] != method
        or selection["seed"] != seed
        or selection["selection_split"] != "development"
        or selection["selection_metric"] != "exact_match"
        or selection["tie_breaker"] != "latest_checkpoint"
        or Path(selection["checkpoint_path"]).resolve() != checkpoint_path.resolve()
    ):
        raise ValueError(f"{label} selection receipt violates the frozen selection contract")
    selected_step = selection["selected_step"]
    accuracy = selection["selection_val_accuracy"]
    if isinstance(selected_step, bool) or not isinstance(selected_step, int) or selected_step < 0:
        raise ValueError(f"{label} selected_step is invalid")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise ValueError(f"{label} selection accuracy is invalid")
    selected_digest = _sha256(
        selection["selected_lora_state_digest"], name="selected_lora_state_digest"
    )

    raw_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} raw log line {line_number} is invalid JSON") from error
        if not isinstance(record, Mapping):
            raise TypeError(f"{label} raw log line {line_number} is not an object")
        raw_records.append(dict(record))
    evaluations = [
        record
        for record in raw_records
        if record.get("kind") == "evaluation" and record.get("split") == "validation"
    ]
    if not evaluations:
        raise ValueError(f"{label} raw log contains no development evaluations")
    evaluation_steps: set[int] = set()
    scored_evaluations: list[tuple[float, int]] = []
    for record in evaluations:
        step = record.get("step")
        val_accuracy = record.get("val_accuracy")
        if (
            record.get("method") != method
            or record.get("seed") != seed
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step in evaluation_steps
            or isinstance(val_accuracy, bool)
            or not isinstance(val_accuracy, (int, float))
            or not math.isfinite(float(val_accuracy))
            or not 0.0 <= float(val_accuracy) <= 1.0
        ):
            raise ValueError(f"{label} raw development evaluation is malformed or duplicated")
        evaluation_steps.add(step)
        scored_evaluations.append((float(val_accuracy), step))
    recomputed_accuracy, recomputed_step = max(scored_evaluations)
    if recomputed_step != selected_step or recomputed_accuracy != float(accuracy):
        raise ValueError(
            f"{label} selection is not latest-step max development accuracy: "
            f"expected step={recomputed_step}, accuracy={recomputed_accuracy}"
        )
    if method == "base" and (selected_step != 0 or evaluation_steps != {0}):
        raise ValueError(f"{label} baseline must be the single non-updating step-zero policy")

    learning_gate = _load_json_mapping(learning_gate_path)
    structural_checks = learning_gate.get("hard_structural_checks")
    if not isinstance(structural_checks, Mapping) or any(
        value is not None and not isinstance(value, bool) for value in structural_checks.values()
    ):
        raise ValueError(f"{label} learning-gate structural checks are malformed")
    expected_failed_structural = sorted(
        name for name, value in structural_checks.items() if value is not None and value is not True
    )
    identity_is_valid = (
        learning_gate.get("role") == "non-updating baseline"
        and "method" not in learning_gate
        and "seed" not in learning_gate
        if method == "base"
        else learning_gate.get("method") == method and learning_gate.get("seed") == seed
    )
    if (
        learning_gate.get("schema") != "rl-no-backward-matched-learning-gate-v2"
        or not identity_is_valid
        or learning_gate.get("passed") is not True
        or learning_gate.get("hard_structural_checks_passed")
        is not (not expected_failed_structural)
        or learning_gate.get("failed_hard_structural_checks") != expected_failed_structural
        or expected_failed_structural
    ):
        raise ValueError(f"{label} learning-gate receipt is not structurally passing")
    best_dev_accuracy = learning_gate.get("best_dev_accuracy")
    if method != "base" and (
        isinstance(best_dev_accuracy, bool)
        or not isinstance(best_dev_accuracy, (int, float))
        or not math.isfinite(float(best_dev_accuracy))
        or float(best_dev_accuracy) != float(accuracy)
    ):
        raise ValueError(f"{label} learning gate differs from recomputed selected dev accuracy")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_checkpoint_keys = {
        "method",
        "seed",
        "selected_step",
        "selection_val_accuracy",
        "lora_state",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_checkpoint_keys:
        raise ValueError(f"{label} checkpoint has missing or unknown fields")
    checkpoint_accuracy = payload.get("selection_val_accuracy")
    if (
        payload["method"] != method
        or payload["seed"] != seed
        or payload["selected_step"] != selected_step
        or isinstance(checkpoint_accuracy, bool)
        or not isinstance(checkpoint_accuracy, (int, float))
        or not math.isfinite(float(checkpoint_accuracy))
        or float(checkpoint_accuracy) != float(accuracy)
    ):
        raise ValueError(f"{label} checkpoint differs from its selection receipt")
    state = payload["lora_state"]
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"{label} checkpoint has no LoRA tensor mapping")
    if any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state.items()
    ):
        raise TypeError(f"{label} checkpoint LoRA state is malformed")
    actual_digest = lora_state_digest(state)
    if actual_digest != selected_digest:
        raise ValueError(f"{label} checkpoint LoRA digest differs from selection")
    parameter_count = sum(tensor.numel() for tensor in state.values())
    if parameter_count != expected_parameter_count:
        raise ValueError(
            f"{label} checkpoint has {parameter_count} LoRA parameters; "
            f"expected {expected_parameter_count}"
        )
    layout = [
        {
            "name": name,
            "shape": list(state[name].shape),
            "dtype": str(state[name].dtype).removeprefix("torch."),
        }
        for name in sorted(state)
    ]
    return {
        "method": method,
        "seed": seed,
        "selected_step": selected_step,
        "selection_val_accuracy": float(accuracy),
        "selected_lora_state_digest": selected_digest,
        "lora_parameter_count": parameter_count,
        "lora_layout_sha256": _json_digest(layout),
        "selection_relpath": str(selection_path.relative_to(benchmark_output)),
        "selection_file_sha256": _file_digest(selection_path),
        "checkpoint_relpath": str(checkpoint_path.relative_to(benchmark_output)),
        "checkpoint_file_sha256": _file_digest(checkpoint_path),
        "raw_relpath": str(raw_path.relative_to(benchmark_output)),
        "raw_file_sha256": _file_digest(raw_path),
        "raw_development_evaluation_count": len(evaluations),
        "recomputed_selection_rule": "maximum val_accuracy, latest step on exact tie",
        "learning_gate_relpath": str(learning_gate_path.relative_to(benchmark_output)),
        "learning_gate_file_sha256": _file_digest(learning_gate_path),
    }


def _validate_training_metadata(
    metadata: Mapping[str, Any],
    manifest: EvaluationSplitManifest,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[int, ...]]:
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("benchmark metadata has no frozen config mapping")
    config = dict(config)
    methods = config.get("methods")
    seeds = config.get("seeds")
    if not isinstance(methods, list) or tuple(methods) != EXPECTED_METHODS:
        raise ValueError(f"locked evaluation requires methods {list(EXPECTED_METHODS)}")
    if not isinstance(seeds, list) or tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"locked evaluation requires exact seeds {list(EXPECTED_SEEDS)}")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("git_dirty") is not False
        or metadata.get("dataset_id") != GSM8K_DATASET_ID
        or metadata.get("dataset_config") != GSM8K_DATASET_CONFIG
        or metadata.get("dataset_revision") != config.get("dataset_revision")
        or config.get("dataset_revision") != EXPECTED_DATASET_REVISION
        or metadata.get("model_name") != config.get("model_name")
        or metadata.get("model_revision") != config.get("model_revision")
    ):
        raise ValueError("benchmark metadata does not bind its frozen model/dataset config")
    resolved_snapshot = metadata.get("resolved_model_snapshot")
    if not isinstance(resolved_snapshot, str) or not resolved_snapshot:
        raise ValueError("benchmark metadata has no resolved model snapshot")
    if (
        metadata.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or metadata.get("evaluation_manifest_dev_ids_sha256") != manifest.dev_ids_sha256
        or metadata.get("evaluation_manifest_locked_test_ids_sha256")
        != manifest.locked_test_ids_sha256
        or metadata.get("val_example_ids") != list(manifest.dev_example_ids)
        or metadata.get("excluded_test_example_ids") != list(manifest.excluded_test_example_ids)
    ):
        raise ValueError("benchmark metadata differs from the locked evaluation manifest")
    if (
        config.get("dev_size") != EXPECTED_DEV_COUNT
        or metadata.get("development_row_loading") != "Dataset.select(committed_dev_source_indices)"
    ):
        raise ValueError("training metadata does not certify the complete committed dev split")
    if (
        metadata.get("test_example_ids") != []
        or metadata.get("locked_test_rows_materialized") is not False
        or metadata.get("locked_test_accessed") is not False
        or metadata.get("locked_test_evaluated") is not False
        or config.get("run_test_evaluation") is not False
        or config.get("test_size") != 0
    ):
        raise ValueError("training artifacts do not prove that locked test remained sealed")
    if (
        config.get("dtype") != "bfloat16"
        or not str(config.get("device", "")).startswith("cuda")
        or config.get("attention_implementation") != "flash_attention_2"
        or config.get("rollout_backend") != "vllm_lora"
        or config.get("vllm_flash_attn_version") != 2
        or config.get("vllm_batch_invariant") is not False
        or config.get("vllm_enable_v1_multiprocessing") is not False
        or config.get("vllm_allow_insecure_serialization") is not False
    ):
        raise ValueError("locked evaluation requires the matched CUDA BF16/vLLM FA2 contract")
    if (
        metadata.get("evaluation_backend") != "same_vllm_0.22_standard_peft_lora_engine"
        or metadata.get("vllm_attention_config")
        != {"backend": "FLASH_ATTN", "flash_attn_version": 2}
        or metadata.get("hf_attention_implementation") != "flash_attention_2"
    ):
        raise ValueError("benchmark metadata does not certify the common vLLM FA2 backend")
    for name in (
        "max_prompt_tokens",
        "max_new_tokens",
        "eval_batch_size",
        "vllm_kv_cache_memory_bytes",
        "expected_lora_parameter_count",
    ):
        _positive_int(config.get(name), name=name)
    lora_config = StandardLoRAConfig.from_mapping(config.get("lora"))
    if (
        metadata.get("adapter_parameter_count") != config["expected_lora_parameter_count"]
        or metadata.get("lora_parameterization") != lora_config.as_dict()
    ):
        raise ValueError("benchmark metadata does not bind the frozen LoRA parameterization")
    source_commit = metadata.get("git_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in _HEX_DIGITS for character in source_commit)
    ):
        raise ValueError("benchmark metadata has no full clean source commit")
    return config, tuple(methods), tuple(seeds)


def _require_committed_file(path: Path, worktree: Path) -> None:
    _require_regular_file_within(path, worktree, name="locked input")
    try:
        relative = path.resolve().relative_to(worktree.resolve())
    except ValueError as error:
        raise ValueError(f"locked input must live in the evaluator worktree: {path}") from error
    tracked = _git_output("ls-files", "--error-unmatch", "--", str(relative), worktree=worktree)
    if tracked is None:
        raise RuntimeError(f"locked input is not committed: {relative}")
    changed = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--quiet", "HEAD", "--", str(relative)],
        check=False,
    )
    if changed.returncode != 0:
        raise RuntimeError(f"locked input differs from HEAD: {relative}")


def _require_regular_file_within(path: Path, root: Path, *, name: str) -> Path:
    """Reject missing, non-regular, symlinked, or root-escaping artifact paths."""

    root = root.resolve()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes its allowed root: {path}") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {cursor}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} does not exist: {path}") from None
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside its allowed root: {path}") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    return resolved


def _resolve_config_input_path(raw_path: Any, worktree: Path, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must not be absolute or contain '..'")
    return _require_regular_file_within(worktree / relative, worktree, name=name)


def _validate_bound_split_inputs(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    manifest: EvaluationSplitManifest,
    receipt: LockedSourceIndexReceipt,
    source_audit: Mapping[str, Any],
    manifest_target: Path,
    worktree: Path,
) -> dict[str, Any]:
    """Bind training dev/quarantine inputs to the answer-free source seal."""

    config_manifest = _resolve_config_input_path(
        config.get("evaluation_manifest"), worktree, name="config evaluation_manifest"
    )
    if config_manifest != manifest_target.resolve():
        raise ValueError("training config evaluation manifest differs from the authorized manifest")
    dev_path = _resolve_config_input_path(
        config.get("dev_source_index_receipt"),
        worktree,
        name="config dev_source_index_receipt",
    )
    quarantine_path = _resolve_config_input_path(
        config.get("touched_test_exclusions"),
        worktree,
        name="config touched_test_exclusions",
    )
    dev = _load_json_mapping(dev_path)
    if set(dev) != {
        "schema",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "evaluation_manifest_sha256",
        "official_test_count",
        "dev_count",
        "entries",
        "receipt_sha256",
    }:
        raise ValueError("committed development source receipt has unknown or missing fields")
    dev_payload = {key: value for key, value in dev.items() if key != "receipt_sha256"}
    dev_receipt_sha = dev.get("receipt_sha256")
    if not isinstance(dev_receipt_sha, str) or dev_receipt_sha != _json_digest(dev_payload):
        raise ValueError("committed development source receipt digest is invalid")
    entries = dev.get("entries")
    if not isinstance(entries, list) or any(
        not isinstance(entry, Mapping) or set(entry) != {"source_index", "example_id"}
        for entry in entries
    ):
        raise TypeError("committed development source entries are malformed")
    if any(
        not isinstance(entry["example_id"], str)
        or not entry["example_id"]
        or isinstance(entry["source_index"], bool)
        or not isinstance(entry["source_index"], int)
        or not 0 <= entry["source_index"] < EXPECTED_OFFICIAL_TEST_COUNT
        for entry in entries
    ):
        raise TypeError("committed development source entry values are malformed")
    dev_ids = tuple(entry["example_id"] for entry in entries)
    dev_indices = tuple(sorted(entry["source_index"] for entry in entries))
    if (
        dev.get("schema") != "rl-no-backward-gsm8k-dev-source-index-v1"
        or dev.get("dataset_id") != GSM8K_DATASET_ID
        or dev.get("dataset_config") != GSM8K_DATASET_CONFIG
        or dev.get("dataset_revision") != config["dataset_revision"]
        or dev.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or dev.get("official_test_count") != EXPECTED_OFFICIAL_TEST_COUNT
        or dev.get("dev_count") != EXPECTED_DEV_COUNT
        or dev_ids != manifest.dev_example_ids
        or dev_indices != receipt.dev_source_indices
    ):
        raise ValueError("development source receipt differs from manifest/sealed indices")
    access_sources = source_audit["access_sources"]
    dev_access = access_sources["development_source_receipt"]
    manifest_access = access_sources["manifest"]
    if (
        dev_access["file_sha256"] != _file_digest(dev_path)
        or dev_access["receipt_sha256"] != dev_receipt_sha
        or manifest_access["file_sha256"] != _file_digest(manifest_target)
        or metadata.get("dev_source_index_receipt_sha256") != dev_receipt_sha
        or metadata.get("dev_source_index_receipt_path") != config.get("dev_source_index_receipt")
        or metadata.get("evaluation_manifest_path") != config.get("evaluation_manifest")
    ):
        raise ValueError(
            "training metadata and sealing audit do not bind the same dev/manifest files"
        )
    quarantine = _load_json_mapping(quarantine_path)
    quarantine_access = access_sources["touched_test_quarantine"]
    if (
        quarantine.get("schema_version") != 1
        or quarantine.get("test_example_ids") != list(manifest.excluded_test_example_ids)
        or quarantine_access["file_sha256"] != _file_digest(quarantine_path)
        or quarantine_access["excluded_test_ids_sha256"] != manifest.excluded_test_ids_sha256
        or quarantine_access["evaluation_manifest_sha256"] != manifest.manifest_sha256
        or quarantine_access["row_count"] != EXPECTED_EXCLUDED_TEST_COUNT
    ):
        raise ValueError("committed touched-test quarantine differs from the manifest")
    return {
        "development_source_receipt_relpath": str(dev_path.relative_to(worktree)),
        "development_source_receipt_file_sha256": _file_digest(dev_path),
        "development_source_receipt_sha256": dev_receipt_sha,
        "development_source_indices_sha256": receipt.dev_source_indices_sha256,
        "development_ids_sha256": manifest.dev_ids_sha256,
        "quarantine_relpath": str(quarantine_path.relative_to(worktree)),
        "quarantine_file_sha256": _file_digest(quarantine_path),
        "excluded_test_ids_sha256": manifest.excluded_test_ids_sha256,
        "development_row_loading": "Dataset.select(committed_dev_source_indices)",
    }


def _validate_prior_question_evidence(
    source_audit: Mapping[str, Any],
    worktree: Path,
) -> tuple[dict[str, Any], ...]:
    sources = source_audit["access_sources"]["prior_exposure_question_samples"]
    receipts: list[dict[str, Any]] = []
    for source in sources:
        metadata_sha = source["metadata_file_sha256"]
        expected_relpath = FINAL_PRIOR_QUESTION_SOURCES.get(metadata_sha)
        expected_metadata_relpath = FINAL_PRIOR_METADATA_SOURCES.get(metadata_sha)
        if (
            expected_relpath is None
            or expected_metadata_relpath is None
            or source.get("committed_question_source") != expected_relpath
            or source.get("committed_metadata_source") != expected_metadata_relpath
        ):
            raise ValueError("sealing audit lacks the canonical committed prior-question evidence")
        metadata_path = _resolve_config_input_path(
            expected_metadata_relpath, worktree, name="committed prior metadata"
        )
        if _file_digest(metadata_path) != metadata_sha:
            raise ValueError("committed prior metadata differs from the sealing audit")
        evidence_path = _resolve_config_input_path(
            expected_relpath, worktree, name="committed prior-question evidence"
        )
        if _file_digest(evidence_path) != source["question_sample_file_sha256"]:
            raise ValueError("committed prior-question evidence differs from the sealing audit")
        inspected = inspect_prior_question_evidence(evidence_path)
        prior_metadata_ids = _load_json_mapping(metadata_path).get("test_example_ids")
        if (
            inspected["row_count"] != source["row_count"]
            or not isinstance(prior_metadata_ids, list)
            or tuple(prior_metadata_ids) != inspected["example_ids"]
            or inspected["question_projection_sha256"] != source["question_projection_sha256"]
        ):
            raise ValueError("prior-question evidence projection differs from the sealing audit")
        receipts.append(
            {
                "metadata_file_sha256": metadata_sha,
                "metadata_relpath": expected_metadata_relpath,
                "evidence_relpath": expected_relpath,
                "evidence_file_sha256": source["question_sample_file_sha256"],
                "question_projection_sha256": source["question_projection_sha256"],
                "row_count": source["row_count"],
                "access_scope": "example_id and question only; all other values lexically skipped",
            }
        )
    if tuple(receipt["metadata_file_sha256"] for receipt in receipts) != tuple(
        sorted(FINAL_PRIOR_QUESTION_SOURCES)
    ):
        # The manifest receipts are sorted by metadata digest, and the seal
        # must use the same stable order.
        receipts.sort(key=lambda receipt: receipt["metadata_file_sha256"])
    if {receipt["metadata_file_sha256"] for receipt in receipts} != set(
        FINAL_PRIOR_QUESTION_SOURCES
    ):
        raise ValueError("prior-question evidence does not cover both quarantine sources")
    return tuple(receipts)


def _selected_policy_pairs(
    methods: Sequence[str],
    seeds: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    """Mirror training: one shared base plus BP/FO for every experimental seed."""

    if tuple(methods) != EXPECTED_METHODS or tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("selected policy pairs require the exact final methods/seeds")
    return EXPECTED_POLICY_PAIRS


def build_locked_evaluation_plan(
    benchmark_output: str | Path,
    manifest_path: str | Path,
    source_index_receipt_path: str | Path,
    *,
    enforce_committed_inputs: bool = True,
) -> dict[str, Any]:
    """Preflight a frozen evaluation without importing datasets or reading answers."""

    benchmark = _reject_symlinked_input(benchmark_output, name="benchmark output").resolve()
    manifest_target = _reject_symlinked_input(manifest_path, name="evaluation manifest").resolve()
    source_target = _reject_symlinked_input(
        source_index_receipt_path, name="source-index receipt"
    ).resolve()
    source_audit_target = _reject_symlinked_input(
        source_target.with_suffix(".audit.json"), name="source-index sealing audit"
    ).resolve()
    manifest = load_evaluation_split_manifest(manifest_target)
    manifest_counts = (
        manifest.official_test_count,
        manifest.excluded_test_count,
        manifest.dev_count,
        manifest.locked_test_count,
    )
    expected_counts = (
        EXPECTED_OFFICIAL_TEST_COUNT,
        EXPECTED_EXCLUDED_TEST_COUNT,
        EXPECTED_DEV_COUNT,
        EXPECTED_LOCKED_TEST_COUNT,
    )
    if manifest_counts != expected_counts:
        raise ValueError("locked evaluation requires exact 1319/384/256/679 manifest counts")
    metadata_path = benchmark / "metadata.json"
    validation_path = benchmark / "artifact_validation.json"
    _require_regular_file_within(metadata_path, benchmark, name="benchmark metadata")
    _require_regular_file_within(validation_path, benchmark, name="artifact validation")
    metadata = _load_json_mapping(metadata_path)
    config, methods, seeds = _validate_training_metadata(metadata, manifest)
    receipt = load_locked_source_index_receipt(
        source_target,
        manifest,
        dataset_revision=str(config["dataset_revision"]),
    )
    source_audit = _load_json_mapping(source_audit_target)
    validate_locked_source_audit(receipt, source_audit, manifest)
    validation = _load_json_mapping(validation_path)
    policy_pairs = _selected_policy_pairs(methods, seeds)
    expected_runs = len(policy_pairs)
    if (
        validation.get("passed") is not True
        or validation.get("status") != "complete"
        or not isinstance(validation.get("output_dir"), str)
        or Path(validation["output_dir"]).resolve() != benchmark
        or validation.get("expected_runs") != expected_runs
        or validation.get("validated_runs") != expected_runs
        or validation.get("errors") != []
        or validation.get("incomplete") != []
    ):
        raise ValueError("benchmark artifact validation is not complete and passing")

    evaluator_commit = _git_output("rev-parse", "HEAD")
    evaluator_status = _git_output("status", "--porcelain")
    worktree_value = _git_output("rev-parse", "--show-toplevel")
    if (
        evaluator_commit is None
        or len(evaluator_commit) != 40
        or evaluator_status is None
        or evaluator_status
        or worktree_value is None
    ):
        raise RuntimeError("locked evaluation planning requires a clean Git worktree")
    worktree = Path(worktree_value).resolve()
    split_input_receipt = _validate_bound_split_inputs(
        config,
        metadata,
        manifest,
        receipt,
        source_audit,
        manifest_target,
        worktree,
    )
    prior_question_evidence = _validate_prior_question_evidence(source_audit, worktree)
    if enforce_committed_inputs:
        _require_committed_file(manifest_target, worktree)
        _require_committed_file(source_target, worktree)
        _require_committed_file(source_audit_target, worktree)
        _require_committed_file(
            worktree / split_input_receipt["development_source_receipt_relpath"], worktree
        )
        _require_committed_file(worktree / split_input_receipt["quarantine_relpath"], worktree)
        for evidence in prior_question_evidence:
            _require_committed_file(worktree / evidence["metadata_relpath"], worktree)
            _require_committed_file(worktree / evidence["evidence_relpath"], worktree)

    checkpoints = [
        _selection_and_checkpoint_receipt(
            benchmark,
            method=method,
            seed=seed,
            expected_parameter_count=int(config["expected_lora_parameter_count"]),
        )
        for method, seed in policy_pairs
    ]
    layout_digests = {checkpoint["lora_layout_sha256"] for checkpoint in checkpoints}
    if len(layout_digests) != 1:
        raise ValueError("selected checkpoints do not share an exact LoRA tensor layout")
    checkpoint_set_digest = _json_digest(checkpoints)
    plan_without_digest = {
        "schema": LOCKED_EVALUATION_PLAN_SCHEMA,
        "training_source_commit": metadata["git_commit"],
        "evaluator_source_commit": evaluator_commit,
        "benchmark_metadata_sha256": _file_digest(metadata_path),
        "artifact_validation_sha256": _file_digest(validation_path),
        "model": {
            "id": config["model_name"],
            "revision": config["model_revision"],
            "resolved_snapshot": metadata["resolved_model_snapshot"],
            "dtype": config["dtype"],
            "attention_implementation": config["attention_implementation"],
        },
        "dataset": {
            "id": GSM8K_DATASET_ID,
            "config": GSM8K_DATASET_CONFIG,
            "revision": config["dataset_revision"],
            "official_test_count": manifest.official_test_count,
            "locked_test_count": manifest.locked_test_count,
        },
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "locked_test_ids_sha256": manifest.locked_test_ids_sha256,
        "manifest_file_sha256": _file_digest(manifest_target),
        "source_index_receipt_sha256": receipt.receipt_sha256,
        "source_index_file_sha256": _file_digest(source_target),
        "source_sealing_audit_file_sha256": _file_digest(source_audit_target),
        "official_question_ids_sha256": receipt.official_question_ids_sha256,
        "locked_source_indices_sha256": receipt.locked_source_indices_sha256,
        "source_sealing_audit_sha256": receipt.sealing_audit_sha256,
        "training_evaluation_inputs": split_input_receipt,
        "prior_question_evidence": list(prior_question_evidence),
        "row_loading": (
            "Dataset.select(committed_locked_source_indices), then authorized opaque-ID reorder"
        ),
        "selection_split": "development",
        "selection_metric": "exact_match",
        "generation": {
            "backend": "vllm_0.22_standard_peft_lora_load_inplace",
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
            "greedy": True,
            "max_prompt_tokens": config["max_prompt_tokens"],
            "max_new_tokens": config["max_new_tokens"],
            "eval_batch_size": config["eval_batch_size"],
        },
        "methods": list(methods),
        "seeds": list(seeds),
        "base_policy_semantics": {
            "evaluated_once": True,
            "seed": seeds[0],
            "role": "shared non-updating reference for every trained seed",
        },
        "checkpoints": checkpoints,
        "checkpoint_set_sha256": checkpoint_set_digest,
    }
    return {**plan_without_digest, "plan_sha256": _json_digest(plan_without_digest)}


def _default_dataset_loader(**kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - H100 runtime only
        raise RuntimeError("locked evaluation requires the pinned datasets runtime") from error
    return load_dataset(**kwargs)


def load_locked_examples(
    manifest: EvaluationSplitManifest,
    receipt: LockedSourceIndexReceipt,
    *,
    dataset_revision: str,
    dataset_loader: Callable[..., Any] | None = None,
) -> tuple[GSM8KExample, ...]:
    """Materialize exactly the committed locked rows through ``Dataset.select``."""

    receipt.validate_manifest(manifest, dataset_revision=dataset_revision)
    loader = dataset_loader or _default_dataset_loader
    official_test = loader(
        path=GSM8K_DATASET_ID,
        name=GSM8K_DATASET_CONFIG,
        split="test",
        revision=dataset_revision,
    )
    if len(official_test) != manifest.official_test_count:
        raise ValueError("pinned GSM8K test row count differs from the committed manifest")
    indices = [entry.source_index for entry in receipt.entries]
    selected_rows = official_test.select(indices)
    examples: list[GSM8KExample] = []
    for entry, row in zip(receipt.entries, selected_rows, strict=True):
        if not isinstance(row, Mapping) or not {"question", "answer"} <= set(row):
            raise TypeError("selected locked GSM8K row is malformed")
        if gsm8k_question_id(row["question"]) != entry.question_id:
            raise ValueError("selected locked row does not match its sealed question ID")
        example = GSM8KExample(
            question=row["question"],
            answer=row["answer"],
            split="test",
            source_index=entry.source_index,
        )
        examples.append(example)
    by_id = {example.example_id: example for example in examples}
    if len(by_id) != len(examples) or set(by_id) != set(manifest.locked_test_example_ids):
        raise RuntimeError("authorized locked rows do not exactly match the manifest ID set")
    return tuple(by_id[example_id] for example_id in manifest.locked_test_example_ids)


def _format_prompts(tokenizer: Any, examples: Sequence[GSM8KExample]) -> tuple[str, ...]:
    def formatter(messages: Sequence[Mapping[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )

    return tuple(format_prompt(example.question, formatter) for example in examples)


def _prompt_token_ids(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_prompt_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]
        row = tuple(int(token_id) for token_id in encoded)
        if not row:
            raise ValueError("locked evaluation prompt encoded to no tokens")
        rows.append(row)
    return tuple(rows)


def evaluate_locked_policy(
    tokenizer: Any,
    generator: ReloadableLoRAGenerator,
    examples: Sequence[GSM8KExample],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    eval_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one already synchronized policy and return auditable samples."""

    if not examples:
        raise ValueError("locked evaluation examples must not be empty")
    prompts = _format_prompts(tokenizer, examples)
    samples: list[dict[str, Any]] = []
    finish_counts: dict[str, int] = {}
    expected_policy_version = generator.policy_version
    if expected_policy_version is None:
        raise RuntimeError("LoRA policy must be synchronized before locked evaluation")
    for start in range(0, len(examples), eval_batch_size):
        batch_examples = examples[start : start + eval_batch_size]
        prompt_ids = _prompt_token_ids(
            tokenizer,
            prompts[start : start + eval_batch_size],
            max_prompt_tokens=max_prompt_tokens,
        )
        generated = generator.generate_greedy(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=(int(tokenizer.eos_token_id),),
        )
        if generated.policy_version != expected_policy_version:
            raise RuntimeError("vLLM returned a different policy version during locked evaluation")
        for example, response_ids, finish_reason in zip(
            batch_examples,
            generated.response_token_ids,
            generated.finish_reasons,
            strict=True,
        ):
            completion = tokenizer.decode(
                list(response_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            correct = bool(exact_match_reward(completion, example.canonical_answer))
            finish_key = "none" if finish_reason is None else str(finish_reason)
            finish_counts[finish_key] = finish_counts.get(finish_key, 0) + 1
            samples.append(
                {
                    "example_id": example.example_id,
                    "source_index": example.source_index,
                    "completion": completion,
                    "predicted_answer": extract_model_answer(completion),
                    "correct": correct,
                    "response_tokens": len(response_ids),
                    "finish_reason": finish_reason,
                }
            )
    correct_count = sum(int(sample["correct"]) for sample in samples)
    total = len(samples)
    mean_tokens = sum(int(sample["response_tokens"]) for sample in samples) / total
    return {
        "accuracy": correct_count / total,
        "correct_count": correct_count,
        "example_count": total,
        "mean_response_tokens": mean_tokens,
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "truncation_fraction": finish_counts.get("length", 0) / total,
        "eos_terminated_fraction": (finish_counts.get("stop", 0) + finish_counts.get("eos", 0))
        / total,
    }, samples


def _load_evaluation_model(config: Mapping[str, Any], *, seed: int) -> tuple[nn.Module, Any, str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = snapshot_download(config["model_name"], revision=config["model_revision"])
    if Path(snapshot).name != config["model_revision"]:
        raise RuntimeError("resolved model snapshot does not match the pinned model revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
    )
    lora_config = StandardLoRAConfig.from_mapping(config["lora"])
    model = attach_standard_lora(base_model, lora_config, initialization_seed=seed)
    model.to(torch.device(config["device"]))
    model.eval()
    if lora_parameter_layout(model).parameter_count != config["expected_lora_parameter_count"]:
        raise RuntimeError("evaluation model LoRA layout differs from the frozen benchmark")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    assert_lora_frozen(model)
    return model, tokenizer, snapshot


def _checkpoint_state(
    benchmark_output: Path, checkpoint: Mapping[str, Any]
) -> Mapping[str, Tensor]:
    path = benchmark_output / str(checkpoint["checkpoint_relpath"])
    if _file_digest(path) != checkpoint["checkpoint_file_sha256"]:
        raise RuntimeError("checkpoint changed after the locked evaluation plan was frozen")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("lora_state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise TypeError("selected checkpoint has no LoRA state")
    if lora_state_digest(state) != checkpoint["selected_lora_state_digest"]:
        raise RuntimeError("selected checkpoint state digest changed after preflight")
    return state


def _runtime_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "datasets", "vllm", "flash-attn"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def _require_output_outside_worktree(output: Path) -> None:
    worktree_value = _git_output("rev-parse", "--show-toplevel")
    if worktree_value is None:
        raise RuntimeError("could not resolve evaluator worktree")
    try:
        output.resolve().relative_to(Path(worktree_value).resolve())
    except ValueError:
        return
    raise ValueError("locked evaluation output must be outside the Git worktree")


def _reserve_output(output: Path) -> None:
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"refusing to reuse locked evaluation output: {output}") from None


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            line = _canonical_json_bytes(dict(row)) + b"\n"
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def run_locked_test_evaluation(
    benchmark_output: str | Path,
    manifest_path: str | Path,
    source_index_receipt_path: str | Path,
    output_dir: str | Path,
    *,
    expected_plan_sha256: str,
    authorize_locked_test_once: bool,
    dataset_loader: Callable[..., Any] | None = None,
) -> Path:
    """Consume the sealed split once and evaluate every selected checkpoint."""

    if not authorize_locked_test_once:
        raise PermissionError("locked answers require --authorize-locked-test-once")
    expected_digest = _sha256(expected_plan_sha256, name="expected_plan_sha256")
    benchmark = Path(benchmark_output).resolve()
    output = Path(output_dir).resolve()
    ledger = benchmark / "locked_test_consumed.json"
    if output.exists():
        raise FileExistsError(f"refusing to reuse locked evaluation output: {output}")
    if ledger.exists():
        raise FileExistsError(f"locked test was already consumed for this benchmark: {ledger}")
    plan = build_locked_evaluation_plan(
        benchmark,
        manifest_path,
        source_index_receipt_path,
    )
    if plan["plan_sha256"] != expected_digest:
        raise ValueError(
            "authorized plan digest differs from current manifests/checkpoints; "
            "run --print-plan and review the frozen inputs again"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("locked matched-LoRA evaluation requires CUDA")
    _require_output_outside_worktree(output)
    _require_output_outside_worktree(ledger)
    metadata = _load_json_mapping(benchmark / "metadata.json")
    if _file_digest(benchmark / "metadata.json") != plan["benchmark_metadata_sha256"]:
        raise RuntimeError("benchmark metadata changed after the evaluation plan was frozen")
    config = dict(metadata["config"])
    first_seed = int(plan["seeds"][0])

    # Resolve both inference stacks before consuming the sealed data.  A
    # dependency/engine failure therefore cannot spend the one-shot split.
    model, tokenizer, resolved_snapshot = _load_evaluation_model(config, seed=first_seed)
    engine = create_standard_lora_vllm_engine(
        config["model_name"],
        revision=config["model_revision"],
        dtype=config["dtype"],
        max_model_len=config["max_prompt_tokens"] + config["max_new_tokens"],
        max_lora_rank=config["lora"]["rank"],
        kv_cache_memory_bytes=config["vllm_kv_cache_memory_bytes"],
        enforce_eager=config["vllm_enforce_eager"],
        flash_attn_version=config["vllm_flash_attn_version"],
        max_num_seqs=config["eval_batch_size"],
        seed=first_seed,
    )
    generator = ReloadableLoRAGenerator(engine, lora_name="locked-selected-policy")
    checkpoint_states = [
        _checkpoint_state(benchmark, checkpoint) for checkpoint in plan["checkpoints"]
    ]
    _reserve_output(output)
    _write_json_exclusive(output / "plan.json", plan)
    consumed_at = time.time()
    ledger_payload = {
        "schema": LOCKED_CONSUMPTION_SCHEMA,
        "status": "started",
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_set_sha256": plan["checkpoint_set_sha256"],
        "evaluation_manifest_sha256": plan["evaluation_manifest_sha256"],
        "locked_test_ids_sha256": plan["locked_test_ids_sha256"],
        "output": str(output),
        "started_unix_seconds": consumed_at,
    }
    try:
        _write_json_exclusive(ledger, ledger_payload)
    except Exception:
        _write_json_exclusive(
            output / "FAILED.json",
            {"stage": "consumption-ledger", "plan_sha256": plan["plan_sha256"]},
        )
        raise

    try:
        manifest = load_evaluation_split_manifest(manifest_path)
        source_receipt = load_locked_source_index_receipt(
            source_index_receipt_path,
            manifest,
            dataset_revision=config["dataset_revision"],
        )
        if (
            manifest.manifest_sha256 != plan["evaluation_manifest_sha256"]
            or source_receipt.receipt_sha256 != plan["source_index_receipt_sha256"]
        ):
            raise RuntimeError("locked manifest/source receipt changed after authorization")
        examples = load_locked_examples(
            manifest,
            source_receipt,
            dataset_revision=config["dataset_revision"],
            dataset_loader=dataset_loader,
        )
        if len(examples) != EXPECTED_LOCKED_TEST_COUNT:
            raise RuntimeError("authorized loader did not materialize all 679 locked examples")
        torch.cuda.reset_peak_memory_stats()
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        reload_probe_prompt = _prompt_token_ids(
            tokenizer,
            _format_prompts(tokenizer, examples[:1]),
            max_prompt_tokens=config["max_prompt_tokens"],
        )[0]
        for version, (checkpoint, state) in enumerate(
            zip(plan["checkpoints"], checkpoint_states, strict=True), start=1
        ):
            load_lora_state_dict(model, state)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            assert_lora_frozen(model)
            live_digest = lora_state_digest(model)
            if live_digest != checkpoint["selected_lora_state_digest"]:
                raise RuntimeError("live evaluation policy differs from selected checkpoint")
            sync_started = time.perf_counter()
            reload_receipt = generator.sync(
                model,
                output / "vllm_lora_exports",
                version=version,
                policy_version=(
                    f"locked/{checkpoint['method']}/seed={checkpoint['seed']}"
                    f"/selected-step={checkpoint['selected_step']}"
                ),
            )
            torch.cuda.synchronize()
            sync_seconds = time.perf_counter() - sync_started
            reload_probe = generator.probe_next_token(reload_probe_prompt)
            if reload_probe.state_digest != live_digest:
                raise RuntimeError("vLLM reload probe differs from the selected LoRA state")
            evaluation_started = time.perf_counter()
            metrics, samples = evaluate_locked_policy(
                tokenizer,
                generator,
                examples,
                max_prompt_tokens=config["max_prompt_tokens"],
                max_new_tokens=config["max_new_tokens"],
                eval_batch_size=config["eval_batch_size"],
            )
            torch.cuda.synchronize()
            evaluation_seconds = time.perf_counter() - evaluation_started
            label = f"gsm8k_{checkpoint['method']}_seed{checkpoint['seed']}"
            samples_relpath = f"samples/{label}.jsonl"
            sample_digest = _write_jsonl_exclusive(output / samples_relpath, samples)
            if (
                tuple(sample["example_id"] for sample in samples)
                != manifest.locked_test_example_ids
            ):
                raise RuntimeError("persisted sample order differs from locked manifest")
            result = {
                "method": checkpoint["method"],
                "seed": checkpoint["seed"],
                "selected_step": checkpoint["selected_step"],
                "selection_val_accuracy": checkpoint["selection_val_accuracy"],
                "selected_lora_state_digest": live_digest,
                "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
                "policy_sync_seconds": sync_seconds,
                "evaluation_seconds": evaluation_seconds,
                **metrics,
                "samples_relpath": samples_relpath,
                "samples_sha256": sample_digest,
                "prediction_sha256": _json_digest(
                    [
                        {
                            "example_id": sample["example_id"],
                            "predicted_answer": sample["predicted_answer"],
                            "correct": sample["correct"],
                        }
                        for sample in samples
                    ]
                ),
                "vllm_lora_reload_receipt": asdict(reload_receipt),
                "vllm_policy_probe": asdict(reload_probe),
            }
            results.append(result)

        receipt_without_digest = {
            "schema": LOCKED_EVALUATION_RESULT_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "checkpoint_set_sha256": plan["checkpoint_set_sha256"],
            "evaluation_manifest_sha256": manifest.manifest_sha256,
            "locked_test_ids_sha256": manifest.locked_test_ids_sha256,
            "source_index_receipt_sha256": source_receipt.receipt_sha256,
            "consumption_ledger_sha256": _file_digest(ledger),
            "resolved_model_snapshot": resolved_snapshot,
            "locked_rows_materialized": True,
            "locked_rows_materialized_after_authorization": True,
            "locked_test_example_count": len(examples),
            "total_wall_time_seconds": time.perf_counter() - started,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "results": results,
            "runtime": _runtime_metadata(),
        }
        receipt = {
            **receipt_without_digest,
            "result_receipt_sha256": _json_digest(receipt_without_digest),
        }
        _write_json_exclusive(output / "results.json", receipt)
        _write_json_exclusive(
            output / "COMPLETE.json",
            {
                "schema": LOCKED_EVALUATION_RESULT_SCHEMA,
                "status": "complete",
                "plan_sha256": plan["plan_sha256"],
                "result_receipt_sha256": receipt["result_receipt_sha256"],
            },
        )
        return output / "results.json"
    except Exception as error:
        try:
            _write_json_exclusive(
                output / "FAILED.json",
                {
                    "schema": LOCKED_EVALUATION_RESULT_SCHEMA,
                    "status": "failed-after-consumption",
                    "plan_sha256": plan["plan_sha256"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except FileExistsError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-index-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--authorize-locked-test-once", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    if args.print_plan:
        if args.authorize_locked_test_once:
            parser.error("--print-plan must not be combined with authorization")
        plan = build_locked_evaluation_plan(
            args.benchmark_output,
            args.manifest,
            args.source_index_receipt,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not args.authorize_locked_test_once:
        parser.error("locked evaluation requires --authorize-locked-test-once")
    if args.output is None or args.expected_plan_sha256 is None:
        parser.error("authorized evaluation requires --output and --expected-plan-sha256")
    result = run_locked_test_evaluation(
        args.benchmark_output,
        args.manifest,
        args.source_index_receipt,
        args.output,
        expected_plan_sha256=args.expected_plan_sha256,
        authorize_locked_test_once=True,
    )
    print(result)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "EXPECTED_LOCKED_TEST_COUNT",
    "LOCKED_EVALUATION_PLAN_SCHEMA",
    "LOCKED_EVALUATION_RESULT_SCHEMA",
    "LOCKED_SOURCE_INDEX_SCHEMA",
    "LockedSourceIndexEntry",
    "LockedSourceIndexReceipt",
    "build_locked_evaluation_plan",
    "evaluate_locked_policy",
    "load_locked_examples",
    "load_locked_source_index_receipt",
    "main",
    "run_locked_test_evaluation",
    "write_locked_source_index_receipt",
]
