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
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .evaluation_manifest import (
    EvaluationSplitManifest,
    load_evaluation_split_manifest,
    validate_evaluation_split_manifest,
)
from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8KExample,
    exact_match_reward,
    extract_model_answer,
    format_prompt,
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

LOCKED_SOURCE_INDEX_SCHEMA = "rl-no-backward-gsm8k-locked-source-index-v1"
LOCKED_EVALUATION_PLAN_SCHEMA = "rl-no-backward-locked-evaluation-plan-v1"
LOCKED_EVALUATION_RESULT_SCHEMA = "rl-no-backward-locked-evaluation-result-v1"
LOCKED_CONSUMPTION_SCHEMA = "rl-no-backward-locked-consumption-v1"
EXPECTED_LOCKED_TEST_COUNT = 679
EXPECTED_METHODS = ("base", "bp_grpo", "fo_npg")
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


@dataclass(frozen=True, slots=True)
class LockedSourceIndexEntry:
    source_index: int
    example_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return {"source_index": self.source_index, "example_id": self.example_id}


@dataclass(frozen=True, slots=True)
class LockedSourceIndexReceipt:
    """Answer-free mapping from the committed locked IDs to Arrow row indices."""

    dataset_id: str
    dataset_config: str
    dataset_revision: str
    evaluation_manifest_sha256: str
    official_test_count: int
    locked_test_count: int
    entries: tuple[LockedSourceIndexEntry, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.dataset_id != GSM8K_DATASET_ID or self.dataset_config != GSM8K_DATASET_CONFIG:
            raise ValueError("locked source receipt is not for pinned GSM8K main")
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision:
            raise ValueError("dataset_revision must be non-empty")
        _sha256(self.evaluation_manifest_sha256, name="evaluation_manifest_sha256")
        _positive_int(self.official_test_count, name="official_test_count")
        _positive_int(self.locked_test_count, name="locked_test_count")
        if len(self.entries) != self.locked_test_count:
            raise ValueError("locked source receipt count differs from its entries")
        indices = tuple(entry.source_index for entry in self.entries)
        ids = tuple(entry.example_id for entry in self.entries)
        if len(set(indices)) != len(indices) or len(set(ids)) != len(ids):
            raise ValueError("locked source receipt contains duplicate indices or IDs")
        if any(index >= self.official_test_count for index in indices):
            raise ValueError("locked source receipt contains an out-of-range index")
        _sha256(self.receipt_sha256, name="receipt_sha256")
        if self.receipt_sha256 != _json_digest(self.payload_without_digest()):
            raise ValueError("locked source receipt digest is invalid")

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": LOCKED_SOURCE_INDEX_SCHEMA,
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "official_test_count": self.official_test_count,
            "locked_test_count": self.locked_test_count,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload_without_digest()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def validate_manifest(
        self,
        manifest: EvaluationSplitManifest,
        *,
        dataset_revision: str,
    ) -> None:
        if (
            self.dataset_revision != dataset_revision
            or self.evaluation_manifest_sha256 != manifest.manifest_sha256
            or self.official_test_count != manifest.official_test_count
            or self.locked_test_count != manifest.locked_test_count
        ):
            raise ValueError("locked source receipt does not bind the pinned dataset/manifest")
        if tuple(entry.example_id for entry in self.entries) != manifest.locked_test_example_ids:
            raise ValueError("locked source receipt IDs differ from the manifest locked order")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LockedSourceIndexReceipt:
        expected = {
            "schema",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "evaluation_manifest_sha256",
            "official_test_count",
            "locked_test_count",
            "entries",
            "receipt_sha256",
        }
        if set(value) != expected:
            raise ValueError("locked source receipt has missing or unknown fields")
        if value["schema"] != LOCKED_SOURCE_INDEX_SCHEMA:
            raise ValueError("unsupported locked source receipt schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or any(
            not isinstance(entry, Mapping) or set(entry) != {"source_index", "example_id"}
            for entry in raw_entries
        ):
            raise TypeError("locked source receipt entries are malformed")
        return cls(
            dataset_id=value["dataset_id"],
            dataset_config=value["dataset_config"],
            dataset_revision=value["dataset_revision"],
            evaluation_manifest_sha256=value["evaluation_manifest_sha256"],
            official_test_count=value["official_test_count"],
            locked_test_count=value["locked_test_count"],
            entries=tuple(
                LockedSourceIndexEntry(
                    source_index=entry["source_index"],
                    example_id=entry["example_id"],
                )
                for entry in raw_entries
            ),
            receipt_sha256=value["receipt_sha256"],
        )


def build_locked_source_index_receipt(
    manifest: EvaluationSplitManifest,
    official_test_example_ids: Sequence[str],
    *,
    dataset_revision: str,
) -> LockedSourceIndexReceipt:
    """Build an answer-free index receipt from an already available ordered ID list.

    This function never loads GSM8K.  Constructing ``official_test_example_ids``
    is intentionally left to the separately authorized data-sealing workflow.
    """

    validate_evaluation_split_manifest(manifest, official_test_example_ids)
    by_id = {example_id: index for index, example_id in enumerate(official_test_example_ids)}
    entries = tuple(
        LockedSourceIndexEntry(source_index=by_id[example_id], example_id=example_id)
        for example_id in manifest.locked_test_example_ids
    )
    payload = {
        "schema": LOCKED_SOURCE_INDEX_SCHEMA,
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "official_test_count": manifest.official_test_count,
        "locked_test_count": manifest.locked_test_count,
        "entries": [entry.as_dict() for entry in entries],
    }
    return LockedSourceIndexReceipt(
        dataset_id=GSM8K_DATASET_ID,
        dataset_config=GSM8K_DATASET_CONFIG,
        dataset_revision=dataset_revision,
        evaluation_manifest_sha256=manifest.manifest_sha256,
        official_test_count=manifest.official_test_count,
        locked_test_count=manifest.locked_test_count,
        entries=entries,
        receipt_sha256=_json_digest(payload),
    )


def load_locked_source_index_receipt(
    path: str | Path,
    manifest: EvaluationSplitManifest,
    *,
    dataset_revision: str,
) -> LockedSourceIndexReceipt:
    receipt = LockedSourceIndexReceipt.from_mapping(_load_json_mapping(path))
    receipt.validate_manifest(manifest, dataset_revision=dataset_revision)
    return receipt


def write_locked_source_index_receipt(
    path: str | Path,
    receipt: LockedSourceIndexReceipt,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(receipt.as_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    )
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to overwrite locked source receipt: {target}"
            ) from None
    return target


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
        or Path(selection["checkpoint_path"]).name != checkpoint_path.name
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
    if (
        payload["method"] != method
        or payload["seed"] != seed
        or payload["selected_step"] != selected_step
        or float(payload["selection_val_accuracy"]) != float(accuracy)
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
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("benchmark seeds must be a non-empty unique integer list")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("git_dirty") is not False
        or metadata.get("dataset_id") != GSM8K_DATASET_ID
        or metadata.get("dataset_config") != GSM8K_DATASET_CONFIG
        or metadata.get("dataset_revision") != config.get("dataset_revision")
        or metadata.get("model_name") != config.get("model_name")
        or metadata.get("model_revision") != config.get("model_revision")
    ):
        raise ValueError("benchmark metadata does not bind its frozen model/dataset config")
    resolved_snapshot = metadata.get("resolved_model_snapshot")
    if not isinstance(resolved_snapshot, str) or not resolved_snapshot:
        raise ValueError("benchmark metadata has no resolved model snapshot")
    if (
        metadata.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or metadata.get("evaluation_manifest_locked_test_ids_sha256")
        != manifest.locked_test_ids_sha256
    ):
        raise ValueError("benchmark metadata differs from the locked evaluation manifest")
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


def _selected_policy_pairs(
    methods: Sequence[str],
    seeds: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    """Mirror training: one shared base plus BP/FO for every experimental seed."""

    if tuple(methods) != EXPECTED_METHODS or not seeds:
        raise ValueError("selected policy pairs require the matched methods and at least one seed")
    return tuple(
        (method, seed)
        for seed_index, seed in enumerate(seeds)
        for method in (methods if seed_index == 0 else tuple(m for m in methods if m != "base"))
    )


def build_locked_evaluation_plan(
    benchmark_output: str | Path,
    manifest_path: str | Path,
    source_index_receipt_path: str | Path,
    *,
    expected_locked_count: int = EXPECTED_LOCKED_TEST_COUNT,
    enforce_committed_inputs: bool = True,
) -> dict[str, Any]:
    """Preflight a frozen evaluation without importing datasets or reading answers."""

    benchmark = Path(benchmark_output).resolve()
    manifest_target = Path(manifest_path).resolve()
    source_target = Path(source_index_receipt_path).resolve()
    manifest = load_evaluation_split_manifest(manifest_target)
    if manifest.locked_test_count != expected_locked_count:
        raise ValueError(
            f"locked manifest contains {manifest.locked_test_count} rows; "
            f"expected {expected_locked_count}"
        )
    metadata_path = benchmark / "metadata.json"
    validation_path = benchmark / "artifact_validation.json"
    metadata = _load_json_mapping(metadata_path)
    config, methods, seeds = _validate_training_metadata(metadata, manifest)
    receipt = load_locked_source_index_receipt(
        source_target,
        manifest,
        dataset_revision=str(config["dataset_revision"]),
    )
    validation = _load_json_mapping(validation_path)
    policy_pairs = _selected_policy_pairs(methods, seeds)
    expected_runs = len(policy_pairs)
    if (
        validation.get("passed") is not True
        or validation.get("status") != "complete"
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
    if enforce_committed_inputs:
        _require_committed_file(manifest_target, worktree)
        _require_committed_file(source_target, worktree)

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
        "row_loading": "Dataset.select(committed_locked_source_indices)",
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
        example = GSM8KExample(
            question=row["question"],
            answer=row["answer"],
            split="test",
            source_index=entry.source_index,
        )
        if example.example_id != entry.example_id:
            raise ValueError("selected locked row does not match its committed example ID")
        examples.append(example)
    if tuple(example.example_id for example in examples) != manifest.locked_test_example_ids:
        raise RuntimeError("materialized locked examples differ from committed manifest order")
    return tuple(examples)


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
    "build_locked_source_index_receipt",
    "evaluate_locked_policy",
    "load_locked_examples",
    "load_locked_source_index_receipt",
    "main",
    "run_locked_test_evaluation",
    "write_locked_source_index_receipt",
]
