"""Validate a completed benchmark artifact directory before publication.

The validator is intentionally independent of the training runner.  It reads
only JSON/YAML metadata, append-only JSONL records, and filesystem structure;
it never imports a model or deserializes a checkpoint.  Missing or short run
artifacts are reported as ``INCOMPLETE`` so an experiment that is still
running cannot be mistaken for a successful release candidate.

Run it with, for example::

    python -m rl_no_backward.validate_artifacts artifacts/final_math_7b

An external config can be supplied with ``--config``.  Otherwise the locked
config embedded in ``metadata.json`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ValidationStatus = Literal["complete", "incomplete", "invalid"]

_SOURCE_COMMIT_KEYS = (
    "source_commit",
    "git_commit",
    "commit",
    "revision",
)
_MODEL_ID_KEYS = ("model_id", "model_name", "name", "id")
_MODEL_REVISION_KEYS = (
    "model_revision",
    "model_commit",
    "snapshot_commit",
    "_commit_hash",
    "revision",
    "commit",
)
_DATASET_ID_KEYS = ("dataset_id", "dataset_name", "name", "id", "path")
_DATASET_REVISION_KEYS = (
    "dataset_revision",
    "dataset_commit",
    "dataset_fingerprint",
    "dataset_fingerprints",
    "fingerprint",
    "fingerprints",
    "split_fingerprints",
    "revision",
    "commit",
)
_MONOTONIC_COUNTERS = (
    "step",
    "wall_time_seconds",
    "environment_samples",
    "generated_tokens",
    "scored_tokens",
    "forward_calls",
    "full_prefix_calls",
    "suffix_calls",
    "backward_calls",
    "peak_gpu_memory_bytes",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
    "cumulative_environment_samples",
    "cumulative_generated_tokens",
    "cumulative_scored_tokens",
    "cumulative_forward_calls",
    "cumulative_backward_calls",
    "cumulative_teacher_forced_examples",
)
_BACKPROP_METHODS = {
    "bp_grpo",
    "grpo",
    "standard_grpo",
    "backprop",
    "backprop_grpo",
}
_ROLLOUT_PROVENANCE_VERSION = "rl-no-backward-rollout-v1"
_ROLLOUT_DIGEST_ALGORITHM = "sha256"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ExpectedRun:
    """One configured method/seed trial and its expected training budget."""

    method: str
    seed: int
    is_base: bool
    is_forward_only: bool

    @property
    def label(self) -> str:
        return f"{self.method}/seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    """Structured validation result returned by the library API."""

    status: ValidationStatus
    output_dir: Path
    expected_runs: int
    validated_runs: int
    incomplete: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "output_dir": str(self.output_dir),
            "expected_runs": self.expected_runs,
            "validated_runs": self.validated_runs,
            "incomplete": list(self.incomplete),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class _Issues:
    def __init__(self) -> None:
        self.incomplete: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def missing(self, message: str) -> None:
        self.incomplete.append(message)

    def invalid(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    else:
        value = _load_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a mapping at its root")
    return dict(value)


def _check_finite_tree(value: Any, location: str, issues: _Issues) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.invalid(f"{location} contains non-finite numeric value {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite_tree(item, f"{location}.{key}", issues)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{location}[{index}]", issues)


def _plain_positive_int(config: Mapping[str, Any], key: str, issues: _Issues) -> int | None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        issues.invalid(f"config.{key} must be a positive integer")
        return None
    return value


def _normalise_config(config: Mapping[str, Any], issues: _Issues) -> dict[str, Any] | None:
    methods = config.get("methods")
    seeds = config.get("seeds")
    if (
        not isinstance(methods, Sequence)
        or isinstance(methods, (str, bytes))
        or not methods
        or any(not isinstance(method, str) or not method.strip() for method in methods)
    ):
        issues.invalid("config.methods must be a non-empty list of method names")
        return None
    clean_methods = [str(method).strip() for method in methods]
    if len(set(clean_methods)) != len(clean_methods):
        issues.invalid("config.methods contains duplicate methods")

    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        issues.invalid("config.seeds must be a non-empty list of integer seeds")
        return None
    clean_seeds = [int(seed) for seed in seeds]
    if len(set(clean_seeds)) != len(clean_seeds):
        issues.invalid("config.seeds contains duplicate seeds")

    numeric = {
        key: _plain_positive_int(config, key, issues)
        for key in ("steps", "batch_size", "group_size", "eval_interval")
    }
    if any(value is None for value in numeric.values()):
        return None

    run_test = config.get("run_test_evaluation", False)
    if not isinstance(run_test, bool):
        issues.invalid("config.run_test_evaluation must be boolean")
        return None
    test_size = config.get("test_size", 0)
    if isinstance(test_size, bool) or not isinstance(test_size, int) or test_size < 0:
        issues.invalid("config.test_size must be a non-negative integer")
        return None
    if run_test and test_size < 1:
        issues.invalid("config.test_size must be positive when official-test evaluation is enabled")

    wandb_mode = config.get("wandb_mode")
    if wandb_mode != "offline":
        issues.invalid("config.wandb_mode must be 'offline' for a publishable offline-run bundle")

    record_rollout_provenance = config.get("record_rollout_provenance", False)
    if not isinstance(record_rollout_provenance, bool):
        issues.invalid("config.record_rollout_provenance must be boolean")
        return None
    vllm_batch_invariant = config.get("vllm_batch_invariant", False)
    if not isinstance(vllm_batch_invariant, bool):
        issues.invalid("config.vllm_batch_invariant must be boolean")
        return None
    if vllm_batch_invariant and config.get("rollout_backend") != "vllm":
        issues.invalid("config.vllm_batch_invariant=true requires rollout_backend='vllm'")
        return None
    vllm_enable_v1_multiprocessing = config.get("vllm_enable_v1_multiprocessing", True)
    if not isinstance(vllm_enable_v1_multiprocessing, bool):
        issues.invalid("config.vllm_enable_v1_multiprocessing must be boolean")
        return None
    vllm_allow_insecure_serialization = config.get("vllm_allow_insecure_serialization", False)
    if not isinstance(vllm_allow_insecure_serialization, bool):
        issues.invalid("config.vllm_allow_insecure_serialization must be boolean")
        return None
    if (
        config.get("rollout_backend") == "vllm"
        and vllm_enable_v1_multiprocessing
        and not vllm_allow_insecure_serialization
    ):
        issues.invalid(
            "multiprocess mutable vLLM artifacts require explicit trusted-local "
            "callable serialization opt-in"
        )
        return None

    return {
        **dict(config),
        "methods": clean_methods,
        "seeds": clean_seeds,
        **numeric,
        "run_test_evaluation": run_test,
        "test_size": test_size,
        "record_rollout_provenance": record_rollout_provenance,
        "vllm_batch_invariant": vllm_batch_invariant,
        "vllm_enable_v1_multiprocessing": vllm_enable_v1_multiprocessing,
        "vllm_allow_insecure_serialization": vllm_allow_insecure_serialization,
    }


def _expected_runs(config: Mapping[str, Any]) -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    seeds = list(config["seeds"])
    for method in config["methods"]:
        normalised = method.strip().lower()
        is_base = normalised == "base"
        method_seeds = seeds[:1] if is_base else seeds
        is_forward = not is_base and normalised not in _BACKPROP_METHODS
        for seed in method_seeds:
            runs.append(
                ExpectedRun(method=method, seed=seed, is_base=is_base, is_forward_only=is_forward)
            )
    return runs


def _lookup(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any | None:
    for path in paths:
        value: Any = mapping
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                break
            value = value[component]
        else:
            return value
    return None


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_key(mapping: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = _nonempty_string(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_provenance_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    """Return the first non-empty scalar or collection used as provenance."""

    for key in keys:
        value = mapping.get(key)
        if _nonempty_string(value) is not None:
            return value
        if isinstance(value, Mapping) and value:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            return value
    return None


def _validate_provenance(
    metadata: Mapping[str, Any], config: Mapping[str, Any], issues: _Issues
) -> None:
    provenance = metadata.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source = provenance.get("source")
    source = source if isinstance(source, Mapping) else provenance
    commit = _first_key(source, _SOURCE_COMMIT_KEYS) or _first_key(metadata, _SOURCE_COMMIT_KEYS)
    if commit is None:
        issues.invalid("metadata is missing source commit provenance")
    elif not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        issues.invalid("source commit provenance must be a hexadecimal commit identifier")

    model = provenance.get("model")
    model = model if isinstance(model, Mapping) else metadata.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    model_id = _first_key(model, _MODEL_ID_KEYS) or _nonempty_string(config.get("model_name"))
    model_revision = (
        _first_key(model, _MODEL_REVISION_KEYS)
        or _first_key(metadata, _MODEL_REVISION_KEYS)
        or _nonempty_string(config.get("model_revision"))
    )
    if model_id is None:
        issues.invalid("metadata/config is missing model identifier provenance")
    if model_revision is None:
        issues.invalid("metadata/config is missing immutable model revision provenance")

    dataset = provenance.get("dataset")
    dataset = dataset if isinstance(dataset, Mapping) else metadata.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    dataset_id = (
        _first_key(dataset, _DATASET_ID_KEYS)
        or _first_key(metadata, ("dataset_id", "dataset_name", "dataset_path"))
        or _first_key(config, ("dataset_id", "dataset_name", "dataset_path"))
    )
    dataset_revision = (
        _first_provenance_value(dataset, _DATASET_REVISION_KEYS)
        or (_first_provenance_value(metadata, _DATASET_REVISION_KEYS))
        or _first_provenance_value(config, _DATASET_REVISION_KEYS)
    )
    if dataset_id is None:
        issues.invalid("metadata/config is missing dataset identifier provenance")
    if dataset_revision is None:
        issues.invalid("metadata/config is missing dataset revision/fingerprint provenance")


def _coerce_id_set(value: Any, location: str, issues: _Issues) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        issues.invalid(f"{location} must be a list of example IDs")
        return set()
    result: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, (str, int)) or isinstance(item, bool) or not str(item).strip():
            issues.invalid(f"{location}[{index}] is not a valid example ID")
            continue
        identifier = str(item)
        if identifier in result:
            issues.invalid(f"{location} contains duplicate ID {identifier!r}")
        result.add(identifier)
    return result


def _collect_excluded_ids(value: Any, path: tuple[str, ...] = ()) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            child_path = (*path, key)
            if "excluded" in key and key.endswith(("ids", "identifiers")):
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    result.update(str(identifier) for identifier in item)
            else:
                result.update(_collect_excluded_ids(item, child_path))
    return result


def _validate_example_ids(
    metadata: Mapping[str, Any], config: Mapping[str, Any], issues: _Issues
) -> None:
    train_ids = _coerce_id_set(
        metadata.get("train_example_ids"), "metadata.train_example_ids", issues
    )
    val_ids = _coerce_id_set(metadata.get("val_example_ids"), "metadata.val_example_ids", issues)
    test_ids = _coerce_id_set(metadata.get("test_example_ids"), "metadata.test_example_ids", issues)

    if train_ids & val_ids:
        issues.invalid("train and validation example IDs overlap")
    excluded = train_ids | val_ids | _collect_excluded_ids(metadata) | _collect_excluded_ids(config)
    overlap = sorted(test_ids & excluded)
    if overlap:
        preview = ", ".join(overlap[:5])
        issues.invalid(f"official-test IDs overlap excluded IDs: {preview}")

    if config["run_test_evaluation"]:
        if not test_ids:
            issues.invalid("metadata.test_example_ids is required for official-test validation")
        elif len(test_ids) != config["test_size"]:
            issues.invalid(
                "metadata.test_example_ids count "
                f"{len(test_ids)} does not match config.test_size {config['test_size']}"
            )


def _find_run_file(directory: Path, run: ExpectedRun, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    token = f"{run.method}_seed{run.seed}{suffix}"
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.name == token or path.name.endswith(f"_{token}"))
    )


def _read_jsonl(path: Path, run: ExpectedRun, issues: _Issues) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as error:
                    issues.missing(
                        f"{run.label} JSONL is incomplete or malformed at line {line_number}: {error}"
                    )
                    return records
                if not isinstance(value, Mapping):
                    issues.invalid(f"{run.label} JSONL line {line_number} is not an object")
                    continue
                record = dict(value)
                _check_finite_tree(record, f"{run.label}.line-{line_number}", issues)
                records.append(record)
    except OSError as error:
        issues.missing(f"could not read {run.label} JSONL: {error}")
    if not records:
        issues.missing(f"{run.label} JSONL contains no complete records")
    return records


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_monotonic_counters(
    records: Sequence[Mapping[str, Any]], run: ExpectedRun, issues: _Issues
) -> None:
    for key in _MONOTONIC_COUNTERS:
        previous: float | None = None
        previous_index = 0
        for index, record in enumerate(records, start=1):
            if key not in record or record[key] is None:
                continue
            value = _numeric(record[key])
            if value is None:
                issues.invalid(f"{run.label} counter {key!r} at record {index} is not numeric/null")
                continue
            if value < 0:
                issues.invalid(f"{run.label} counter {key!r} at record {index} is negative")
            if previous is not None and value + 1e-12 < previous:
                issues.invalid(
                    f"{run.label} counter {key!r} decreases from {previous:g} "
                    f"at record {previous_index} to {value:g} at record {index}"
                )
            previous = value
            previous_index = index


def _expected_validation_steps(config: Mapping[str, Any], run: ExpectedRun) -> list[int]:
    if run.is_base:
        return [0]
    steps = int(config["steps"])
    interval = int(config["eval_interval"])
    scheduled = list(range(interval, steps + 1, interval))
    if not scheduled or scheduled[-1] != steps:
        scheduled.append(steps)
    return [0, *scheduled]


def _split_evaluations(
    evaluations: Sequence[Mapping[str, Any]], run: ExpectedRun, issues: _Issues
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    validation: list[Mapping[str, Any]] = []
    test: list[Mapping[str, Any]] = []
    for record in evaluations:
        raw_split = record.get("split")
        split = str(raw_split).strip().lower() if raw_split is not None else "validation"
        if split == "test":
            test.append(record)
        elif split in {"validation", "val", "valid", "dev", "evaluation", "eval", ""}:
            validation.append(record)
        else:
            issues.invalid(
                f"{run.label} has evaluation record with unsupported split {raw_split!r}"
            )
    return validation, test


def _validate_selection_record(
    record: Mapping[str, Any], run: ExpectedRun, config: Mapping[str, Any], issues: _Issues
) -> None:
    if str(record.get("split", "")).strip().lower() != "test":
        issues.invalid(f"{run.label} official-test record must set split='test' explicitly")
    selected_step = record.get("selected_step")
    if isinstance(selected_step, bool) or not isinstance(selected_step, int):
        issues.invalid(f"{run.label} official-test record is missing integer selected_step")
    elif not 0 <= selected_step <= int(config["steps"]):
        issues.invalid(
            f"{run.label} selected_step {selected_step} is outside the configured budget"
        )
    selection_value = None
    for key in ("selection_metric", "selection_val_accuracy", "selection_accuracy"):
        if key in record:
            selection_value = _numeric(record[key])
            break
    if selection_value is None:
        issues.invalid(f"{run.label} official-test record is missing a finite selection metric")


def _last_numeric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        value = _numeric(record.get(key))
        if value is not None:
            return value
    return None


def _validate_rollout_provenance(
    train_records: Sequence[Mapping[str, Any]],
    run: ExpectedRun,
    config: Mapping[str, Any],
    issues: _Issues,
) -> None:
    """Require the opt-in cryptographic identity on every fixed rollout."""

    if not config["record_rollout_provenance"]:
        return
    observed_digests: set[str] = set()
    for index, record in enumerate(train_records, start=1):
        label = f"{run.label} train-step record {index}"
        if record.get("rollout_provenance_version") != _ROLLOUT_PROVENANCE_VERSION:
            issues.invalid(
                f"{label} must preserve rollout_provenance_version={_ROLLOUT_PROVENANCE_VERSION!r}"
            )
        if record.get("rollout_digest_algorithm") != _ROLLOUT_DIGEST_ALGORITHM:
            issues.invalid(
                f"{label} must preserve rollout_digest_algorithm={_ROLLOUT_DIGEST_ALGORITHM!r}"
            )
        step = record.get("step")
        rollout_seed = record.get("rollout_seed")
        if isinstance(step, bool) or not isinstance(step, int):
            issues.invalid(f"{label} has no integer step for rollout-seed verification")
        else:
            expected_seed = 20_000 + run.seed * 1_000 + step
            if rollout_seed != expected_seed:
                issues.invalid(
                    f"{label} rollout_seed {rollout_seed!r} does not match "
                    f"the deterministic scheduled seed {expected_seed}"
                )
        valid_digests: dict[str, str] = {}
        for key in (
            "rollout_token_digest",
            "behavior_logprob_digest",
            "rollout_digest",
        ):
            value = record.get(key)
            if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
                issues.invalid(f"{label} {key} is not a lowercase SHA-256 digest")
            else:
                valid_digests[key] = value
        if (
            isinstance(rollout_seed, int)
            and not isinstance(rollout_seed, bool)
            and rollout_seed >= 0
            and len(valid_digests) == 3
        ):
            combined = hashlib.sha256()
            combined.update(f"{_ROLLOUT_PROVENANCE_VERSION}/combined\0".encode("ascii"))
            combined.update(struct.pack("<Q", rollout_seed))
            combined.update(bytes.fromhex(valid_digests["rollout_token_digest"]))
            combined.update(bytes.fromhex(valid_digests["behavior_logprob_digest"]))
            if valid_digests["rollout_digest"] != combined.hexdigest():
                issues.invalid(
                    f"{label} rollout_digest does not bind its seed and component digests"
                )
        combined = record.get("rollout_digest")
        if isinstance(combined, str) and _SHA256_HEX.fullmatch(combined):
            if combined in observed_digests:
                issues.invalid(f"{label} repeats an earlier rollout_digest despite a new seed")
            observed_digests.add(combined)


def _record_nonnegative_int(
    record: Mapping[str, Any],
    key: str,
    label: str,
    issues: _Issues,
) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.invalid(f"{label} {key} must be a non-negative integer")
        return None
    return value


def _validate_focus_ablation_records(
    train_records: Sequence[Mapping[str, Any]],
    run: ExpectedRun,
    config: Mapping[str, Any],
    issues: _Issues,
) -> None:
    """Recompute the optional FOCUS audit trail from publishable raw records."""

    if run.method.strip().lower() != "fo_focus_npg":
        return
    forward = config.get("forward")
    focus = config.get("focus")
    if not isinstance(forward, Mapping):
        issues.invalid(f"{run.label} config.forward must define the FOCUS probe budget")
        return
    if not isinstance(focus, Mapping):
        issues.invalid(f"{run.label} config.focus must define the cross-sketch state")
        return

    directions = forward.get("directions")
    family_rank = focus.get("family_rank")
    line_search_steps = forward.get("line_search_steps")
    forward_micro_batch = forward.get("scoring_micro_batch_size")
    old_score_micro_batch = config.get("scoring_micro_batch_size")
    adapter_parameters = config.get("expected_lora_parameter_count")
    batch_size = config.get("batch_size")
    group_size = config.get("group_size")
    integer_config = {
        "forward.directions": directions,
        "focus.family_rank": family_rank,
        "forward.line_search_steps": line_search_steps,
        "forward.scoring_micro_batch_size": forward_micro_batch,
        "scoring_micro_batch_size": old_score_micro_batch,
        "expected_lora_parameter_count": adapter_parameters,
        "batch_size": batch_size,
        "group_size": group_size,
    }
    invalid_config = False
    for name, value in integer_config.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.invalid(f"{run.label} config.{name} must be a positive integer")
            invalid_config = True
    if invalid_config:
        return
    assert isinstance(directions, int)
    assert isinstance(family_rank, int)
    assert isinstance(line_search_steps, int)
    assert isinstance(forward_micro_batch, int)
    assert isinstance(old_score_micro_batch, int)
    assert isinstance(adapter_parameters, int)
    assert isinstance(batch_size, int)
    assert isinstance(group_size, int)
    if directions != 8:
        issues.invalid(f"{run.label} FOCUS requires q=8, not q={directions}")
    if family_rank != 2:
        issues.invalid(f"{run.label} FOCUS requires rank-two state per LoRA family")
    if batch_size < 2 or batch_size % 2:
        issues.invalid(f"{run.label} FOCUS requires an even prompt batch of at least two")

    responses_per_step = batch_size * group_size
    probe_evaluations = 2 * directions
    old_score_calls = math.ceil(responses_per_step / old_score_micro_batch)
    probe_micro_batches = math.ceil(responses_per_step / forward_micro_batch)
    expected_state_cap = adapter_parameters * family_rank + 2 * family_rank
    previous = {
        "environment_samples": 0,
        "generated_tokens": 0,
        "scored_tokens": 0,
        "forward_calls": 0,
        "teacher_forced_examples": 0,
    }
    bootstrap_count = 0
    for record_index, record in enumerate(train_records, start=1):
        label = f"{run.label} FOCUS train-step record {record_index}"
        step = _record_nonnegative_int(record, "step", label, issues)
        if step != record_index:
            issues.invalid(f"{label} step must be the contiguous value {record_index}")

        bootstrap = record.get("focus_bootstrap_b_only")
        if not isinstance(bootstrap, bool):
            issues.invalid(f"{label} focus_bootstrap_b_only must be boolean")
            bootstrap = False
        else:
            bootstrap_count += int(bootstrap)
            if bootstrap != (record_index == 1):
                issues.invalid(f"{label} must use one B-only bootstrap at step one")

        a_updates = _record_nonnegative_int(record, "focus_a_update_count", label, issues)
        b_updates = _record_nonnegative_int(record, "focus_b_update_count", label, issues)
        if a_updates is not None and a_updates != record_index - 1:
            issues.invalid(f"{label} A-state update count must equal step minus one")
        if b_updates is not None and b_updates != record_index:
            issues.invalid(f"{label} B-state update count must equal step")

        ranks: list[int] = []
        for rank_key in ("focus_a_rank", "focus_b_rank"):
            rank = _record_nonnegative_int(record, rank_key, label, issues)
            if rank is not None:
                ranks.append(rank)
                if rank > family_rank:
                    issues.invalid(f"{label} {rank_key} exceeds the configured rank cap")

        first_half = _record_nonnegative_int(record, "focus_first_half_prompts", label, issues)
        second_half = _record_nonnegative_int(record, "focus_second_half_prompts", label, issues)
        if first_half is not None and first_half != batch_size // 2:
            issues.invalid(f"{label} first prompt half has the wrong size")
        if second_half is not None and second_half != batch_size // 2:
            issues.invalid(f"{label} second prompt half has the wrong size")
        if (
            first_half is not None
            and second_half is not None
            and (first_half + second_half != batch_size)
        ):
            issues.invalid(f"{label} prompt halves do not partition the configured batch")

        sketches = _record_nonnegative_int(record, "focus_cross_sketch_count", label, issues)
        expected_sketches = 1 if record_index == 1 else 2
        if sketches is not None and sketches != expected_sketches:
            issues.invalid(f"{label} cross-sketch family count must be {expected_sketches}")
        state_numel = _record_nonnegative_int(record, "focus_state_numel", label, issues)
        state_cap = _record_nonnegative_int(record, "focus_state_numel_cap", label, issues)
        if state_cap is not None and state_cap != expected_state_cap:
            issues.invalid(f"{label} state cap does not match rank * P plus eigenvalues")
        if state_numel is not None and state_numel > expected_state_cap:
            issues.invalid(f"{label} state exceeds the independently recomputed storage cap")
        if state_numel is not None and len(ranks) == 2:
            minimum_state = sum(ranks)
            if state_numel < minimum_state or (state_numel == 0) != (minimum_state == 0):
                issues.invalid(f"{label} state size is inconsistent with its family ranks")

        state_policy_evaluations = _record_nonnegative_int(
            record,
            "focus_state_update_policy_evaluations",
            label,
            issues,
        )
        if state_policy_evaluations is not None and state_policy_evaluations != 0:
            issues.invalid(f"{label} cross-sketch update used extra policy evaluations")
        trials = _record_nonnegative_int(record, "line_search_trials", label, issues)
        if trials is not None and trials > line_search_steps:
            issues.invalid(f"{label} line-search trials exceed the configured maximum")
        policy_evaluations = _record_nonnegative_int(record, "policy_evaluations", label, issues)
        if (
            trials is not None
            and policy_evaluations is not None
            and (policy_evaluations != probe_evaluations + trials)
        ):
            issues.invalid(f"{label} policy evaluations are not q8 probes plus line search")

        backward_calls = _record_nonnegative_int(record, "backward_calls", label, issues)
        if backward_calls is not None and backward_calls != 0:
            issues.invalid(f"{label} reports a backward call in strict inference-only FOCUS")
        counters = {key: _record_nonnegative_int(record, key, label, issues) for key in previous}
        if any(value is None for value in counters.values()):
            continue
        resolved = {key: int(value) for key, value in counters.items()}
        deltas = {key: resolved[key] - previous[key] for key in previous}
        if any(value < 0 for value in deltas.values()):
            issues.invalid(f"{label} cumulative compute counters decrease")
        if deltas["environment_samples"] != responses_per_step:
            issues.invalid(f"{label} environment-sample delta differs from B * G")
        generated_delta = deltas["generated_tokens"]
        if generated_delta < responses_per_step:
            issues.invalid(f"{label} generated-token delta is smaller than the response count")

        old_calls = _record_nonnegative_int(
            record, "old_policy_rescore_forward_calls", label, issues
        )
        old_examples = _record_nonnegative_int(
            record, "old_policy_rescore_teacher_forced_examples", label, issues
        )
        old_tokens = _record_nonnegative_int(
            record, "old_policy_rescore_scored_tokens", label, issues
        )
        if old_calls is not None and old_calls != old_score_calls:
            issues.invalid(f"{label} old-policy rescore call count is inconsistent")
        if old_examples is not None and old_examples != responses_per_step:
            issues.invalid(f"{label} old-policy rescore example count is inconsistent")
        if old_tokens is not None and old_tokens != generated_delta:
            issues.invalid(f"{label} old-policy scored tokens differ from rollout tokens")
        if policy_evaluations is not None:
            expected_forward_delta = old_score_calls + policy_evaluations * probe_micro_batches
            expected_example_delta = responses_per_step * (1 + policy_evaluations)
            expected_scored_delta = generated_delta * (1 + policy_evaluations)
            if deltas["forward_calls"] != expected_forward_delta:
                issues.invalid(f"{label} forward-call delta includes unaccounted evaluations")
            if deltas["teacher_forced_examples"] != expected_example_delta:
                issues.invalid(f"{label} teacher-forced example delta is inconsistent")
            if deltas["scored_tokens"] != expected_scored_delta:
                issues.invalid(f"{label} scored-token delta includes unaccounted evaluations")
        previous = resolved

    if bootstrap_count != 1:
        issues.invalid(
            f"{run.label} FOCUS must contain exactly one B-only bootstrap; "
            f"observed {bootstrap_count}"
        )


def _validate_run_records(
    records: Sequence[Mapping[str, Any]],
    run: ExpectedRun,
    config: Mapping[str, Any],
    issues: _Issues,
) -> bool:
    if not records:
        return False
    for index, record in enumerate(records, start=1):
        if record.get("method") != run.method:
            issues.invalid(
                f"{run.label} record {index} method {record.get('method')!r} does not match filename"
            )
        if record.get("seed") != run.seed:
            issues.invalid(
                f"{run.label} record {index} seed {record.get('seed')!r} does not match filename"
            )

    _validate_monotonic_counters(records, run, issues)
    train_records = [record for record in records if record.get("kind") == "train_step"]
    evaluations = [record for record in records if record.get("kind") == "evaluation"]
    unknown_kinds = sorted(
        {str(record.get("kind")) for record in records} - {"train_step", "evaluation"}
    )
    if unknown_kinds:
        issues.invalid(f"{run.label} has unsupported record kinds: {unknown_kinds}")

    _validate_rollout_provenance(train_records, run, config, issues)
    _validate_focus_ablation_records(train_records, run, config, issues)

    expected_train = 0 if run.is_base else int(config["steps"])
    if len(train_records) < expected_train:
        issues.missing(
            f"{run.label} has {len(train_records)}/{expected_train} expected train-step records"
        )
    elif len(train_records) > expected_train:
        issues.invalid(
            f"{run.label} has {len(train_records)} train-step records; expected {expected_train}"
        )

    expected_validation_steps = _expected_validation_steps(config, run)
    validation, test = _split_evaluations(evaluations, run, issues)
    if len(validation) < len(expected_validation_steps):
        issues.missing(
            f"{run.label} has {len(validation)}/{len(expected_validation_steps)} "
            "expected validation evaluations"
        )
    elif len(validation) > len(expected_validation_steps):
        issues.invalid(
            f"{run.label} has {len(validation)} validation evaluations; "
            f"expected {len(expected_validation_steps)}"
        )
    observed_validation_steps = [record.get("step") for record in validation]
    if (
        len(validation) >= len(expected_validation_steps)
        and observed_validation_steps != expected_validation_steps
    ):
        issues.invalid(
            f"{run.label} validation steps {observed_validation_steps} do not match "
            f"expected {expected_validation_steps}"
        )

    expected_test = 1 if config["run_test_evaluation"] else 0
    if len(test) < expected_test:
        issues.missing(f"{run.label} is missing its explicit official-test evaluation")
    elif len(test) > expected_test:
        issues.invalid(
            f"{run.label} has {len(test)} official-test evaluations; expected {expected_test}"
        )
    if expected_test and test:
        _validate_selection_record(test[0], run, config, issues)

    final_environment_samples = _last_numeric(records, "environment_samples")
    expected_environment_samples = (
        0
        if run.is_base
        else int(config["steps"]) * int(config["batch_size"]) * int(config["group_size"])
    )
    if final_environment_samples is None:
        issues.missing(f"{run.label} has no final environment_samples counter")
    elif (
        final_environment_samples < expected_environment_samples
        and len(train_records) < expected_train
    ):
        issues.missing(
            f"{run.label} environment budget is incomplete: "
            f"{final_environment_samples:g}/{expected_environment_samples}"
        )
    elif final_environment_samples != expected_environment_samples:
        issues.invalid(
            f"{run.label} final environment_samples is {final_environment_samples:g}; "
            f"expected {expected_environment_samples}"
        )

    if run.is_forward_only:
        observed_backward = 0
        for index, record in enumerate(records, start=1):
            if "backward_calls" not in record:
                issues.invalid(
                    f"{run.label} record {index} is missing backward_calls audit counter"
                )
                continue
            value = _numeric(record.get("backward_calls"))
            if value is None:
                issues.invalid(f"{run.label} record {index} backward_calls is not numeric")
                continue
            observed_backward += 1
            if value != 0:
                issues.invalid(
                    f"{run.label} is forward-only but record {index} reports backward_calls={value:g}"
                )
        if observed_backward == 0:
            issues.invalid(f"{run.label} has no usable backward_calls audit counters")

    return (
        len(train_records) == expected_train
        and len(validation) == len(expected_validation_steps)
        and len(test) == expected_test
        and final_environment_samples == expected_environment_samples
    )


def _validate_payload_file(path: Path, label: str, issues: _Issues) -> None:
    if path.stat().st_size == 0:
        issues.missing(f"{label} exists but is empty: {path}")
        return
    try:
        value = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        issues.invalid(f"{label} is not valid finite JSON: {path}: {error}")
        return
    _check_finite_tree(value, label, issues)


def _validate_wandb(output: Path, expected_count: int, issues: _Issues) -> None:
    wandb_dir = output / "wandb"
    run_dirs = sorted(path for path in wandb_dir.glob("offline-run-*") if path.is_dir())
    if len(run_dirs) < expected_count:
        issues.missing(f"W&B has {len(run_dirs)}/{expected_count} expected offline-run directories")
    elif len(run_dirs) > expected_count:
        issues.invalid(
            f"W&B has {len(run_dirs)} offline-run directories; expected exactly {expected_count} "
            "(stale or duplicate runs would make sync ambiguous)"
        )

    for directory in run_dirs:
        binaries = sorted(directory.glob("run-*.wandb"))
        if len(binaries) != 1 or binaries[0].stat().st_size == 0:
            issues.missing(f"W&B offline run is missing one non-empty run binary: {directory}")
        debug_log = directory / "logs" / "debug.log"
        if debug_log.is_file():
            try:
                debug_text = debug_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            started = "run started" in debug_text
            finished = "finishing run" in debug_text or "got exitcode: 0" in debug_text
            if started and not finished:
                issues.missing(
                    f"W&B offline run has started but not finished cleanly: {directory.name}"
                )


def _compare_embedded_config(
    embedded: Mapping[str, Any], external: Mapping[str, Any], issues: _Issues
) -> None:
    keys = (
        "model_name",
        "methods",
        "seeds",
        "steps",
        "batch_size",
        "group_size",
        "scoring_micro_batch_size",
        "eval_interval",
        "expected_lora_parameter_count",
        "forward",
        "focus",
        "run_test_evaluation",
        "test_size",
        "wandb_mode",
        "record_rollout_provenance",
        "vllm_batch_invariant",
        "vllm_enable_v1_multiprocessing",
        "vllm_allow_insecure_serialization",
    )
    mismatches = [
        key
        for key in keys
        if key in embedded and key in external and embedded[key] != external[key]
    ]
    defaults = {
        "record_rollout_provenance": False,
        "vllm_batch_invariant": False,
        "vllm_enable_v1_multiprocessing": True,
        "vllm_allow_insecure_serialization": False,
    }
    for defaulted_key, default in defaults.items():
        if (
            embedded.get(defaulted_key, default) != external.get(defaulted_key, default)
            and defaulted_key not in mismatches
        ):
            mismatches.append(defaulted_key)
    if mismatches:
        issues.invalid(f"external config disagrees with metadata.config for keys: {mismatches}")


def validate_benchmark_artifacts(
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ArtifactValidationResult:
    """Validate one benchmark directory and return a non-throwing result.

    Filesystem absence and truncated runs are classified as incomplete.
    Malformed records, leakage, provenance gaps, and invariant violations are
    classified as invalid.  If both are present, ``incomplete`` takes priority
    in the headline status while every invalidity remains listed in ``errors``.
    """

    output = Path(output_dir).resolve()
    issues = _Issues()
    metadata_file = Path(metadata_path).resolve() if metadata_path else output / "metadata.json"
    if not output.is_dir():
        issues.missing(f"benchmark output directory does not exist: {output}")
        return ArtifactValidationResult(
            status="incomplete",
            output_dir=output,
            expected_runs=0,
            validated_runs=0,
            incomplete=tuple(issues.incomplete),
            errors=(),
            warnings=(),
        )
    if not metadata_file.is_file():
        issues.missing(f"metadata file does not exist: {metadata_file}")
        return ArtifactValidationResult(
            status="incomplete",
            output_dir=output,
            expected_runs=0,
            validated_runs=0,
            incomplete=tuple(issues.incomplete),
            errors=(),
            warnings=(),
        )

    try:
        metadata = _load_mapping(metadata_file)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        issues.invalid(f"could not load metadata: {error}")
        metadata = {}
    _check_finite_tree(metadata, "metadata", issues)
    embedded_config = metadata.get("config")
    if not isinstance(embedded_config, Mapping):
        issues.invalid("metadata.config must contain the locked benchmark config")
        embedded_config = {}

    if config_path is not None:
        config_file = Path(config_path).resolve()
        if not config_file.is_file():
            issues.missing(f"config file does not exist: {config_file}")
            external_config: Mapping[str, Any] = {}
        else:
            try:
                external_config = _load_mapping(config_file)
            except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
                issues.invalid(f"could not load config: {error}")
                external_config = {}
        _compare_embedded_config(embedded_config, external_config, issues)
        raw_config = external_config
    else:
        raw_config = embedded_config
    _check_finite_tree(raw_config, "config", issues)
    config = _normalise_config(raw_config, issues)
    if config is None:
        status: ValidationStatus = "incomplete" if issues.incomplete else "invalid"
        return ArtifactValidationResult(
            status=status,
            output_dir=output,
            expected_runs=0,
            validated_runs=0,
            incomplete=tuple(issues.incomplete),
            errors=tuple(issues.errors),
            warnings=tuple(issues.warnings),
        )

    _validate_provenance(metadata, config, issues)
    _validate_example_ids(metadata, config, issues)
    expected_runs = _expected_runs(config)
    validated_runs = 0
    matched_raw: set[Path] = set()
    matched_checkpoints: set[Path] = set()
    matched_samples: set[Path] = set()

    for run in expected_runs:
        raw_matches = _find_run_file(output / "raw", run, ".jsonl")
        checkpoint_matches = _find_run_file(output / "checkpoints", run, ".pt")
        sample_matches = _find_run_file(output / "samples", run, ".json")
        for label, matches in (
            ("JSONL", raw_matches),
            ("checkpoint", checkpoint_matches),
            ("samples", sample_matches),
        ):
            if not matches:
                issues.missing(f"{run.label} is missing expected {label} artifact")
            elif len(matches) > 1:
                issues.invalid(f"{run.label} has ambiguous duplicate {label} artifacts: {matches}")
        if checkpoint_matches:
            checkpoint = checkpoint_matches[0]
            matched_checkpoints.add(checkpoint)
            if checkpoint.stat().st_size == 0:
                issues.missing(f"{run.label} checkpoint exists but is empty: {checkpoint}")
        if sample_matches:
            sample = sample_matches[0]
            matched_samples.add(sample)
            _validate_payload_file(sample, f"{run.label} samples", issues)
        if raw_matches:
            raw = raw_matches[0]
            matched_raw.add(raw)
            records = _read_jsonl(raw, run, issues)
            if _validate_run_records(records, run, config, issues):
                validated_runs += 1

    for directory, suffix, matched, label in (
        (output / "raw", ".jsonl", matched_raw, "JSONL"),
        (output / "checkpoints", ".pt", matched_checkpoints, "checkpoint"),
        (output / "samples", ".json", matched_samples, "samples"),
    ):
        if directory.is_dir():
            extras = sorted(path for path in directory.glob(f"*{suffix}") if path not in matched)
            if extras:
                issues.invalid(
                    f"unexpected {label} artifacts could contaminate the sweep: {extras}"
                )

    _validate_wandb(output, len(expected_runs), issues)
    status = "incomplete" if issues.incomplete else "invalid" if issues.errors else "complete"
    return ArtifactValidationResult(
        status=status,
        output_dir=output,
        expected_runs=len(expected_runs),
        validated_runs=validated_runs,
        incomplete=tuple(issues.incomplete),
        errors=tuple(issues.errors),
        warnings=tuple(issues.warnings),
    )


def _render_text(result: ArtifactValidationResult) -> str:
    headline = result.status.upper()
    lines = [
        f"{headline}: {result.output_dir}",
        f"validated runs: {result.validated_runs}/{result.expected_runs}",
    ]
    if result.incomplete:
        lines.append("incomplete artifacts:")
        lines.extend(f"  - {message}" for message in result.incomplete)
    if result.errors:
        lines.append("invalid artifacts:")
        lines.extend(f"  - {message}" for message in result.errors)
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {message}" for message in result.warnings)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="benchmark artifact directory")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="metadata JSON/YAML (default: OUTPUT_DIR/metadata.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional locked config JSON/YAML; checked against metadata.config",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = validate_benchmark_artifacts(
        arguments.output_dir,
        metadata_path=arguments.metadata,
        config_path=arguments.config,
    )
    if arguments.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(_render_text(result))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    raise SystemExit(main())


__all__ = [
    "ArtifactValidationResult",
    "ExpectedRun",
    "build_parser",
    "main",
    "validate_benchmark_artifacts",
]
