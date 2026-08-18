"""Question-only sealing for the one-shot GSM8K locked-test source indices.

The evaluation manifest uses opaque example IDs derived from complete GSM8K
rows.  This module never reconstructs those IDs.  Instead, it maps the two
already exposed ID/question lists and the committed development indices back
to the pinned test split using only questions, then seals the 679-row source-
index complement.  Full-row membership is checked later, after the one-shot
evaluator has been explicitly authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation_manifest import (
    EvaluationSplitManifest,
    load_evaluation_split_manifest,
    read_prior_test_exposure,
)
from .gsm8k import GSM8K_DATASET_CONFIG, GSM8K_DATASET_ID

LOCKED_SOURCE_INDEX_SCHEMA = "rl-no-backward-gsm8k-locked-source-index-v2"
LOCKED_SOURCE_AUDIT_SCHEMA = "rl-no-backward-gsm8k-locked-source-sealing-audit-v1"
QUESTION_ID_DOMAIN = "rl-no-backward-gsm8k-question-id-v1"
EXPECTED_OFFICIAL_TEST_COUNT = 1_319
EXPECTED_EXCLUDED_COUNT = 384
EXPECTED_DEV_COUNT = 256
EXPECTED_LOCKED_COUNT = 679
PINNED_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PINNED_TEST_DATA_FILE = "main/test-00000-of-00001.parquet"
FINAL_PRIOR_QUESTION_SOURCES = {
    "59299f926b9a3e5f560d73d4d233842dbebd12d7526805642bd668971c4297bc": (
        "configs/gsm8k_pilot_128_question_projection.json"
    ),
    "d3a3add74fe0a4ef2dcd9b1a766a90ad00db2a0b129d77ef58355f44c946f8b1": (
        "artifacts/pilot_residual_core_v3/samples/gsm8k_base_seed0.json"
    ),
}
FINAL_PRIOR_METADATA_SOURCES = {
    "59299f926b9a3e5f560d73d4d233842dbebd12d7526805642bd668971c4297bc": (
        "artifacts/pilot_gsm8k_1p5b/metadata.json"
    ),
    "d3a3add74fe0a4ef2dcd9b1a766a90ad00db2a0b129d77ef58355f44c946f8b1": (
        "artifacts/pilot_residual_core_v3/metadata.json"
    ),
}
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


def _sequence_digest(values: Sequence[Any], *, domain: str) -> str:
    return _json_digest({"domain": domain, "count": len(values), "values": list(values)})


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


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{target} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{target} must contain a JSON object")
    return dict(value)


_JSON_WHITESPACE = frozenset(b" \t\r\n")


def _skip_json_whitespace(data: mmap.mmap, index: int) -> int:
    while index < len(data) and data[index] in _JSON_WHITESPACE:
        index += 1
    return index


def _scan_json_string(data: mmap.mmap, index: int, *, decode: bool) -> tuple[str | None, int]:
    if index >= len(data) or data[index] != ord('"'):
        raise ValueError("expected a JSON string")
    start = index
    index += 1
    while index < len(data):
        character = data[index]
        if character == ord('"'):
            end = index + 1
            if not decode:
                return None, end
            try:
                value = json.loads(bytes(data[start:end]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("selected JSON string is malformed") from error
            if not isinstance(value, str):
                raise TypeError("selected JSON value must be a string")
            return value, end
        if character == ord("\\"):
            index += 2
        else:
            index += 1
    raise ValueError("unterminated JSON string")


def _skip_json_value(data: mmap.mmap, index: int) -> int:
    """Lexically skip one JSON value without decoding any string value."""

    index = _skip_json_whitespace(data, index)
    if index >= len(data):
        raise ValueError("unexpected end of JSON value")
    character = data[index]
    if character == ord('"'):
        _, index = _scan_json_string(data, index, decode=False)
        return index
    if character in (ord("{"), ord("[")):
        opener = character
        closer = ord("}") if opener == ord("{") else ord("]")
        index = _skip_json_whitespace(data, index + 1)
        if index < len(data) and data[index] == closer:
            return index + 1
        while True:
            if opener == ord("{"):
                _, index = _scan_json_string(data, index, decode=False)
                index = _skip_json_whitespace(data, index)
                if index >= len(data) or data[index] != ord(":"):
                    raise ValueError("malformed skipped JSON object")
                index += 1
            index = _skip_json_value(data, index)
            index = _skip_json_whitespace(data, index)
            if index >= len(data):
                raise ValueError("unterminated skipped JSON container")
            if data[index] == closer:
                return index + 1
            if data[index] != ord(","):
                raise ValueError("malformed skipped JSON container")
            index = _skip_json_whitespace(data, index + 1)
    start = index
    while index < len(data) and data[index] not in b",]} \t\r\n":
        index += 1
    if index == start:
        raise ValueError("malformed JSON scalar")
    return index


def _project_json_object_array(
    path: str | Path,
    *,
    selected_fields: frozenset[str],
) -> list[dict[str, str]]:
    """Decode selected strings while lexically skipping all other values."""

    rows: list[dict[str, str]] = []
    with (
        Path(path).open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        index = _skip_json_whitespace(data, 0)
        if index >= len(data) or data[index] != ord("["):
            raise ValueError("prior question sample must contain a JSON array")
        index = _skip_json_whitespace(data, index + 1)
        if index < len(data) and data[index] == ord("]"):
            return rows
        while True:
            if index >= len(data) or data[index] != ord("{"):
                raise ValueError("prior question sample row must be a JSON object")
            index = _skip_json_whitespace(data, index + 1)
            row: dict[str, str] = {}
            if index < len(data) and data[index] == ord("}"):
                index += 1
            else:
                while True:
                    key, index = _scan_json_string(data, index, decode=True)
                    assert key is not None
                    index = _skip_json_whitespace(data, index)
                    if index >= len(data) or data[index] != ord(":"):
                        raise ValueError("malformed prior question sample object")
                    index = _skip_json_whitespace(data, index + 1)
                    if key in selected_fields:
                        if key in row:
                            raise ValueError(f"duplicate selected field {key!r}")
                        value, index = _scan_json_string(data, index, decode=True)
                        assert value is not None
                        row[key] = value
                    else:
                        index = _skip_json_value(data, index)
                    index = _skip_json_whitespace(data, index)
                    if index >= len(data):
                        raise ValueError("unterminated prior question sample object")
                    if data[index] == ord("}"):
                        index += 1
                        break
                    if data[index] != ord(","):
                        raise ValueError("malformed prior question sample object")
                    index = _skip_json_whitespace(data, index + 1)
            if set(row) != selected_fields:
                raise ValueError("prior question sample row lacks selected question/ID fields")
            rows.append(row)
            index = _skip_json_whitespace(data, index)
            if index >= len(data):
                raise ValueError("unterminated prior question sample array")
            if data[index] == ord("]"):
                index = _skip_json_whitespace(data, index + 1)
                if index != len(data):
                    raise ValueError("trailing data after prior question sample array")
                return rows
            if data[index] != ord(","):
                raise ValueError("malformed prior question sample array")
            index = _skip_json_whitespace(data, index + 1)


def gsm8k_question_id(question: str) -> str:
    """Return a domain-separated identity derived from question text only."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    payload = QUESTION_ID_DOMAIN.encode("ascii") + b"\0" + question.strip().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PriorQuestionRecord:
    example_id: str
    question: str

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("prior example_id must be non-empty")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("prior question must be non-empty")
        object.__setattr__(self, "question", self.question.strip())


@dataclass(frozen=True, slots=True)
class DevelopmentSourceEntry:
    source_index: int
    example_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("development source_index must be non-negative")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("development example_id must be non-empty")


@dataclass(frozen=True, slots=True)
class LockedSourceIndexEntry:
    source_index: int
    question_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        _sha256(self.question_id, name="question_id")

    def as_dict(self) -> dict[str, Any]:
        return {"source_index": self.source_index, "question_id": self.question_id}


@dataclass(frozen=True, slots=True)
class LockedSourceIndexReceipt:
    """Answer-free, content-addressed locked source-index complement."""

    dataset_id: str
    dataset_config: str
    dataset_revision: str
    evaluation_manifest_sha256: str
    official_test_ids_sha256: str
    official_test_count: int
    excluded_count: int
    dev_count: int
    locked_test_count: int
    official_question_ids_sha256: str
    excluded_source_indices: tuple[int, ...]
    dev_source_indices: tuple[int, ...]
    excluded_source_indices_sha256: str
    dev_source_indices_sha256: str
    locked_source_indices_sha256: str
    partition_source_indices_sha256: str
    sealing_audit_sha256: str
    entries: tuple[LockedSourceIndexEntry, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.dataset_id != GSM8K_DATASET_ID or self.dataset_config != GSM8K_DATASET_CONFIG:
            raise ValueError("locked source receipt is not for pinned GSM8K main")
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision:
            raise ValueError("dataset_revision must be non-empty")
        for name in (
            "evaluation_manifest_sha256",
            "official_test_ids_sha256",
            "official_question_ids_sha256",
            "excluded_source_indices_sha256",
            "dev_source_indices_sha256",
            "locked_source_indices_sha256",
            "partition_source_indices_sha256",
            "sealing_audit_sha256",
            "receipt_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        counts = (
            self.official_test_count,
            self.excluded_count,
            self.dev_count,
            self.locked_test_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts
        ):
            raise ValueError("source receipt counts must be positive integers")
        if (
            self.excluded_count + self.dev_count + self.locked_test_count
            != self.official_test_count
        ):
            raise ValueError("source receipt partition counts do not cover the official test split")
        if (
            len(self.excluded_source_indices) != self.excluded_count
            or len(self.dev_source_indices) != self.dev_count
            or len(self.entries) != self.locked_test_count
        ):
            raise ValueError("source receipt partition lengths differ from their counts")
        locked_indices = tuple(entry.source_index for entry in self.entries)
        question_ids = tuple(entry.question_id for entry in self.entries)
        for name, values in (
            ("excluded_source_indices", self.excluded_source_indices),
            ("dev_source_indices", self.dev_source_indices),
            ("locked_source_indices", locked_indices),
        ):
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be sorted and unique")
            if any(index < 0 or index >= self.official_test_count for index in values):
                raise ValueError(f"{name} contains an out-of-range index")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("locked question IDs must be unique")
        excluded = set(self.excluded_source_indices)
        dev = set(self.dev_source_indices)
        locked = set(locked_indices)
        if excluded & dev or excluded & locked or dev & locked:
            raise ValueError("source-index partitions must be disjoint")
        if excluded | dev | locked != set(range(self.official_test_count)):
            raise ValueError("source-index partitions do not exactly cover the official test split")
        expected_digests = {
            "excluded_source_indices_sha256": _sequence_digest(
                self.excluded_source_indices, domain="excluded-source-indices"
            ),
            "dev_source_indices_sha256": _sequence_digest(
                self.dev_source_indices, domain="dev-source-indices"
            ),
            "locked_source_indices_sha256": _sequence_digest(
                locked_indices, domain="locked-source-indices"
            ),
            "partition_source_indices_sha256": _sequence_digest(
                tuple(sorted(excluded | dev | locked)), domain="partition-source-indices"
            ),
        }
        for name, expected in expected_digests.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} does not match the sealed source indices")
        if self.receipt_sha256 != _json_digest(self.payload_without_digest()):
            raise ValueError("locked source receipt digest is invalid")

    @property
    def locked_source_indices(self) -> tuple[int, ...]:
        return tuple(entry.source_index for entry in self.entries)

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": LOCKED_SOURCE_INDEX_SCHEMA,
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "official_test_ids_sha256": self.official_test_ids_sha256,
            "official_test_count": self.official_test_count,
            "excluded_count": self.excluded_count,
            "dev_count": self.dev_count,
            "locked_test_count": self.locked_test_count,
            "official_question_ids_sha256": self.official_question_ids_sha256,
            "excluded_source_indices": list(self.excluded_source_indices),
            "dev_source_indices": list(self.dev_source_indices),
            "excluded_source_indices_sha256": self.excluded_source_indices_sha256,
            "dev_source_indices_sha256": self.dev_source_indices_sha256,
            "locked_source_indices_sha256": self.locked_source_indices_sha256,
            "partition_source_indices_sha256": self.partition_source_indices_sha256,
            "sealing_audit_sha256": self.sealing_audit_sha256,
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
            or self.official_test_ids_sha256 != manifest.official_test_ids_sha256
            or self.official_test_count != manifest.official_test_count
            or self.excluded_count != manifest.excluded_test_count
            or self.dev_count != manifest.dev_count
            or self.locked_test_count != manifest.locked_test_count
        ):
            raise ValueError("locked source receipt does not bind the pinned dataset/manifest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LockedSourceIndexReceipt:
        expected = {
            "schema",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "evaluation_manifest_sha256",
            "official_test_ids_sha256",
            "official_test_count",
            "excluded_count",
            "dev_count",
            "locked_test_count",
            "official_question_ids_sha256",
            "excluded_source_indices",
            "dev_source_indices",
            "excluded_source_indices_sha256",
            "dev_source_indices_sha256",
            "locked_source_indices_sha256",
            "partition_source_indices_sha256",
            "sealing_audit_sha256",
            "entries",
            "receipt_sha256",
        }
        if set(value) != expected:
            raise ValueError("locked source receipt has missing or unknown fields")
        if value["schema"] != LOCKED_SOURCE_INDEX_SCHEMA:
            raise ValueError("unsupported locked source receipt schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or any(
            not isinstance(entry, Mapping) or set(entry) != {"source_index", "question_id"}
            for entry in raw_entries
        ):
            raise TypeError("locked source receipt entries are malformed")
        if not isinstance(value["excluded_source_indices"], list) or not isinstance(
            value["dev_source_indices"], list
        ):
            raise TypeError("source receipt partition indices must be lists")
        return cls(
            dataset_id=value["dataset_id"],
            dataset_config=value["dataset_config"],
            dataset_revision=value["dataset_revision"],
            evaluation_manifest_sha256=value["evaluation_manifest_sha256"],
            official_test_ids_sha256=value["official_test_ids_sha256"],
            official_test_count=value["official_test_count"],
            excluded_count=value["excluded_count"],
            dev_count=value["dev_count"],
            locked_test_count=value["locked_test_count"],
            official_question_ids_sha256=value["official_question_ids_sha256"],
            excluded_source_indices=tuple(value["excluded_source_indices"]),
            dev_source_indices=tuple(value["dev_source_indices"]),
            excluded_source_indices_sha256=value["excluded_source_indices_sha256"],
            dev_source_indices_sha256=value["dev_source_indices_sha256"],
            locked_source_indices_sha256=value["locked_source_indices_sha256"],
            partition_source_indices_sha256=value["partition_source_indices_sha256"],
            sealing_audit_sha256=value["sealing_audit_sha256"],
            entries=tuple(
                LockedSourceIndexEntry(
                    source_index=entry["source_index"], question_id=entry["question_id"]
                )
                for entry in raw_entries
            ),
            receipt_sha256=value["receipt_sha256"],
        )


def _load_development_source_entries(
    path: str | Path,
    manifest: EvaluationSplitManifest,
    *,
    dataset_revision: str,
) -> tuple[tuple[DevelopmentSourceEntry, ...], dict[str, Any]]:
    value = _load_json_mapping(path)
    expected = {
        "schema",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "evaluation_manifest_sha256",
        "official_test_count",
        "dev_count",
        "entries",
        "receipt_sha256",
    }
    if set(value) != expected or value["schema"] != "rl-no-backward-gsm8k-dev-source-index-v1":
        raise ValueError("unsupported development source-index receipt")
    serialized_digest = value["receipt_sha256"]
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if serialized_digest != _json_digest(payload):
        raise ValueError("development source-index receipt digest is invalid")
    if (
        value["dataset_id"] != GSM8K_DATASET_ID
        or value["dataset_config"] != GSM8K_DATASET_CONFIG
        or value["dataset_revision"] != dataset_revision
        or value["evaluation_manifest_sha256"] != manifest.manifest_sha256
        or value["official_test_count"] != manifest.official_test_count
        or value["dev_count"] != manifest.dev_count
    ):
        raise ValueError("development receipt does not bind the pinned dataset/manifest")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or any(
        not isinstance(entry, Mapping) or set(entry) != {"source_index", "example_id"}
        for entry in raw_entries
    ):
        raise TypeError("development source-index entries are malformed")
    entries = tuple(
        DevelopmentSourceEntry(entry["source_index"], entry["example_id"]) for entry in raw_entries
    )
    if tuple(entry.example_id for entry in entries) != manifest.dev_example_ids:
        raise ValueError("development source IDs differ from the manifest order")
    source_indices = tuple(sorted(entry.source_index for entry in entries))
    if len(set(source_indices)) != len(source_indices) or any(
        index >= manifest.official_test_count for index in source_indices
    ):
        raise ValueError("development source indices must be unique and in range")
    return entries, {
        "file_sha256": _file_digest(path),
        "receipt_sha256": serialized_digest,
        "fields_used": [
            "schema",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "evaluation_manifest_sha256",
            "official_test_count",
            "dev_count",
            "entries[].source_index",
            "entries[].example_id",
            "receipt_sha256",
        ],
        "row_count": len(entries),
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "dev_ids_sha256": manifest.dev_ids_sha256,
        "dev_source_indices_sha256": _sequence_digest(source_indices, domain="dev-source-indices"),
    }


def _load_touched_test_quarantine(
    path: str | Path,
    manifest: EvaluationSplitManifest,
) -> dict[str, Any]:
    value = _load_json_mapping(path)
    ids = value.get("test_example_ids")
    if (
        value.get("schema_version") != 1
        or not isinstance(ids, list)
        or tuple(ids) != manifest.excluded_test_example_ids
    ):
        raise ValueError("touched-test quarantine differs from the manifest exclusion set")
    return {
        "file_sha256": _file_digest(path),
        "fields_used": ["schema_version", "test_example_ids"],
        "schema_version": 1,
        "row_count": len(ids),
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "excluded_test_ids_sha256": manifest.excluded_test_ids_sha256,
    }


def _load_prior_question_records(
    metadata_paths: Sequence[str | Path],
    sample_paths: Sequence[str | Path],
    manifest: EvaluationSplitManifest,
) -> tuple[tuple[PriorQuestionRecord, ...], list[dict[str, Any]]]:
    if len(metadata_paths) != len(sample_paths) or not metadata_paths:
        raise ValueError("prior metadata and question-sample paths must be paired")
    exposure = read_prior_test_exposure(metadata_paths)
    if (
        exposure.excluded_test_example_ids != manifest.excluded_test_example_ids
        or exposure.metadata_receipts != manifest.prior_metadata_receipts
    ):
        raise ValueError("prior exposure inputs differ from the committed manifest quarantine")
    records: list[PriorQuestionRecord] = []
    sources: list[dict[str, Any]] = []
    manifest_receipts = {
        receipt.file_sha256: receipt for receipt in manifest.prior_metadata_receipts
    }
    for metadata_path, sample_path in zip(metadata_paths, sample_paths, strict=True):
        metadata = _load_json_mapping(metadata_path)
        metadata_file_sha256 = _file_digest(metadata_path)
        metadata_ids = metadata.get("test_example_ids")
        if not isinstance(metadata_ids, list):
            raise TypeError("prior metadata test_example_ids must be a list")
        manifest_receipt = manifest_receipts.get(metadata_file_sha256)
        if manifest_receipt is None:
            raise ValueError("prior metadata file is absent from the evaluation manifest")
        raw_samples = _project_json_object_array(
            sample_path,
            selected_fields=frozenset({"example_id", "question"}),
        )
        projected: list[PriorQuestionRecord] = []
        for row in raw_samples:
            projected.append(PriorQuestionRecord(row["example_id"], row["question"]))
        if [record.example_id for record in projected] != metadata_ids:
            raise ValueError("prior question sample IDs differ from their metadata order")
        records.extend(projected)
        sources.append(
            {
                "metadata_file_sha256": metadata_file_sha256,
                "metadata_test_ids_sha256": manifest_receipt.test_ids_sha256,
                "committed_metadata_source": FINAL_PRIOR_METADATA_SOURCES.get(
                    metadata_file_sha256, "uncommitted-test-fixture"
                ),
                "question_sample_file_sha256": _file_digest(sample_path),
                "committed_question_source": FINAL_PRIOR_QUESTION_SOURCES.get(
                    metadata_file_sha256, "uncommitted-test-fixture"
                ),
                "row_count": len(projected),
                "fields_used": ["example_id", "question"],
                "projection_parser": (
                    "selective JSON lexer; nonselected values skipped without decoding"
                ),
                "question_projection_sha256": _sequence_digest(
                    [
                        {
                            "example_id": record.example_id,
                            "question_id": gsm8k_question_id(record.question),
                        }
                        for record in projected
                    ],
                    domain="prior-question-projection",
                ),
                "reference_answer_values_used": False,
                "completion_values_used": False,
            }
        )
    if len({record.example_id for record in records}) != len(records):
        raise ValueError("prior question records contain duplicate example IDs")
    if tuple(sorted(record.example_id for record in records)) != manifest.excluded_test_example_ids:
        raise ValueError("prior question records do not exactly cover the quarantined IDs")
    return tuple(records), sources


def inspect_prior_question_evidence(path: str | Path) -> dict[str, Any]:
    """Return an answer-free receipt for committed prior question evidence.

    Only ``example_id`` and ``question`` string values are decoded. All other
    values, including reference answers and completions, are skipped lexically.
    """

    raw_rows = _project_json_object_array(
        path,
        selected_fields=frozenset({"example_id", "question"}),
    )
    records = tuple(PriorQuestionRecord(row["example_id"], row["question"]) for row in raw_rows)
    if len({record.example_id for record in records}) != len(records):
        raise ValueError("prior question evidence contains duplicate example IDs")
    return {
        "row_count": len(records),
        "example_ids": tuple(record.example_id for record in records),
        "question_projection_sha256": _sequence_digest(
            [
                {
                    "example_id": record.example_id,
                    "question_id": gsm8k_question_id(record.question),
                }
                for record in records
            ],
            domain="prior-question-projection",
        ),
    }


def _load_pinned_question_column(*, dataset_revision: str) -> tuple[str, ...]:
    """Read only the question column from the revision-pinned Parquet file."""

    try:
        from huggingface_hub import hf_hub_download
        from pyarrow import parquet
    except ImportError as error:  # pragma: no cover - sealing runtime only
        raise RuntimeError("source sealing requires huggingface_hub and pyarrow") from error
    parquet_path = hf_hub_download(
        repo_id=GSM8K_DATASET_ID,
        repo_type="dataset",
        filename=PINNED_TEST_DATA_FILE,
        revision=dataset_revision,
    )
    table = parquet.read_table(parquet_path, columns=["question"])
    if table.column_names != ["question"]:
        raise RuntimeError("Parquet projection returned a non-question column")
    questions = table.column("question").to_pylist()
    if any(not isinstance(question, str) for question in questions):
        raise TypeError("Parquet question column contains a non-string value")
    return tuple(question.strip() for question in questions)


def load_official_test_questions(
    *,
    dataset_revision: str,
    dataset_loader: Callable[..., Any] | None = None,
) -> tuple[str, ...]:
    """Project to the question column before materializing any dataset row."""

    if dataset_loader is None:
        return _load_pinned_question_column(dataset_revision=dataset_revision)
    dataset = dataset_loader(
        path=GSM8K_DATASET_ID,
        name=GSM8K_DATASET_CONFIG,
        split="test",
        revision=dataset_revision,
    )
    select_columns = getattr(dataset, "select_columns", None)
    if not callable(select_columns):
        raise TypeError("pinned dataset does not support question-only projection")
    question_rows = select_columns(["question"])
    column_names = getattr(question_rows, "column_names", None)
    if column_names is not None and list(column_names) != ["question"]:
        raise RuntimeError("dataset projection materialized a non-question column")
    questions: list[str] = []
    for row in question_rows:
        if not isinstance(row, Mapping) or set(row) != {"question"}:
            raise TypeError("question-only dataset row contains unexpected columns")
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("official test question must be non-empty")
        questions.append(question.strip())
    return tuple(questions)


def seal_locked_source_indices(
    manifest: EvaluationSplitManifest,
    official_questions: Sequence[str],
    prior_question_records: Sequence[PriorQuestionRecord],
    development_entries: Sequence[DevelopmentSourceEntry],
    *,
    dataset_revision: str,
    access_sources: Mapping[str, Any],
    dataset_projection: str = "pyarrow.parquet.read_table(columns=['question'])",
) -> tuple[LockedSourceIndexReceipt, dict[str, Any]]:
    """Derive and seal the locked complement without reference answers."""

    questions = tuple(question.strip() for question in official_questions)
    if len(questions) != manifest.official_test_count or len(set(questions)) != len(questions):
        raise ValueError("official questions must be unique and cover the pinned test split")
    if len(prior_question_records) != manifest.excluded_test_count:
        raise ValueError("prior question records do not cover the quarantine")
    if tuple(sorted(record.example_id for record in prior_question_records)) != (
        manifest.excluded_test_example_ids
    ):
        raise ValueError("prior question records differ from the quarantined ID set")
    if tuple(entry.example_id for entry in development_entries) != manifest.dev_example_ids:
        raise ValueError("development source entries differ from manifest order")
    source_by_question = {question: index for index, question in enumerate(questions)}
    try:
        excluded_indices = tuple(
            sorted(source_by_question[record.question] for record in prior_question_records)
        )
    except KeyError as error:
        raise ValueError("a quarantined question is absent from the pinned test split") from error
    dev_indices = tuple(sorted(entry.source_index for entry in development_entries))
    if len(set(excluded_indices)) != len(excluded_indices):
        raise ValueError("quarantined questions map to duplicate source indices")
    if len(set(dev_indices)) != len(dev_indices):
        raise ValueError("development receipt contains duplicate source indices")
    if set(excluded_indices) & set(dev_indices):
        raise ValueError("quarantine and development source indices overlap")
    locked_indices = tuple(
        sorted(set(range(manifest.official_test_count)) - set(excluded_indices) - set(dev_indices))
    )
    if (
        len(excluded_indices) != manifest.excluded_test_count
        or len(dev_indices) != manifest.dev_count
        or len(locked_indices) != manifest.locked_test_count
    ):
        raise ValueError("derived source-index partition has incorrect counts")
    question_ids = tuple(gsm8k_question_id(question) for question in questions)
    entries = tuple(
        LockedSourceIndexEntry(source_index=index, question_id=question_ids[index])
        for index in locked_indices
    )
    partition = {
        "excluded_count": len(excluded_indices),
        "dev_count": len(dev_indices),
        "locked_test_count": len(locked_indices),
        "excluded_source_indices_sha256": _sequence_digest(
            excluded_indices, domain="excluded-source-indices"
        ),
        "dev_source_indices_sha256": _sequence_digest(dev_indices, domain="dev-source-indices"),
        "locked_source_indices_sha256": _sequence_digest(
            locked_indices, domain="locked-source-indices"
        ),
        "partition_source_indices_sha256": _sequence_digest(
            tuple(range(manifest.official_test_count)), domain="partition-source-indices"
        ),
    }
    audit_without_digest = {
        "schema": LOCKED_SOURCE_AUDIT_SCHEMA,
        "dataset": {
            "id": GSM8K_DATASET_ID,
            "config": GSM8K_DATASET_CONFIG,
            "revision": dataset_revision,
            "split": "test",
            "row_count": len(questions),
            "repository_file": (
                PINNED_TEST_DATA_FILE
                if dataset_projection == "pyarrow.parquet.read_table(columns=['question'])"
                else None
            ),
            "column_projection": dataset_projection,
            "row_value_columns_materialized": ["question"],
            "row_value_columns_not_materialized": ["answer"],
        },
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "official_test_ids_sha256": manifest.official_test_ids_sha256,
        "official_question_ids_sha256": _sequence_digest(
            question_ids, domain="official-question-ids"
        ),
        "access_sources": dict(access_sources),
        "partition": partition,
        "locked_membership_derivation": (
            "source-index complement of exact quarantined question/ID evidence "
            "and committed development indices"
        ),
        "opaque_locked_example_id_verification": "deferred_to_authorized_evaluator",
        "reference_answer_values_used": False,
        "model_evaluation_performed": False,
    }
    audit = {**audit_without_digest, "audit_sha256": _json_digest(audit_without_digest)}
    payload = {
        "schema": LOCKED_SOURCE_INDEX_SCHEMA,
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "official_test_ids_sha256": manifest.official_test_ids_sha256,
        "official_test_count": manifest.official_test_count,
        "excluded_count": manifest.excluded_test_count,
        "dev_count": manifest.dev_count,
        "locked_test_count": manifest.locked_test_count,
        "official_question_ids_sha256": audit["official_question_ids_sha256"],
        "excluded_source_indices": list(excluded_indices),
        "dev_source_indices": list(dev_indices),
        **{key: value for key, value in partition.items() if key.endswith("sha256")},
        "sealing_audit_sha256": audit["audit_sha256"],
        "entries": [entry.as_dict() for entry in entries],
    }
    receipt = LockedSourceIndexReceipt(
        dataset_id=GSM8K_DATASET_ID,
        dataset_config=GSM8K_DATASET_CONFIG,
        dataset_revision=dataset_revision,
        evaluation_manifest_sha256=manifest.manifest_sha256,
        official_test_ids_sha256=manifest.official_test_ids_sha256,
        official_test_count=manifest.official_test_count,
        excluded_count=manifest.excluded_test_count,
        dev_count=manifest.dev_count,
        locked_test_count=manifest.locked_test_count,
        official_question_ids_sha256=audit["official_question_ids_sha256"],
        excluded_source_indices=excluded_indices,
        dev_source_indices=dev_indices,
        excluded_source_indices_sha256=partition["excluded_source_indices_sha256"],
        dev_source_indices_sha256=partition["dev_source_indices_sha256"],
        locked_source_indices_sha256=partition["locked_source_indices_sha256"],
        partition_source_indices_sha256=partition["partition_source_indices_sha256"],
        sealing_audit_sha256=audit["audit_sha256"],
        entries=entries,
        receipt_sha256=_json_digest(payload),
    )
    return receipt, audit


def validate_locked_source_seal(
    receipt: LockedSourceIndexReceipt,
    audit: Mapping[str, Any],
    manifest: EvaluationSplitManifest,
    official_questions: Sequence[str],
    *,
    dataset_revision: str,
    allow_injected_dataset_loader: bool = False,
) -> None:
    """Re-verify a seal using only the pinned question projection."""

    receipt.validate_manifest(manifest, dataset_revision=dataset_revision)
    validate_locked_source_audit(
        receipt,
        audit,
        manifest,
        allow_injected_dataset_loader=allow_injected_dataset_loader,
    )
    questions = tuple(question.strip() for question in official_questions)
    question_ids = tuple(gsm8k_question_id(question) for question in questions)
    if len(question_ids) != receipt.official_test_count:
        raise ValueError("question projection count differs from the source receipt")
    if (
        _sequence_digest(question_ids, domain="official-question-ids")
        != receipt.official_question_ids_sha256
    ):
        raise ValueError("ordered official question IDs differ from the source receipt")
    for entry in receipt.entries:
        if question_ids[entry.source_index] != entry.question_id:
            raise ValueError("locked source entry question ID differs from pinned dataset")


def validate_locked_source_audit(
    receipt: LockedSourceIndexReceipt,
    audit: Mapping[str, Any],
    manifest: EvaluationSplitManifest,
    *,
    allow_injected_dataset_loader: bool = False,
) -> None:
    """Validate answer-free audit claims without opening the GSM8K dataset."""

    expected_counts = (
        EXPECTED_OFFICIAL_TEST_COUNT,
        EXPECTED_EXCLUDED_COUNT,
        EXPECTED_DEV_COUNT,
        EXPECTED_LOCKED_COUNT,
    )
    receipt_counts = (
        receipt.official_test_count,
        receipt.excluded_count,
        receipt.dev_count,
        receipt.locked_test_count,
    )
    manifest_counts = (
        manifest.official_test_count,
        manifest.excluded_test_count,
        manifest.dev_count,
        manifest.locked_test_count,
    )
    if receipt_counts != expected_counts or manifest_counts != expected_counts:
        raise ValueError("locked seal requires exact 1319/384/256/679 partition counts")
    if not allow_injected_dataset_loader and receipt.dataset_revision != PINNED_GSM8K_REVISION:
        raise ValueError("locked seal is not bound to the pinned GSM8K revision")

    expected_keys = {
        "schema",
        "dataset",
        "evaluation_manifest_sha256",
        "official_test_ids_sha256",
        "official_question_ids_sha256",
        "access_sources",
        "partition",
        "locked_membership_derivation",
        "opaque_locked_example_id_verification",
        "reference_answer_values_used",
        "model_evaluation_performed",
        "audit_sha256",
    }
    if set(audit) != expected_keys:
        raise ValueError("locked source sealing audit has missing or unknown fields")
    if audit.get("schema") != LOCKED_SOURCE_AUDIT_SCHEMA:
        raise ValueError("unsupported locked source sealing audit schema")
    audit_payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if audit.get("audit_sha256") != _json_digest(audit_payload):
        raise ValueError("locked source sealing audit digest is invalid")
    if audit["audit_sha256"] != receipt.sealing_audit_sha256:
        raise ValueError("source receipt does not bind the sealing audit")
    if (
        audit.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or audit.get("official_test_ids_sha256") != manifest.official_test_ids_sha256
        or audit.get("official_question_ids_sha256") != receipt.official_question_ids_sha256
    ):
        raise ValueError("sealing audit does not bind the receipt/manifest identities")
    dataset = audit.get("dataset")
    production_projection = "pyarrow.parquet.read_table(columns=['question'])"
    injected_projection = "injected dataset_loader.select_columns(['question'])"
    valid_projection = (
        dataset.get("column_projection") == production_projection
        and dataset.get("repository_file") == PINNED_TEST_DATA_FILE
        if isinstance(dataset, Mapping)
        else False
    )
    if allow_injected_dataset_loader and isinstance(dataset, Mapping):
        valid_projection |= (
            dataset.get("column_projection") == injected_projection
            and dataset.get("repository_file") is None
        )
    if not isinstance(dataset, Mapping) or (
        dataset.get("id") != GSM8K_DATASET_ID
        or dataset.get("config") != GSM8K_DATASET_CONFIG
        or dataset.get("revision") != receipt.dataset_revision
        or dataset.get("split") != "test"
        or dataset.get("row_count") != receipt.official_test_count
        or not valid_projection
        or dataset.get("row_value_columns_materialized") != ["question"]
        or dataset.get("row_value_columns_not_materialized") != ["answer"]
    ):
        raise ValueError("sealing audit does not certify the question-only dataset projection")
    partition = audit.get("partition")
    if not isinstance(partition, Mapping) or (
        partition.get("excluded_count") != receipt.excluded_count
        or partition.get("dev_count") != receipt.dev_count
        or partition.get("locked_test_count") != receipt.locked_test_count
        or partition.get("excluded_source_indices_sha256") != receipt.excluded_source_indices_sha256
        or partition.get("dev_source_indices_sha256") != receipt.dev_source_indices_sha256
        or partition.get("locked_source_indices_sha256") != receipt.locked_source_indices_sha256
        or partition.get("partition_source_indices_sha256")
        != receipt.partition_source_indices_sha256
    ):
        raise ValueError("sealing audit partition differs from the source receipt")
    access_sources = audit.get("access_sources")
    if not isinstance(access_sources, Mapping) or set(access_sources) != {
        "manifest",
        "development_source_receipt",
        "touched_test_quarantine",
        "prior_exposure_question_samples",
    }:
        raise TypeError("sealing audit access_sources must contain the exact sealed inputs")
    manifest_access = access_sources.get("manifest")
    if not isinstance(manifest_access, Mapping) or (
        set(manifest_access) != {"file_sha256", "fields_used"}
        or manifest_access.get("fields_used")
        != [
            "manifest_sha256",
            "official_test_ids_sha256",
            "excluded_test_example_ids",
            "dev_example_ids",
            "locked_test_example_ids",
            "prior_metadata_receipts",
        ]
    ):
        raise ValueError("sealing audit manifest access record is malformed")
    _sha256(manifest_access.get("file_sha256"), name="manifest file_sha256")
    dev_access = access_sources.get("development_source_receipt")
    expected_dev_fields = [
        "schema",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "evaluation_manifest_sha256",
        "official_test_count",
        "dev_count",
        "entries[].source_index",
        "entries[].example_id",
        "receipt_sha256",
    ]
    if not isinstance(dev_access, Mapping) or (
        set(dev_access)
        != {
            "file_sha256",
            "receipt_sha256",
            "fields_used",
            "row_count",
            "evaluation_manifest_sha256",
            "dev_ids_sha256",
            "dev_source_indices_sha256",
        }
        or dev_access.get("fields_used") != expected_dev_fields
        or dev_access.get("row_count") != receipt.dev_count
        or dev_access.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or dev_access.get("dev_ids_sha256") != manifest.dev_ids_sha256
        or dev_access.get("dev_source_indices_sha256") != receipt.dev_source_indices_sha256
    ):
        raise ValueError("sealing audit development-receipt access record is malformed")
    _sha256(dev_access.get("file_sha256"), name="development receipt file_sha256")
    _sha256(dev_access.get("receipt_sha256"), name="development receipt_sha256")
    quarantine_access = access_sources.get("touched_test_quarantine")
    if not isinstance(quarantine_access, Mapping) or (
        set(quarantine_access)
        != {
            "file_sha256",
            "fields_used",
            "schema_version",
            "row_count",
            "evaluation_manifest_sha256",
            "excluded_test_ids_sha256",
        }
        or quarantine_access.get("fields_used") != ["schema_version", "test_example_ids"]
        or quarantine_access.get("schema_version") != 1
        or quarantine_access.get("row_count") != receipt.excluded_count
        or quarantine_access.get("evaluation_manifest_sha256") != manifest.manifest_sha256
        or quarantine_access.get("excluded_test_ids_sha256") != manifest.excluded_test_ids_sha256
    ):
        raise ValueError("sealing audit touched-test quarantine access record is malformed")
    _sha256(quarantine_access.get("file_sha256"), name="quarantine file_sha256")
    prior_sources = access_sources.get("prior_exposure_question_samples")
    expected_prior_keys = {
        "metadata_file_sha256",
        "metadata_test_ids_sha256",
        "committed_metadata_source",
        "question_sample_file_sha256",
        "committed_question_source",
        "row_count",
        "fields_used",
        "projection_parser",
        "question_projection_sha256",
        "reference_answer_values_used",
        "completion_values_used",
    }
    if (
        not isinstance(prior_sources, list)
        or not prior_sources
        or any(
            not isinstance(source, Mapping)
            or set(source) != expected_prior_keys
            or source.get("fields_used") != ["example_id", "question"]
            or not isinstance(source.get("committed_metadata_source"), str)
            or not source["committed_metadata_source"]
            or not isinstance(source.get("committed_question_source"), str)
            or not source["committed_question_source"]
            or source.get("projection_parser")
            != "selective JSON lexer; nonselected values skipped without decoding"
            or source.get("reference_answer_values_used") is not False
            or source.get("completion_values_used") is not False
            for source in prior_sources
        )
    ):
        raise ValueError("sealing audit prior-source access scope is not answer-free")
    prior_by_metadata = {source["metadata_file_sha256"]: source for source in prior_sources}
    expected_prior_receipts = {
        prior.file_sha256: prior for prior in manifest.prior_metadata_receipts
    }
    if set(prior_by_metadata) != set(expected_prior_receipts) or any(
        prior_by_metadata[file_sha]["row_count"] != prior.test_id_count
        or prior_by_metadata[file_sha]["metadata_test_ids_sha256"] != prior.test_ids_sha256
        for file_sha, prior in expected_prior_receipts.items()
    ):
        raise ValueError("sealing audit prior sources differ from manifest metadata receipts")
    if sum(source["row_count"] for source in prior_sources) != receipt.excluded_count:
        raise ValueError("sealing audit prior-source counts do not cover the quarantine")
    for source in prior_sources:
        for name in (
            "metadata_file_sha256",
            "metadata_test_ids_sha256",
            "question_sample_file_sha256",
            "question_projection_sha256",
        ):
            _sha256(source.get(name), name=f"prior source {name}")
        expected_source = FINAL_PRIOR_QUESTION_SOURCES.get(source["metadata_file_sha256"])
        expected_metadata = FINAL_PRIOR_METADATA_SOURCES.get(source["metadata_file_sha256"])
        if not allow_injected_dataset_loader and (
            source.get("committed_question_source") != expected_source
            or source.get("committed_metadata_source") != expected_metadata
        ):
            raise ValueError("sealing audit does not bind canonical prior-question evidence")
    if audit.get("reference_answer_values_used") is not False:
        raise ValueError("sealing audit does not certify answer-free construction")
    if audit.get("model_evaluation_performed") is not False:
        raise ValueError("sealing audit reports a model evaluation")
    if audit.get("opaque_locked_example_id_verification") != "deferred_to_authorized_evaluator":
        raise ValueError("sealing audit does not defer opaque ID verification")


def load_locked_source_index_receipt(
    path: str | Path,
    manifest: EvaluationSplitManifest,
    *,
    dataset_revision: str,
) -> LockedSourceIndexReceipt:
    receipt = LockedSourceIndexReceipt.from_mapping(_load_json_mapping(path))
    receipt.validate_manifest(manifest, dataset_revision=dataset_revision)
    return receipt


def _write_json_immutable(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to overwrite immutable seal artifact: {target}"
            ) from None
    return target


def write_answer_free_question_projection(
    source_sample_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Persist only ordered example IDs and questions from a prior sample artifact."""

    rows = _project_json_object_array(
        source_sample_path,
        selected_fields=frozenset({"example_id", "question"}),
    )
    target = Path(output_path)
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if target.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to overwrite answer-free question projection: {target}"
            ) from None
    return target


def write_locked_source_index_receipt(
    path: str | Path,
    receipt: LockedSourceIndexReceipt,
) -> Path:
    return _write_json_immutable(path, receipt.as_dict())


def seal_locked_source_indices_from_files(
    manifest_path: str | Path,
    dev_source_receipt_path: str | Path,
    prior_metadata_paths: Sequence[str | Path],
    prior_sample_paths: Sequence[str | Path],
    output_path: str | Path,
    audit_output_path: str | Path,
    *,
    dataset_revision: str,
    quarantine_path: str | Path,
    dataset_loader: Callable[..., Any] | None = None,
) -> tuple[Path, Path]:
    manifest = load_evaluation_split_manifest(manifest_path)
    if (
        manifest.official_test_count != EXPECTED_OFFICIAL_TEST_COUNT
        or manifest.excluded_test_count != EXPECTED_EXCLUDED_COUNT
        or manifest.dev_count != EXPECTED_DEV_COUNT
        or manifest.locked_test_count != EXPECTED_LOCKED_COUNT
    ):
        raise ValueError("committed manifest does not have the expected 1319/384/256/679 counts")
    if dataset_loader is None and dataset_revision != PINNED_GSM8K_REVISION:
        raise ValueError("source sealing requires the pinned GSM8K dataset revision")
    dev_entries, dev_access = _load_development_source_entries(
        dev_source_receipt_path, manifest, dataset_revision=dataset_revision
    )
    quarantine_access = _load_touched_test_quarantine(quarantine_path, manifest)
    prior_records, prior_access = _load_prior_question_records(
        prior_metadata_paths, prior_sample_paths, manifest
    )
    questions = load_official_test_questions(
        dataset_revision=dataset_revision, dataset_loader=dataset_loader
    )
    access_sources = {
        "manifest": {
            "file_sha256": _file_digest(manifest_path),
            "fields_used": [
                "manifest_sha256",
                "official_test_ids_sha256",
                "excluded_test_example_ids",
                "dev_example_ids",
                "locked_test_example_ids",
                "prior_metadata_receipts",
            ],
        },
        "development_source_receipt": dev_access,
        "touched_test_quarantine": quarantine_access,
        "prior_exposure_question_samples": prior_access,
    }
    receipt, audit = seal_locked_source_indices(
        manifest,
        questions,
        prior_records,
        dev_entries,
        dataset_revision=dataset_revision,
        access_sources=access_sources,
        dataset_projection=(
            "pyarrow.parquet.read_table(columns=['question'])"
            if dataset_loader is None
            else "injected dataset_loader.select_columns(['question'])"
        ),
    )
    validate_locked_source_seal(
        receipt,
        audit,
        manifest,
        questions,
        dataset_revision=dataset_revision,
        allow_injected_dataset_loader=dataset_loader is not None,
    )
    audit_path = _write_json_immutable(audit_output_path, audit)
    receipt_path = write_locked_source_index_receipt(output_path, receipt)
    return receipt_path, audit_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev-source-receipt", type=Path, required=True)
    parser.add_argument("--touched-test-quarantine", type=Path, required=True)
    parser.add_argument("--prior-metadata", type=Path, action="append", required=True)
    parser.add_argument("--prior-question-samples", type=Path, action="append", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt_path, audit_path = seal_locked_source_indices_from_files(
        args.manifest,
        args.dev_source_receipt,
        args.prior_metadata,
        args.prior_question_samples,
        args.output,
        args.audit_output,
        dataset_revision=args.dataset_revision,
        quarantine_path=args.touched_test_quarantine,
    )
    receipt = _load_json_mapping(receipt_path)
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path),
                "audit_path": str(audit_path),
                "official_test_count": receipt["official_test_count"],
                "excluded_count": receipt["excluded_count"],
                "dev_count": receipt["dev_count"],
                "locked_test_count": receipt["locked_test_count"],
                "receipt_sha256": receipt["receipt_sha256"],
                "sealing_audit_sha256": receipt["sealing_audit_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "EXPECTED_DEV_COUNT",
    "EXPECTED_EXCLUDED_COUNT",
    "EXPECTED_LOCKED_COUNT",
    "EXPECTED_OFFICIAL_TEST_COUNT",
    "FINAL_PRIOR_METADATA_SOURCES",
    "FINAL_PRIOR_QUESTION_SOURCES",
    "LOCKED_SOURCE_AUDIT_SCHEMA",
    "LOCKED_SOURCE_INDEX_SCHEMA",
    "PINNED_GSM8K_REVISION",
    "PINNED_TEST_DATA_FILE",
    "DevelopmentSourceEntry",
    "LockedSourceIndexEntry",
    "LockedSourceIndexReceipt",
    "PriorQuestionRecord",
    "gsm8k_question_id",
    "inspect_prior_question_evidence",
    "load_locked_source_index_receipt",
    "load_official_test_questions",
    "main",
    "seal_locked_source_indices",
    "seal_locked_source_indices_from_files",
    "validate_locked_source_audit",
    "validate_locked_source_seal",
    "write_answer_free_question_projection",
    "write_locked_source_index_receipt",
]
