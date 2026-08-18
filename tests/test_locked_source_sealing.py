from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from rl_no_backward.evaluation_manifest import (
    build_evaluation_split_manifest,
    load_evaluation_split_manifest,
    write_evaluation_split_manifest,
)
from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.locked_source_sealing import (
    EXPECTED_DEV_COUNT,
    EXPECTED_EXCLUDED_COUNT,
    EXPECTED_LOCKED_COUNT,
    EXPECTED_OFFICIAL_TEST_COUNT,
    LockedSourceIndexReceipt,
    _json_digest,
    inspect_prior_question_evidence,
    load_locked_source_index_receipt,
    load_official_test_questions,
    seal_locked_source_indices_from_files,
    validate_locked_source_audit,
    validate_locked_source_seal,
    write_answer_free_question_projection,
)


def _canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class _QuestionProjection:
    def __init__(self, questions):
        self.questions = questions
        self.column_names = ["question"]

    def __iter__(self):
        return iter({"question": question} for question in self.questions)


class _ProjectionOnlyDataset:
    def __init__(self, questions):
        self.questions = questions
        self.requested_columns = None

    def __iter__(self):
        raise AssertionError("unprojected dataset rows must never be materialized")

    def select_columns(self, columns):
        self.requested_columns = list(columns)
        assert self.requested_columns == ["question"]
        return _QuestionProjection(self.questions)


def test_official_loader_materializes_question_column_only() -> None:
    dataset = _ProjectionOnlyDataset((" One? ", "Two?"))
    questions = load_official_test_questions(
        dataset_revision="dataset-commit",
        dataset_loader=lambda **_: dataset,
    )
    assert questions == ("One?", "Two?")
    assert dataset.requested_columns == ["question"]


def test_answer_free_projection_lexically_skips_unused_values(tmp_path) -> None:
    source = tmp_path / "prior-samples.json"
    source.write_text(
        json.dumps(
            [
                {
                    "example_id": "id-1",
                    "question": "Question only?",
                    "reference_answer": "SECRET-REFERENCE-VALUE",
                    "completion": {"nested": ["SECRET-COMPLETION-VALUE"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    output = write_answer_free_question_projection(source, tmp_path / "projection.json")
    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == [{"example_id": "id-1", "question": "Question only?"}]
    assert "SECRET-REFERENCE-VALUE" not in serialized
    assert "SECRET-COMPLETION-VALUE" not in serialized

    # A skipped value is not even UTF-8 decoded. The selected strings remain
    # valid and are still projected deterministically.
    undecodable = tmp_path / "undecodable-unused-value.json"
    undecodable.write_bytes(b'[{"example_id":"id-2","question":"Still selected?","answer":"\xff"}]')
    undecodable_output = write_answer_free_question_projection(
        undecodable, tmp_path / "undecodable-projection.json"
    )
    assert json.loads(undecodable_output.read_text(encoding="utf-8")) == [
        {"example_id": "id-2", "question": "Still selected?"}
    ]


def test_exact_679_sealing_pipeline_is_answer_free_and_reproducible(tmp_path) -> None:
    examples = tuple(
        GSM8KExample(
            question=f"Pinned question {index}?",
            answer=f"private synthetic reasoning\n#### {index}",
            split="test",
            source_index=index,
        )
        for index in range(EXPECTED_OFFICIAL_TEST_COUNT)
    )
    metadata_paths = []
    sample_paths = []
    cursor = 0
    for group, count in enumerate((128, 256)):
        selected = examples[cursor : cursor + count]
        cursor += count
        metadata_path = tmp_path / f"prior-{group}.json"
        metadata_path.write_text(
            json.dumps({"test_example_ids": [example.example_id for example in selected]}),
            encoding="utf-8",
        )
        sample_path = tmp_path / f"questions-{group}.json"
        sample_path.write_text(
            json.dumps(
                [
                    {
                        "example_id": example.example_id,
                        "question": example.question,
                        "reference_answer": "must remain lexically skipped",
                        "nested_ignored": {"completion": "also skipped"},
                    }
                    for example in selected
                ]
            ),
            encoding="utf-8",
        )
        metadata_paths.append(metadata_path)
        sample_paths.append(sample_path)

    manifest = build_evaluation_split_manifest(
        [example.example_id for example in examples],
        metadata_paths,
        dev_size=EXPECTED_DEV_COUNT,
        locked_test_size=EXPECTED_LOCKED_COUNT,
        seed=314159,
        namespace="exact-679-sealing-test",
    )
    assert manifest.excluded_test_count == EXPECTED_EXCLUDED_COUNT
    manifest_path = write_evaluation_split_manifest(tmp_path / "manifest.json", manifest)
    by_id = {example.example_id: example.source_index for example in examples}
    dev_payload = {
        "schema": "rl-no-backward-gsm8k-dev-source-index-v1",
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": "dataset-commit",
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "official_test_count": EXPECTED_OFFICIAL_TEST_COUNT,
        "dev_count": EXPECTED_DEV_COUNT,
        "entries": [
            {"source_index": by_id[example_id], "example_id": example_id}
            for example_id in manifest.dev_example_ids
        ],
    }
    dev_payload["receipt_sha256"] = _canonical_digest(dev_payload)
    dev_path = tmp_path / "dev-indices.json"
    dev_path.write_text(json.dumps(dev_payload), encoding="utf-8")
    quarantine_path = tmp_path / "quarantine.json"
    quarantine_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "test_example_ids": list(manifest.excluded_test_example_ids),
            }
        ),
        encoding="utf-8",
    )
    dataset = _ProjectionOnlyDataset(tuple(example.question for example in examples))
    receipt_path, audit_path = seal_locked_source_indices_from_files(
        manifest_path,
        dev_path,
        metadata_paths,
        sample_paths,
        tmp_path / "locked-indices.json",
        tmp_path / "locked-indices.audit.json",
        dataset_revision="dataset-commit",
        quarantine_path=quarantine_path,
        dataset_loader=lambda **_: dataset,
    )
    receipt = load_locked_source_index_receipt(
        receipt_path,
        manifest,
        dataset_revision="dataset-commit",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert receipt.official_test_count == EXPECTED_OFFICIAL_TEST_COUNT
    assert receipt.excluded_count == EXPECTED_EXCLUDED_COUNT
    assert receipt.dev_count == EXPECTED_DEV_COUNT
    assert receipt.locked_test_count == EXPECTED_LOCKED_COUNT
    assert len(receipt.entries) == EXPECTED_LOCKED_COUNT
    assert set(receipt.excluded_source_indices).isdisjoint(receipt.dev_source_indices)
    assert set(receipt.locked_source_indices) == (
        set(range(EXPECTED_OFFICIAL_TEST_COUNT))
        - set(receipt.excluded_source_indices)
        - set(receipt.dev_source_indices)
    )
    serialized = receipt_path.read_text(encoding="utf-8")
    assert "private synthetic reasoning" not in serialized
    assert '"question"' not in serialized
    assert audit["dataset"]["row_value_columns_materialized"] == ["question"]
    assert audit["dataset"]["row_value_columns_not_materialized"] == ["answer"]
    assert audit["dataset"]["column_projection"] == (
        "injected dataset_loader.select_columns(['question'])"
    )
    assert audit["dataset"]["repository_file"] is None
    assert audit["access_sources"]["development_source_receipt"]["dev_ids_sha256"] == (
        manifest.dev_ids_sha256
    )
    assert (
        audit["access_sources"]["touched_test_quarantine"]["excluded_test_ids_sha256"]
        == manifest.excluded_test_ids_sha256
    )
    assert all(
        source["projection_parser"].startswith("selective JSON lexer")
        for source in audit["access_sources"]["prior_exposure_question_samples"]
    )
    assert audit["reference_answer_values_used"] is False
    assert audit["model_evaluation_performed"] is False
    validate_locked_source_seal(
        receipt,
        audit,
        manifest,
        [example.question for example in examples],
        dataset_revision="dataset-commit",
        allow_injected_dataset_loader=True,
    )
    with pytest.raises(ValueError, match="ordered official question IDs"):
        validate_locked_source_seal(
            receipt,
            audit,
            manifest,
            ["mutated question", *(example.question for example in examples[1:])],
            dataset_revision="dataset-commit",
            allow_injected_dataset_loader=True,
        )

    # Existing identical artifacts are accepted; any content change is not.
    same_receipt, same_audit = seal_locked_source_indices_from_files(
        manifest_path,
        dev_path,
        metadata_paths,
        sample_paths,
        receipt_path,
        audit_path,
        dataset_revision="dataset-commit",
        quarantine_path=quarantine_path,
        dataset_loader=lambda **_: dataset,
    )
    assert same_receipt == receipt_path
    assert same_audit == audit_path


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    (
        ("development_source_receipt", "row_count", 255, "development-receipt access"),
        ("development_source_receipt", "dev_ids_sha256", "0" * 64, "development-receipt"),
        (
            "development_source_receipt",
            "dev_source_indices_sha256",
            "0" * 64,
            "development-receipt",
        ),
        ("touched_test_quarantine", "row_count", 383, "quarantine access"),
        (
            "touched_test_quarantine",
            "excluded_test_ids_sha256",
            "0" * 64,
            "quarantine access",
        ),
    ),
)
def test_seal_audit_rejects_rehashed_split_access_tampering(section, field, value, error) -> None:
    root = Path.cwd()
    manifest = load_evaluation_split_manifest(
        root / "configs/gsm8k_standard_lora_eval_manifest.json"
    )
    receipt = load_locked_source_index_receipt(
        root / "configs/gsm8k_standard_lora_locked_source_indices.json",
        manifest,
        dataset_revision="740312add88f781978c0658806c59bc2815b9866",
    )
    audit = json.loads(
        (root / "configs/gsm8k_standard_lora_locked_source_indices.audit.json").read_text()
    )
    tampered = deepcopy(audit)
    tampered["access_sources"][section][field] = value
    tampered_payload = {key: value for key, value in tampered.items() if key != "audit_sha256"}
    tampered["audit_sha256"] = _json_digest(tampered_payload)
    receipt_mapping = receipt.as_dict()
    receipt_mapping["sealing_audit_sha256"] = tampered["audit_sha256"]
    receipt_mapping["receipt_sha256"] = _json_digest(
        {key: value for key, value in receipt_mapping.items() if key != "receipt_sha256"}
    )
    rebound_receipt = LockedSourceIndexReceipt.from_mapping(receipt_mapping)
    with pytest.raises(ValueError, match=error):
        validate_locked_source_audit(rebound_receipt, tampered, manifest)


def test_committed_first_prior_projection_is_answer_free_and_reproducible() -> None:
    root = Path.cwd()
    projection_path = root / "configs/gsm8k_pilot_128_question_projection.json"
    metadata_path = root / "artifacts/pilot_gsm8k_1p5b/metadata.json"
    audit = json.loads(
        (root / "configs/gsm8k_standard_lora_locked_source_indices.audit.json").read_text()
    )
    source = audit["access_sources"]["prior_exposure_question_samples"][0]
    rows = json.loads(projection_path.read_text(encoding="utf-8"))
    assert len(rows) == 128
    assert all(set(row) == {"example_id", "question"} for row in rows)
    assert all("answer" not in row and "completion" not in row for row in rows)
    inspected = inspect_prior_question_evidence(projection_path)
    assert inspected["row_count"] == source["row_count"]
    assert inspected["question_projection_sha256"] == source["question_projection_sha256"]
    metadata_ids = json.loads(metadata_path.read_text(encoding="utf-8"))["test_example_ids"]
    assert inspected["example_ids"] == tuple(metadata_ids)
