from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_no_backward.evaluation_manifest import (
    EvaluationSplitManifest,
    build_evaluation_split_manifest,
    ids_for_evaluation_split,
    load_evaluation_split_manifest,
    read_prior_test_exposure,
    select_items_by_example_id,
    validate_evaluation_split_manifest,
    write_evaluation_split_manifest,
)

_COMMITTED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "gsm8k_standard_lora_eval_manifest.json"
)
_COMMITTED_PRIOR_METADATA = (
    Path(__file__).resolve().parents[1] / "artifacts" / "pilot_gsm8k_1p5b" / "metadata.json",
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "pilot_residual_core_v3"
    / "metadata.json",
)


def _write_metadata(path: Path, ids: list[str], **extra: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"test_example_ids": ids, **extra}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _synthetic_contract(tmp_path: Path) -> tuple[list[str], list[Path], EvaluationSplitManifest]:
    official = [f"test-{index:04d}" for index in range(1_319)]
    first = _write_metadata(tmp_path / "pilot" / "metadata.json", official[:128], run="pilot")
    second = _write_metadata(
        tmp_path / "optimized-v3" / "metadata.json",
        official[128:384],
        run="optimized-v3",
    )
    manifest = build_evaluation_split_manifest(
        official,
        [first, second],
        dev_size=256,
        locked_test_size=679,
        seed=314_159,
        namespace="standard-lora-eval-v1",
    )
    return official, [first, second], manifest


def test_current_quarantine_contract_is_complete_disjoint_and_deterministic(tmp_path: Path) -> None:
    official, metadata_paths, manifest = _synthetic_contract(tmp_path)
    repeated = build_evaluation_split_manifest(
        official,
        list(reversed(metadata_paths)),
        dev_size=256,
        locked_test_size=679,
        seed=314_159,
        namespace="standard-lora-eval-v1",
    )

    assert manifest == repeated
    assert manifest.official_test_count == 1_319
    assert manifest.excluded_test_count == 384
    assert manifest.dev_count == 256
    assert manifest.locked_test_count == 679
    excluded = set(manifest.excluded_test_example_ids)
    dev = set(manifest.dev_example_ids)
    locked = set(manifest.locked_test_example_ids)
    assert not (excluded & dev or excluded & locked or dev & locked)
    assert excluded | dev | locked == set(official)
    assert manifest.manifest_sha256 == "01a3e297e21b372a0e3c8e07c80d686004fb4771e4868aec5daf7c98f5e8a746"
    validate_evaluation_split_manifest(
        manifest,
        official,
        prior_metadata_paths=metadata_paths,
    )


def test_committed_gsm8k_manifest_locks_all_known_exposures_and_digests() -> None:
    manifest = load_evaluation_split_manifest(_COMMITTED_MANIFEST)
    exposure = read_prior_test_exposure(_COMMITTED_PRIOR_METADATA)

    assert manifest.official_test_count == 1_319
    assert manifest.excluded_test_count == 384
    assert manifest.dev_count == 256
    assert manifest.locked_test_count == 679
    assert manifest.namespace == "standard-lora-eval-v1"
    assert manifest.seed == 314_159
    assert manifest.manifest_sha256 == (
        "4046b03c22bc7b6016aaa921d0de9aeee6dc3cf6c7e0b18998222e2df5189793"
    )
    assert manifest.excluded_test_ids_sha256 == (
        "2478bbf24bf3f4c68797c345a8e5fad20d13081c27974aef89945f6bd6748799"
    )
    assert manifest.dev_ids_sha256 == (
        "c626e4bfefb61569a581d2f6290afc03212118cb30f2568d2bf8ffb9662339b2"
    )
    assert manifest.locked_test_ids_sha256 == (
        "5206b42ee9d80dc0f1ab532e3d7fe06fa092d8f3038998771266ddae86424a8e"
    )
    assert exposure.excluded_test_example_ids == manifest.excluded_test_example_ids
    assert exposure.metadata_receipts == manifest.prior_metadata_receipts


def test_metadata_receipts_are_content_addressed_and_union_overlaps(tmp_path: Path) -> None:
    first = _write_metadata(tmp_path / "a.json", ["a", "b"])
    second = _write_metadata(tmp_path / "b.json", ["b", "c"])

    exposure = read_prior_test_exposure([first, second, first])

    assert exposure.excluded_test_example_ids == ("a", "b", "c")
    assert len(exposure.metadata_receipts) == 2
    assert [receipt.test_id_count for receipt in exposure.metadata_receipts] == [2, 2]


def test_builder_rejects_unknown_exclusions_and_unassigned_eligible_ids(tmp_path: Path) -> None:
    unknown = _write_metadata(tmp_path / "unknown.json", ["not-official"])
    with pytest.raises(ValueError, match="outside the official"):
        build_evaluation_split_manifest(
            ["a", "b", "c"],
            [unknown],
            dev_size=1,
            locked_test_size=1,
            seed=0,
            namespace="test",
        )

    known = _write_metadata(tmp_path / "known.json", ["a"])
    with pytest.raises(ValueError, match=r"does not match dev_size \+ locked_test_size"):
        build_evaluation_split_manifest(
            ["a", "b", "c", "d"],
            [known],
            dev_size=1,
            locked_test_size=1,
            seed=0,
            namespace="test",
        )


def test_manifest_write_is_idempotent_but_refuses_mutation_and_detects_tampering(
    tmp_path: Path,
) -> None:
    official, metadata_paths, manifest = _synthetic_contract(tmp_path)
    path = tmp_path / "split-manifest.json"

    assert write_evaluation_split_manifest(path, manifest) == path
    assert write_evaluation_split_manifest(path, manifest) == path
    loaded = load_evaluation_split_manifest(path)
    assert loaded == manifest
    validate_evaluation_split_manifest(
        loaded,
        official,
        prior_metadata_paths=metadata_paths,
    )

    other = build_evaluation_split_manifest(
        official,
        metadata_paths,
        dev_size=256,
        locked_test_size=679,
        seed=314_159,
        namespace="changed",
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_evaluation_split_manifest(path, other)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dev_example_ids"][0], payload["dev_example_ids"][1] = (
        payload["dev_example_ids"][1],
        payload["dev_example_ids"][0],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ids_sha256"):
        load_evaluation_split_manifest(path)


def test_split_and_item_helpers_preserve_committed_order(tmp_path: Path) -> None:
    _, _, manifest = _synthetic_contract(tmp_path)
    requested = manifest.dev_example_ids[:3]
    items = [{"id": example_id} for example_id in reversed(requested)]

    selected = select_items_by_example_id(
        items,
        requested,
        get_example_id=lambda item: item["id"],
    )

    assert tuple(item["id"] for item in selected) == requested
    assert ids_for_evaluation_split(manifest, "dev") == manifest.dev_example_ids
    assert ids_for_evaluation_split(manifest, "locked_test") == manifest.locked_test_example_ids
    with pytest.raises(ValueError, match="split must"):
        ids_for_evaluation_split(manifest, "test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing requested"):
        select_items_by_example_id(
            items[:-1],
            requested,
            get_example_id=lambda item: item["id"],
        )
