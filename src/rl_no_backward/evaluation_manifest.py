"""Immutable, answer-free evaluation split manifests.

The public GSM8K test split is small enough that repeatedly inspecting it can
quietly turn it into a validation set.  This module makes that exposure
explicit.  It consumes only example IDs from prior run metadata, quarantines
their union, and deterministically partitions the remaining official-test IDs
into a development split and a locked final split.

Manifest construction deliberately accepts strings rather than dataset rows.
Questions and answers therefore never enter the manifest or its hashing path.
Callers may load answers later, after selecting one of the already committed
ID lists for an authorized evaluation.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

EVALUATION_MANIFEST_SCHEMA = "rl-no-backward-evaluation-split-v1"
RANKING_ALGORITHM = "sha256(utf8(seed) + NUL + utf8(namespace) + NUL + utf8(example_id))"
_DIGEST_ALGORITHM = "sha256"
_HEX_DIGITS = frozenset("0123456789abcdef")
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "digest_algorithm",
        "ranking_algorithm",
        "seed",
        "namespace",
        "official_test_count",
        "excluded_test_count",
        "dev_count",
        "locked_test_count",
        "official_test_ids_sha256",
        "excluded_test_ids_sha256",
        "eligible_ranked_ids_sha256",
        "dev_ids_sha256",
        "locked_test_ids_sha256",
        "partition_ids_sha256",
        "prior_metadata_receipts",
        "excluded_test_example_ids",
        "dev_example_ids",
        "locked_test_example_ids",
        "manifest_sha256",
    }
)

T = TypeVar("T")


def _sha256_hex(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _example_ids(values: Iterable[str], *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(values)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate example IDs")
    return result


def _positive_size(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _namespace(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("namespace must be a non-empty string")
    if "\0" in value:
        raise ValueError("namespace must not contain NUL")
    return value


def _ids_digest(values: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{EVALUATION_MANIFEST_SCHEMA}/{domain}\0".encode("ascii"))
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_payload(
    *,
    seed: int,
    namespace: str,
    official_test_count: int,
    official_test_ids_sha256: str,
    excluded_test_example_ids: Sequence[str],
    dev_example_ids: Sequence[str],
    locked_test_example_ids: Sequence[str],
    excluded_test_ids_sha256: str,
    eligible_ranked_ids_sha256: str,
    dev_ids_sha256: str,
    locked_test_ids_sha256: str,
    partition_ids_sha256: str,
    prior_metadata_receipts: Sequence[PriorMetadataReceipt],
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "digest_algorithm": _DIGEST_ALGORITHM,
        "ranking_algorithm": RANKING_ALGORITHM,
        "seed": seed,
        "namespace": namespace,
        "official_test_count": official_test_count,
        "excluded_test_count": len(excluded_test_example_ids),
        "dev_count": len(dev_example_ids),
        "locked_test_count": len(locked_test_example_ids),
        "official_test_ids_sha256": official_test_ids_sha256,
        "excluded_test_ids_sha256": excluded_test_ids_sha256,
        "eligible_ranked_ids_sha256": eligible_ranked_ids_sha256,
        "dev_ids_sha256": dev_ids_sha256,
        "locked_test_ids_sha256": locked_test_ids_sha256,
        "partition_ids_sha256": partition_ids_sha256,
        "prior_metadata_receipts": [receipt.as_dict() for receipt in prior_metadata_receipts],
        "excluded_test_example_ids": list(excluded_test_example_ids),
        "dev_example_ids": list(dev_example_ids),
        "locked_test_example_ids": list(locked_test_example_ids),
    }


@dataclass(frozen=True, slots=True, order=True)
class PriorMetadataReceipt:
    """Content-addressed receipt for one prior run metadata file."""

    file_sha256: str
    test_ids_sha256: str
    test_id_count: int

    def __post_init__(self) -> None:
        _sha256_hex(self.file_sha256, name="file_sha256")
        _sha256_hex(self.test_ids_sha256, name="test_ids_sha256")
        _positive_size(self.test_id_count, name="test_id_count")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "file_sha256": self.file_sha256,
            "test_ids_sha256": self.test_ids_sha256,
            "test_id_count": self.test_id_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PriorMetadataReceipt:
        expected = {"file_sha256", "test_ids_sha256", "test_id_count"}
        if set(value) != expected:
            raise ValueError("prior metadata receipt has missing or unknown fields")
        return cls(
            file_sha256=value["file_sha256"],
            test_ids_sha256=value["test_ids_sha256"],
            test_id_count=value["test_id_count"],
        )


@dataclass(frozen=True, slots=True)
class PriorTestExposure:
    """Union of test IDs observed by one or more prior benchmark runs."""

    excluded_test_example_ids: tuple[str, ...]
    metadata_receipts: tuple[PriorMetadataReceipt, ...]

    def __post_init__(self) -> None:
        ids = _example_ids(
            self.excluded_test_example_ids,
            name="excluded_test_example_ids",
        )
        if ids != tuple(sorted(ids)):
            raise ValueError("excluded_test_example_ids must be sorted")
        if not self.metadata_receipts:
            raise ValueError("at least one prior metadata receipt is required")
        if tuple(sorted(self.metadata_receipts)) != self.metadata_receipts:
            raise ValueError("metadata_receipts must be sorted")
        if len(set(self.metadata_receipts)) != len(self.metadata_receipts):
            raise ValueError("metadata_receipts contains duplicates")


@dataclass(frozen=True, slots=True)
class EvaluationSplitManifest:
    """A content-addressed quarantine/dev/locked-test partition."""

    seed: int
    namespace: str
    official_test_count: int
    official_test_ids_sha256: str
    excluded_test_example_ids: tuple[str, ...]
    dev_example_ids: tuple[str, ...]
    locked_test_example_ids: tuple[str, ...]
    excluded_test_ids_sha256: str
    eligible_ranked_ids_sha256: str
    dev_ids_sha256: str
    locked_test_ids_sha256: str
    partition_ids_sha256: str
    prior_metadata_receipts: tuple[PriorMetadataReceipt, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        _seed(self.seed)
        _namespace(self.namespace)
        _positive_size(self.official_test_count, name="official_test_count")
        excluded = _example_ids(
            self.excluded_test_example_ids,
            name="excluded_test_example_ids",
        )
        dev = _example_ids(self.dev_example_ids, name="dev_example_ids")
        locked = _example_ids(
            self.locked_test_example_ids,
            name="locked_test_example_ids",
        )
        if excluded != tuple(sorted(excluded)):
            raise ValueError("excluded_test_example_ids must be sorted")
        if set(excluded) & set(dev):
            raise ValueError("excluded and development IDs must be disjoint")
        if set(excluded) & set(locked):
            raise ValueError("excluded and locked-test IDs must be disjoint")
        if set(dev) & set(locked):
            raise ValueError("development and locked-test IDs must be disjoint")
        if len(excluded) + len(dev) + len(locked) != self.official_test_count:
            raise ValueError("manifest partitions do not cover official_test_count")
        if not self.prior_metadata_receipts:
            raise ValueError("at least one prior metadata receipt is required")
        if tuple(sorted(self.prior_metadata_receipts)) != self.prior_metadata_receipts:
            raise ValueError("prior_metadata_receipts must be sorted")
        if len(set(self.prior_metadata_receipts)) != len(self.prior_metadata_receipts):
            raise ValueError("prior_metadata_receipts contains duplicates")
        for name, value in (
            ("official_test_ids_sha256", self.official_test_ids_sha256),
            ("excluded_test_ids_sha256", self.excluded_test_ids_sha256),
            ("eligible_ranked_ids_sha256", self.eligible_ranked_ids_sha256),
            ("dev_ids_sha256", self.dev_ids_sha256),
            ("locked_test_ids_sha256", self.locked_test_ids_sha256),
            ("partition_ids_sha256", self.partition_ids_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            _sha256_hex(value, name=name)
        self._validate_internal_digests()

    @property
    def excluded_test_count(self) -> int:
        return len(self.excluded_test_example_ids)

    @property
    def dev_count(self) -> int:
        return len(self.dev_example_ids)

    @property
    def locked_test_count(self) -> int:
        return len(self.locked_test_example_ids)

    @property
    def eligible_ranked_example_ids(self) -> tuple[str, ...]:
        return self.dev_example_ids + self.locked_test_example_ids

    def _payload_without_manifest_digest(self) -> dict[str, Any]:
        return _manifest_payload(
            seed=self.seed,
            namespace=self.namespace,
            official_test_count=self.official_test_count,
            official_test_ids_sha256=self.official_test_ids_sha256,
            excluded_test_example_ids=self.excluded_test_example_ids,
            dev_example_ids=self.dev_example_ids,
            locked_test_example_ids=self.locked_test_example_ids,
            excluded_test_ids_sha256=self.excluded_test_ids_sha256,
            eligible_ranked_ids_sha256=self.eligible_ranked_ids_sha256,
            dev_ids_sha256=self.dev_ids_sha256,
            locked_test_ids_sha256=self.locked_test_ids_sha256,
            partition_ids_sha256=self.partition_ids_sha256,
            prior_metadata_receipts=self.prior_metadata_receipts,
        )

    def _validate_internal_digests(self) -> None:
        expected = {
            "excluded_test_ids_sha256": _ids_digest(
                self.excluded_test_example_ids,
                domain="excluded-test-ids",
            ),
            "eligible_ranked_ids_sha256": _ids_digest(
                self.eligible_ranked_example_ids,
                domain="eligible-ranked-ids",
            ),
            "dev_ids_sha256": _ids_digest(self.dev_example_ids, domain="dev-ids"),
            "locked_test_ids_sha256": _ids_digest(
                self.locked_test_example_ids,
                domain="locked-test-ids",
            ),
            "partition_ids_sha256": _ids_digest(
                self.excluded_test_example_ids + self.eligible_ranked_example_ids,
                domain="partition-ids",
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the manifest contents")
        expected_manifest = hashlib.sha256(
            _canonical_json_bytes(self._payload_without_manifest_digest())
        ).hexdigest()
        if self.manifest_sha256 != expected_manifest:
            raise ValueError("manifest_sha256 does not match the manifest contents")

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload_without_manifest_digest()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationSplitManifest:
        if set(value) != _MANIFEST_KEYS:
            raise ValueError("evaluation manifest has missing or unknown fields")
        if value["schema"] != EVALUATION_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported evaluation manifest schema {value['schema']!r}")
        if value["digest_algorithm"] != _DIGEST_ALGORITHM:
            raise ValueError("evaluation manifest must use SHA-256")
        if value["ranking_algorithm"] != RANKING_ALGORITHM:
            raise ValueError("evaluation manifest uses an unsupported ranking algorithm")
        receipts_value = value["prior_metadata_receipts"]
        if not isinstance(receipts_value, list):
            raise TypeError("prior_metadata_receipts must be a list")
        manifest = cls(
            seed=value["seed"],
            namespace=value["namespace"],
            official_test_count=value["official_test_count"],
            official_test_ids_sha256=value["official_test_ids_sha256"],
            excluded_test_example_ids=tuple(value["excluded_test_example_ids"]),
            dev_example_ids=tuple(value["dev_example_ids"]),
            locked_test_example_ids=tuple(value["locked_test_example_ids"]),
            excluded_test_ids_sha256=value["excluded_test_ids_sha256"],
            eligible_ranked_ids_sha256=value["eligible_ranked_ids_sha256"],
            dev_ids_sha256=value["dev_ids_sha256"],
            locked_test_ids_sha256=value["locked_test_ids_sha256"],
            partition_ids_sha256=value["partition_ids_sha256"],
            prior_metadata_receipts=tuple(
                PriorMetadataReceipt.from_mapping(receipt) for receipt in receipts_value
            ),
            manifest_sha256=value["manifest_sha256"],
        )
        expected_counts = (
            manifest.excluded_test_count,
            manifest.dev_count,
            manifest.locked_test_count,
        )
        serialized_counts = (
            value["excluded_test_count"],
            value["dev_count"],
            value["locked_test_count"],
        )
        if serialized_counts != expected_counts:
            raise ValueError("serialized manifest counts do not match the ID lists")
        return manifest


def read_prior_test_exposure(metadata_paths: Sequence[str | Path]) -> PriorTestExposure:
    """Read the union of ``test_example_ids`` from prior run metadata files."""

    paths = tuple(Path(path) for path in metadata_paths)
    if not paths:
        raise ValueError("at least one prior metadata path is required")
    excluded: set[str] = set()
    receipts: set[PriorMetadataReceipt] = set()
    for path in paths:
        raw = path.read_bytes()
        try:
            metadata = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"prior metadata is not valid UTF-8 JSON: {path}") from error
        if not isinstance(metadata, Mapping):
            raise TypeError(f"prior metadata must contain a JSON object: {path}")
        values = metadata.get("test_example_ids")
        if values is None:
            raise ValueError(f"prior metadata is missing test_example_ids: {path}")
        if not isinstance(values, list):
            raise TypeError(f"test_example_ids in prior metadata must be a list: {path}")
        ids = _example_ids(values, name=f"test_example_ids in {path}")
        excluded.update(ids)
        receipts.add(
            PriorMetadataReceipt(
                file_sha256=hashlib.sha256(raw).hexdigest(),
                test_ids_sha256=_ids_digest(ids, domain="prior-metadata-test-ids"),
                test_id_count=len(ids),
            )
        )
    return PriorTestExposure(
        excluded_test_example_ids=tuple(sorted(excluded)),
        metadata_receipts=tuple(sorted(receipts)),
    )


def _rank_eligible_ids(
    official_test_example_ids: Sequence[str],
    excluded_test_example_ids: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[str, ...]:
    excluded = set(excluded_test_example_ids)
    ranked: list[tuple[bytes, str]] = []
    for example_id in official_test_example_ids:
        if example_id in excluded:
            continue
        key = f"{seed}\0{namespace}\0{example_id}".encode()
        ranked.append((hashlib.sha256(key).digest(), example_id))
    ranked.sort()
    return tuple(example_id for _, example_id in ranked)


def build_evaluation_split_manifest(
    official_test_example_ids: Sequence[str],
    prior_metadata_paths: Sequence[str | Path],
    *,
    dev_size: int,
    locked_test_size: int,
    seed: int,
    namespace: str,
) -> EvaluationSplitManifest:
    """Quarantine prior exposures and partition every remaining official-test ID.

    The input contains IDs only.  The function refuses to leave eligible IDs
    unassigned, so a dataset revision or an incomplete quarantine list cannot
    silently change the locked evaluation population.
    """

    official = _example_ids(official_test_example_ids, name="official_test_example_ids")
    target_dev_size = _positive_size(dev_size, name="dev_size")
    target_locked_size = _positive_size(locked_test_size, name="locked_test_size")
    target_seed = _seed(seed)
    target_namespace = _namespace(namespace)
    exposure = read_prior_test_exposure(prior_metadata_paths)
    official_set = set(official)
    unknown_excluded = set(exposure.excluded_test_example_ids) - official_set
    if unknown_excluded:
        preview = ", ".join(sorted(unknown_excluded)[:3])
        raise ValueError(f"prior metadata contains IDs outside the official test split: {preview}")
    ranked = _rank_eligible_ids(
        official,
        exposure.excluded_test_example_ids,
        seed=target_seed,
        namespace=target_namespace,
    )
    target_total = target_dev_size + target_locked_size
    if len(ranked) != target_total:
        raise ValueError(
            "eligible official-test count does not match dev_size + locked_test_size: "
            f"{len(ranked)} != {target_dev_size} + {target_locked_size}"
        )
    dev = ranked[:target_dev_size]
    locked = ranked[target_dev_size:]
    payload_values: dict[str, Any] = {
        "seed": target_seed,
        "namespace": target_namespace,
        "official_test_count": len(official),
        "official_test_ids_sha256": _ids_digest(official, domain="official-test-ids"),
        "excluded_test_example_ids": exposure.excluded_test_example_ids,
        "dev_example_ids": dev,
        "locked_test_example_ids": locked,
        "excluded_test_ids_sha256": _ids_digest(
            exposure.excluded_test_example_ids,
            domain="excluded-test-ids",
        ),
        "eligible_ranked_ids_sha256": _ids_digest(ranked, domain="eligible-ranked-ids"),
        "dev_ids_sha256": _ids_digest(dev, domain="dev-ids"),
        "locked_test_ids_sha256": _ids_digest(locked, domain="locked-test-ids"),
        "partition_ids_sha256": _ids_digest(
            exposure.excluded_test_example_ids + ranked,
            domain="partition-ids",
        ),
        "prior_metadata_receipts": exposure.metadata_receipts,
    }
    manifest_sha256 = hashlib.sha256(
        _canonical_json_bytes(_manifest_payload(**payload_values))
    ).hexdigest()
    return EvaluationSplitManifest(manifest_sha256=manifest_sha256, **payload_values)


def validate_evaluation_split_manifest(
    manifest: EvaluationSplitManifest,
    official_test_example_ids: Sequence[str],
    *,
    prior_metadata_paths: Sequence[str | Path] | None = None,
) -> None:
    """Validate dataset identity, quarantine inputs, ranking, and partition contents."""

    official = _example_ids(official_test_example_ids, name="official_test_example_ids")
    if len(official) != manifest.official_test_count:
        raise ValueError("official test count does not match the manifest")
    if _ids_digest(official, domain="official-test-ids") != manifest.official_test_ids_sha256:
        raise ValueError("official test IDs or their dataset order do not match the manifest")
    official_set = set(official)
    partition_set = (
        set(manifest.excluded_test_example_ids)
        | set(manifest.dev_example_ids)
        | set(manifest.locked_test_example_ids)
    )
    if partition_set != official_set:
        raise ValueError("manifest partitions do not exactly cover the official test IDs")
    expected_ranked = _rank_eligible_ids(
        official,
        manifest.excluded_test_example_ids,
        seed=manifest.seed,
        namespace=manifest.namespace,
    )
    if expected_ranked != manifest.eligible_ranked_example_ids:
        raise ValueError("development/locked-test IDs do not match deterministic ranking")
    if prior_metadata_paths is not None:
        exposure = read_prior_test_exposure(prior_metadata_paths)
        if exposure.excluded_test_example_ids != manifest.excluded_test_example_ids:
            raise ValueError("prior metadata exclusion union does not match the manifest")
        if exposure.metadata_receipts != manifest.prior_metadata_receipts:
            raise ValueError("prior metadata content receipts do not match the manifest")


def write_evaluation_split_manifest(
    path: str | Path,
    manifest: EvaluationSplitManifest,
) -> Path:
    """Create a manifest once; refuse both mutation and silent overwrite."""

    target = Path(path)
    payload = json.dumps(
        manifest.as_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        existing = target.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite immutable manifest: {target}") from None
    return target


def load_evaluation_split_manifest(path: str | Path) -> EvaluationSplitManifest:
    """Load and cryptographically validate one persisted manifest."""

    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"evaluation manifest is not valid UTF-8 JSON: {target}") from error
    if not isinstance(value, Mapping):
        raise TypeError("evaluation manifest must contain a JSON object")
    return EvaluationSplitManifest.from_mapping(value)


def ids_for_evaluation_split(
    manifest: EvaluationSplitManifest,
    split: Literal["dev", "locked_test"],
) -> tuple[str, ...]:
    """Return the precommitted IDs for one authorized evaluation split."""

    if split == "dev":
        return manifest.dev_example_ids
    if split == "locked_test":
        return manifest.locked_test_example_ids
    raise ValueError("split must be 'dev' or 'locked_test'")


def select_items_by_example_id(
    items: Iterable[T],
    example_ids: Sequence[str],
    *,
    get_example_id: Callable[[T], str],
) -> tuple[T, ...]:
    """Select items in manifest order, rejecting missing or duplicate source IDs."""

    requested = _example_ids(example_ids, name="example_ids")
    by_id: dict[str, T] = {}
    for item in items:
        example_id = get_example_id(item)
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("get_example_id must return a non-empty string")
        if example_id in by_id:
            raise ValueError(f"source items contain duplicate example ID {example_id!r}")
        by_id[example_id] = item
    missing = [example_id for example_id in requested if example_id not in by_id]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"source items are missing requested example IDs: {preview}")
    return tuple(by_id[example_id] for example_id in requested)


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "RANKING_ALGORITHM",
    "EvaluationSplitManifest",
    "PriorMetadataReceipt",
    "PriorTestExposure",
    "build_evaluation_split_manifest",
    "ids_for_evaluation_split",
    "load_evaluation_split_manifest",
    "read_prior_test_exposure",
    "select_items_by_example_id",
    "validate_evaluation_split_manifest",
    "write_evaluation_split_manifest",
]
