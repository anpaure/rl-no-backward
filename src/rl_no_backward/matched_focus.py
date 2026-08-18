"""Forward-only covariance-guided subspaces for the matched LoRA policy.

This module contains the stateful core of FOCUS, independent of a rollout or
optimizer implementation.  It deliberately works with scalar directional
derivatives supplied by a caller: no reverse-mode differentiation API is used
here.

The implementation makes four contracts explicit:

* the flattened PEFT policy is partitioned canonically into all ``lora_A`` and
  all ``lora_B`` coordinates;
* two *raw*, independent Gaussian scouts per family define unbiased gradient
  sketches, even though the directions evaluated by the model are the stable
  orthonormal columns obtained by QR;
* a rank-two positive covariance approximation is maintained separately for
  A and B using a truncated exponential moving average of the symmetric
  cross-sketch operator; and
* standard LoRA starts with B equal to zero, so the first search basis can use
  all eight directions on B before switching to the disjoint A4/B4 basis.

Persistent state contains at most ``rank * P`` basis scalars (plus four small
eigenvalue/counter tensors).  Neither a dense ``P x P`` covariance nor a full
gradient is retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .standard_lora import LoRAParameterLayout, lora_parameter_layout

LoRAFamily = Literal["A", "B"]
ProbeKind = Literal["active", "fill", "scout", "bootstrap"]

FOCUS_FAMILIES: tuple[LoRAFamily, LoRAFamily] = ("A", "B")
_LORA_FAMILY_PATTERN = re.compile(r"\.lora_([AB])\.")


def _validate_family(family: str) -> LoRAFamily:
    if family not in FOCUS_FAMILIES:
        raise ValueError(f"unknown LoRA family {family!r}; expected 'A' or 'B'")
    return cast(LoRAFamily, family)


def _family_from_name(name: str) -> LoRAFamily:
    matches = _LORA_FAMILY_PATTERN.findall(name)
    if len(matches) != 1:
        raise ValueError(f"LoRA tensor name must contain exactly one lora_A/lora_B marker: {name!r}")
    return cast(LoRAFamily, matches[0])


def _pair_key(name: str) -> str:
    return _LORA_FAMILY_PATTERN.sub(".lora_*.", name)


@dataclass(frozen=True, slots=True)
class LoRAFamilySpan:
    """One tensor's interval in both the full and family-local vectors."""

    name: str
    family: LoRAFamily
    full_start: int
    full_stop: int
    family_start: int
    family_stop: int
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return self.full_stop - self.full_start


@dataclass(frozen=True, slots=True)
class CanonicalLoRAPartition:
    """Canonical, span-based partition of a standard LoRA parameter vector."""

    spans: tuple[LoRAFamilySpan, ...]
    full_dimension: int
    a_dimension: int
    b_dimension: int
    layout_digest: str
    pair_keys: tuple[str, ...]

    @classmethod
    def from_layout(cls, layout: LoRAParameterLayout) -> CanonicalLoRAPartition:
        if not layout.names:
            raise ValueError("LoRA layout is empty")
        count = len(layout.names)
        if not (len(layout.shapes) == len(layout.numels) == len(layout.dtypes) == count):
            raise ValueError("LoRA layout fields have inconsistent lengths")
        if len(set(layout.names)) != count:
            raise ValueError("LoRA layout contains duplicate tensor names")
        if tuple(sorted(layout.names)) != layout.names:
            raise ValueError("LoRA layout names must use canonical lexicographic order")

        family_offsets: dict[LoRAFamily, int] = {"A": 0, "B": 0}
        full_offset = 0
        spans: list[LoRAFamilySpan] = []
        pairs: dict[str, dict[LoRAFamily, tuple[tuple[int, ...], int]]] = {}
        digest_payload: list[dict[str, Any]] = []
        for name, shape, numel, dtype in zip(
            layout.names,
            layout.shapes,
            layout.numels,
            layout.dtypes,
            strict=True,
        ):
            if not isinstance(numel, int) or isinstance(numel, bool) or numel < 1:
                raise ValueError(f"LoRA tensor {name!r} has invalid numel {numel!r}")
            canonical_shape = tuple(int(value) for value in shape)
            if not canonical_shape or any(value < 1 for value in canonical_shape):
                raise ValueError(f"LoRA tensor {name!r} has invalid shape {shape!r}")
            if math.prod(canonical_shape) != numel:
                raise ValueError(
                    f"LoRA tensor {name!r} shape implies {math.prod(canonical_shape)} values, "
                    f"not {numel}"
                )
            family = _family_from_name(name)
            family_start = family_offsets[family]
            family_stop = family_start + numel
            spans.append(
                LoRAFamilySpan(
                    name=name,
                    family=family,
                    full_start=full_offset,
                    full_stop=full_offset + numel,
                    family_start=family_start,
                    family_stop=family_stop,
                    shape=canonical_shape,
                )
            )
            family_offsets[family] = family_stop
            full_offset += numel

            key = _pair_key(name)
            family_pair = pairs.setdefault(key, {})
            if family in family_pair:
                raise ValueError(f"LoRA pair {key!r} contains more than one family-{family} tensor")
            family_pair[family] = (canonical_shape, numel)
            digest_payload.append(
                {
                    "name": name,
                    "shape": list(canonical_shape),
                    "numel": numel,
                    "dtype": dtype,
                    "family": family,
                }
            )

        incomplete = sorted(key for key, value in pairs.items() if set(value) != set(FOCUS_FAMILIES))
        if incomplete:
            raise ValueError(f"LoRA modules do not have paired A/B tensors: {incomplete[:8]}")
        for key, value in pairs.items():
            a_shape, _ = value["A"]
            b_shape, _ = value["B"]
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise ValueError(f"LoRA pair {key!r} must contain rank-two weight matrices")
            if a_shape[0] != b_shape[1]:
                raise ValueError(
                    f"LoRA pair {key!r} has inconsistent ranks: A={a_shape}, B={b_shape}"
                )

        encoded = json.dumps(
            {"schema": 1, "tensors": digest_payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            spans=tuple(spans),
            full_dimension=full_offset,
            a_dimension=family_offsets["A"],
            b_dimension=family_offsets["B"],
            layout_digest=hashlib.sha256(encoded).hexdigest(),
            pair_keys=tuple(sorted(pairs)),
        )

    @classmethod
    def from_model(cls, model: nn.Module) -> CanonicalLoRAPartition:
        return cls.from_layout(lora_parameter_layout(model))

    def family_dimension(self, family: LoRAFamily) -> int:
        resolved = _validate_family(family)
        return self.a_dimension if resolved == "A" else self.b_dimension

    def family_spans(self, family: LoRAFamily) -> tuple[LoRAFamilySpan, ...]:
        resolved = _validate_family(family)
        return tuple(span for span in self.spans if span.family == resolved)

    def _validate_full_values(self, values: Tensor) -> None:
        if not isinstance(values, Tensor) or values.ndim not in {1, 2}:
            raise TypeError("full LoRA values must be a rank-one or rank-two torch.Tensor")
        if values.shape[0] != self.full_dimension:
            raise ValueError(
                f"full LoRA values have leading dimension {values.shape[0]}; "
                f"expected {self.full_dimension}"
            )

    def gather(self, values: Tensor, family: LoRAFamily) -> Tensor:
        """Gather one family's coordinates while preserving canonical tensor order."""

        self._validate_full_values(values)
        resolved = _validate_family(family)
        pieces = [values[span.full_start : span.full_stop] for span in self.family_spans(resolved)]
        return torch.cat(pieces, dim=0)

    def embed(self, values: Tensor, family: LoRAFamily) -> Tensor:
        """Embed family-local vector(s) into a zeroed full LoRA vector."""

        if not isinstance(values, Tensor) or values.ndim not in {1, 2}:
            raise TypeError("family values must be a rank-one or rank-two torch.Tensor")
        resolved = _validate_family(family)
        expected = self.family_dimension(resolved)
        if values.shape[0] != expected:
            raise ValueError(
                f"family-{resolved} values have leading dimension {values.shape[0]}; expected {expected}"
            )
        output_shape = (self.full_dimension, *values.shape[1:])
        output = torch.zeros(output_shape, dtype=values.dtype, device=values.device)
        for span in self.family_spans(resolved):
            output[span.full_start : span.full_stop].copy_(
                values[span.family_start : span.family_stop]
            )
        return output

    def merge(self, a_values: Tensor, b_values: Tensor) -> Tensor:
        """Merge canonical family-local vectors or matrices into full layout order."""

        if a_values.ndim != b_values.ndim or a_values.shape[1:] != b_values.shape[1:]:
            raise ValueError("A and B values must have the same trailing shape")
        return self.embed(a_values, "A") + self.embed(b_values, "B")


def canonical_lora_partition(model_or_layout: nn.Module | LoRAParameterLayout) -> CanonicalLoRAPartition:
    """Build the canonical A/B partition from a model or an existing layout."""

    if isinstance(model_or_layout, LoRAParameterLayout):
        return CanonicalLoRAPartition.from_layout(model_or_layout)
    if isinstance(model_or_layout, nn.Module):
        return CanonicalLoRAPartition.from_model(model_or_layout)
    raise TypeError("expected a torch module or LoRAParameterLayout")


def requires_b_only_bootstrap(
    center: Tensor,
    partition: CanonicalLoRAPartition,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return whether standard LoRA's B family is still effectively all zero."""

    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("bootstrap tolerance must be non-negative and finite")
    b_values = partition.gather(center, "B")
    if not torch.isfinite(b_values).all():
        raise ValueError("LoRA B values must be finite")
    return bool(b_values.abs().amax().item() <= tolerance)


@dataclass(frozen=True, slots=True)
class FocusCovarianceConfig:
    """Fixed-rank cross-sketch covariance tracking controls."""

    family_rank: int = 2
    ema_decay: float = 0.95
    relative_eigenvalue_floor: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.family_rank != 2:
            raise ValueError("the matched FOCUS design requires rank two per LoRA family")
        if not isinstance(self.ema_decay, (int, float)) or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if (
            not isinstance(self.relative_eigenvalue_floor, (int, float))
            or not math.isfinite(self.relative_eigenvalue_floor)
            or self.relative_eigenvalue_floor < 0
        ):
            raise ValueError("relative_eigenvalue_floor must be non-negative and finite")


def _orthonormal_span(matrix: Tensor, *, tolerance: float = 1.0e-7) -> Tensor:
    """Deterministic twice-reorthogonalized basis for a very thin matrix."""

    if matrix.ndim != 2:
        raise ValueError("span input must be a matrix")
    columns: list[Tensor] = []
    scale = max(float(matrix.norm(dim=0).amax().item()), 1.0) if matrix.shape[1] else 1.0
    for index in range(matrix.shape[1]):
        residual = matrix[:, index].clone()
        for _ in range(2):
            for existing in columns:
                residual = residual - torch.dot(existing, residual) * existing
        norm = float(residual.norm().item())
        if math.isfinite(norm) and norm > tolerance * scale:
            columns.append(residual / norm)
    if not columns:
        return matrix.new_empty((matrix.shape[0], 0))
    return torch.stack(columns, dim=1)


def _canonicalize_column_signs(basis: Tensor) -> Tensor:
    """Remove eigensolver/QR sign ambiguity using each column's largest entry."""

    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    if basis.shape[1] == 0:
        return basis
    indices = basis.abs().argmax(dim=0)
    pivots = basis[indices, torch.arange(basis.shape[1], device=basis.device)]
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return basis * signs


@torch.inference_mode()
def symmetric_cross_sketch_action(
    first_sketch: Tensor,
    second_sketch: Tensor,
    vectors: Tensor,
) -> Tensor:
    """Apply ``(g1 g2^T + g2 g1^T) / 2`` without forming a dense matrix."""

    if first_sketch.ndim != 1 or second_sketch.ndim != 1:
        raise ValueError("cross sketches must be vectors")
    if first_sketch.shape != second_sketch.shape:
        raise ValueError("cross sketches must have identical shapes")
    squeeze = vectors.ndim == 1
    matrix = vectors[:, None] if squeeze else vectors
    if matrix.ndim != 2 or matrix.shape[0] != first_sketch.numel():
        raise ValueError("action vectors have the wrong leading dimension")
    result = 0.5 * (
        first_sketch[:, None] * (second_sketch @ matrix)[None, :]
        + second_sketch[:, None] * (first_sketch @ matrix)[None, :]
    )
    return result[:, 0] if squeeze else result


class CrossSketchCovarianceEMA:
    """Rank-capped positive approximation to a cross-sketch covariance EMA.

    One update is

    ``C <- PSD_rank_k(decay * C + (1-decay) * sym(g1 g2^T))``.

    The update is solved exactly in the span of the previous basis and the two
    new sketches, whose width is at most ``k + 2``.  Consequently both
    persistent memory and update arithmetic stay linear in the family size.
    """

    def __init__(
        self,
        dimension: int,
        config: FocusCovarianceConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("covariance dimension must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("covariance state dtype must be floating point")
        self.dimension = dimension
        self.config = config or FocusCovarianceConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.basis = torch.empty(dimension, 0, device=self.device, dtype=self.dtype)
        self.eigenvalues = torch.empty(0, device=self.device, dtype=self.dtype)
        self.update_count = 0

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def persistent_numel(self) -> int:
        return self.basis.numel() + self.eigenvalues.numel()

    @torch.inference_mode()
    def update(self, first_sketch: Tensor, second_sketch: Tensor) -> None:
        for label, sketch in (("first", first_sketch), ("second", second_sketch)):
            if not isinstance(sketch, Tensor) or sketch.ndim != 1:
                raise TypeError(f"{label} cross sketch must be a rank-one tensor")
            if sketch.numel() != self.dimension:
                raise ValueError(
                    f"{label} cross sketch has {sketch.numel()} values; expected {self.dimension}"
                )
            if not torch.isfinite(sketch).all():
                raise ValueError(f"{label} cross sketch contains non-finite values")
        first = first_sketch.to(device=self.device, dtype=self.dtype)
        second = second_sketch.to(device=self.device, dtype=self.dtype)
        candidates = torch.cat([self.basis, first[:, None], second[:, None]], dim=1)
        span = _orthonormal_span(candidates)
        if span.shape[1] == 0:
            self.update_count += 1
            return

        old_coordinates = span.T @ self.basis
        small = self.config.ema_decay * (
            (old_coordinates * self.eigenvalues[None, :]) @ old_coordinates.T
        )
        first_coordinates = span.T @ first
        second_coordinates = span.T @ second
        cross_weight = 1.0 - self.config.ema_decay
        small = small + 0.5 * cross_weight * (
            torch.outer(first_coordinates, second_coordinates)
            + torch.outer(second_coordinates, first_coordinates)
        )
        small = 0.5 * (small + small.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(small.float())
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        largest = max(float(eigenvalues[0].item()), 0.0)
        floor = self.config.relative_eigenvalue_floor * max(largest, 1.0e-30)
        retained = eigenvalues > floor
        keep = min(int(retained.sum().item()), self.config.family_rank)
        if keep:
            selected_values = eigenvalues[:keep].to(dtype=self.dtype)
            selected_basis = span @ eigenvectors[:, :keep].to(dtype=self.dtype)
            selected_basis = _canonicalize_column_signs(selected_basis)
            self.basis = selected_basis.contiguous()
            self.eigenvalues = selected_values.contiguous()
        else:
            self.basis = torch.empty(
                self.dimension,
                0,
                device=self.device,
                dtype=self.dtype,
            )
            self.eigenvalues = torch.empty(0, device=self.device, dtype=self.dtype)
        self.update_count += 1
        if self.persistent_numel > self.dimension * self.config.family_rank + self.config.family_rank:
            raise RuntimeError("cross-sketch state exceeded its O(Pk) storage cap")

    @torch.inference_mode()
    def dense_for_testing(self, *, maximum_dimension: int = 64) -> Tensor:
        """Materialize a tiny state for diagnostics; reject production-sized use."""

        if self.dimension > maximum_dimension:
            raise ValueError("refusing to materialize a production-sized covariance")
        return (self.basis * self.eigenvalues[None, :]) @ self.basis.T

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dimension": self.dimension,
            "config": asdict(self.config),
            "dtype": str(self.dtype).removeprefix("torch."),
            "update_count": self.update_count,
            "basis": self.basis.detach().cpu().clone(),
            "eigenvalues": self.eigenvalues.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> CrossSketchCovarianceEMA:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported cross-sketch covariance state schema")
        config_value = value.get("config")
        if not isinstance(config_value, Mapping):
            raise TypeError("cross-sketch state has no config mapping")
        config = FocusCovarianceConfig(**dict(config_value))
        dtype_name = value.get("dtype")
        if not isinstance(dtype_name, str) or not hasattr(torch, dtype_name):
            raise ValueError("cross-sketch state has an invalid dtype")
        dtype = getattr(torch, dtype_name)
        state = cls(int(value["dimension"]), config, device=device, dtype=dtype)
        basis = value.get("basis")
        eigenvalues = value.get("eigenvalues")
        if not isinstance(basis, Tensor) or not isinstance(eigenvalues, Tensor):
            raise TypeError("cross-sketch state basis/eigenvalues must be tensors")
        if basis.ndim != 2 or basis.shape[0] != state.dimension:
            raise ValueError("cross-sketch state basis has the wrong shape")
        if eigenvalues.ndim != 1 or eigenvalues.numel() != basis.shape[1]:
            raise ValueError("cross-sketch eigenvalues do not match the basis")
        if basis.shape[1] > config.family_rank:
            raise ValueError("cross-sketch state exceeds configured rank")
        if not torch.isfinite(basis).all() or not torch.isfinite(eigenvalues).all():
            raise ValueError("cross-sketch state contains non-finite values")
        if (eigenvalues <= 0).any():
            raise ValueError("cross-sketch covariance eigenvalues must be positive")
        if eigenvalues.numel() > 1 and (eigenvalues[:-1] < eigenvalues[1:]).any():
            raise ValueError("cross-sketch eigenvalues must be descending")
        moved_basis = basis.to(device=state.device, dtype=state.dtype)
        identity = torch.eye(moved_basis.shape[1], device=state.device, dtype=state.dtype)
        if moved_basis.shape[1] and not torch.allclose(
            moved_basis.T @ moved_basis,
            identity,
            atol=2.0e-5,
            rtol=2.0e-5,
        ):
            raise ValueError("cross-sketch basis is not orthonormal")
        update_count = value.get("update_count")
        if isinstance(update_count, bool) or not isinstance(update_count, int) or update_count < 0:
            raise ValueError("cross-sketch update_count must be a non-negative integer")
        state.basis = moved_basis.contiguous()
        state.eigenvalues = eigenvalues.to(device=state.device, dtype=state.dtype).contiguous()
        state.update_count = update_count
        return state


@dataclass(frozen=True, slots=True)
class FocusScoutPair:
    family: LoRAFamily
    first_raw_column: int
    second_raw_column: int


@dataclass(frozen=True, slots=True)
class FocusCrossSketch:
    """The two independent family-local gradient sketches for one EMA update."""

    family: LoRAFamily
    first: Tensor
    second: Tensor
    first_derivative: Tensor
    second_derivative: Tensor


@dataclass(frozen=True, slots=True)
class FocusProbePlan:
    """Orthonormal evaluation basis plus exact raw-direction reconstruction."""

    basis: Tensor
    reconstruction: Tensor
    raw_families: tuple[LoRAFamily, ...]
    raw_kinds: tuple[ProbeKind, ...]
    scout_pairs: tuple[FocusScoutPair, ...]
    partition_digest: str
    bootstrap_b_only: bool

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError("FOCUS basis must be a matrix")
        directions = self.basis.shape[1]
        if directions != 8:
            raise ValueError(f"matched FOCUS requires q=8 directions, received {directions}")
        if self.reconstruction.shape != (directions, directions):
            raise ValueError("FOCUS reconstruction must be square with basis width")
        if len(self.raw_families) != directions or len(self.raw_kinds) != directions:
            raise ValueError("FOCUS raw-direction metadata does not match basis width")
        if not torch.isfinite(self.basis).all() or not torch.isfinite(self.reconstruction).all():
            raise ValueError("FOCUS plan contains non-finite values")
        gram = self.basis.T @ self.basis
        identity = torch.eye(directions, device=self.basis.device, dtype=self.basis.dtype)
        if not torch.allclose(gram, identity, atol=3.0e-5, rtol=3.0e-5):
            raise ValueError("FOCUS evaluation basis is not orthonormal")
        expected_families = {"B"} if self.bootstrap_b_only else set(FOCUS_FAMILIES)
        if set(self.raw_families) != expected_families:
            raise ValueError("FOCUS plan families do not match bootstrap mode")
        if {pair.family for pair in self.scout_pairs} != expected_families:
            raise ValueError("FOCUS scout pairs do not cover the planned families")

    @property
    def directions(self) -> int:
        return self.basis.shape[1]

    def reconstruct_raw_derivatives(self, orthonormal_coordinates: Tensor) -> Tensor:
        """Convert derivatives along Q into derivatives along raw ``Q R`` columns."""

        if not isinstance(orthonormal_coordinates, Tensor) or orthonormal_coordinates.ndim < 1:
            raise TypeError("directional coordinates must be a torch.Tensor")
        if orthonormal_coordinates.shape[-1] != self.directions:
            raise ValueError(
                f"directional coordinates have width {orthonormal_coordinates.shape[-1]}; "
                f"expected {self.directions}"
            )
        transform = self.reconstruction.to(
            device=orthonormal_coordinates.device,
            dtype=orthonormal_coordinates.dtype,
        )
        return orthonormal_coordinates @ transform

    def raw_direction(self, index: int) -> Tensor:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.directions:
            raise IndexError("raw direction index is out of range")
        return self.basis @ self.reconstruction[:, index]

    def projected_vector(self, orthonormal_coordinates: Tensor) -> Tensor:
        """Lift q directional coordinates into the full LoRA vector."""

        if orthonormal_coordinates.ndim != 1 or orthonormal_coordinates.numel() != self.directions:
            raise ValueError("projected coordinates must be a q-vector")
        return self.basis @ orthonormal_coordinates.to(
            device=self.basis.device,
            dtype=self.basis.dtype,
        )

    def cross_sketches_from_half_batches(
        self,
        partition: CanonicalLoRAPartition,
        first_half_coordinates: Tensor,
        second_half_coordinates: Tensor,
    ) -> tuple[FocusCrossSketch, ...]:
        """Reconstruct independent raw scouts and their half-batch derivatives.

        ``first_half_coordinates`` and ``second_half_coordinates`` must be the
        derivatives of two independent data halves along every orthonormal
        basis column.  For each family, raw scout one is paired only with the
        first half and raw scout two only with the second half.
        """

        if partition.layout_digest != self.partition_digest:
            raise ValueError("FOCUS plan was built for a different LoRA layout")
        for label, coordinates in (
            ("first", first_half_coordinates),
            ("second", second_half_coordinates),
        ):
            if not isinstance(coordinates, Tensor) or coordinates.ndim != 1:
                raise TypeError(f"{label}-half directional coordinates must be a vector")
            if coordinates.numel() != self.directions:
                raise ValueError(
                    f"{label}-half directional coordinates have {coordinates.numel()} values; "
                    f"expected {self.directions}"
                )
            if not torch.isfinite(coordinates).all():
                raise ValueError(f"{label}-half directional coordinates contain non-finite values")
        first_raw = self.reconstruct_raw_derivatives(first_half_coordinates)
        second_raw = self.reconstruct_raw_derivatives(second_half_coordinates)
        observations: list[FocusCrossSketch] = []
        for pair in self.scout_pairs:
            first_direction = partition.gather(self.raw_direction(pair.first_raw_column), pair.family)
            second_direction = partition.gather(
                self.raw_direction(pair.second_raw_column), pair.family
            )
            first_derivative = first_raw[pair.first_raw_column]
            second_derivative = second_raw[pair.second_raw_column]
            observations.append(
                FocusCrossSketch(
                    family=pair.family,
                    first=first_direction * first_derivative,
                    second=second_direction * second_derivative,
                    first_derivative=first_derivative,
                    second_derivative=second_derivative,
                )
            )
        return tuple(observations)


class MatchedFocusState:
    """Serializable rank-two A/B covariance state for the matched LoRA policy."""

    def __init__(
        self,
        partition: CanonicalLoRAPartition,
        config: FocusCovarianceConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.partition = partition
        self.config = config or FocusCovarianceConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self._families = {
            family: CrossSketchCovarianceEMA(
                partition.family_dimension(family),
                self.config,
                device=self.device,
                dtype=self.dtype,
            )
            for family in FOCUS_FAMILIES
        }

    def family_state(self, family: LoRAFamily) -> CrossSketchCovarianceEMA:
        return self._families[_validate_family(family)]

    @property
    def persistent_numel(self) -> int:
        return sum(state.persistent_numel for state in self._families.values())

    @property
    def persistent_numel_cap(self) -> int:
        rank = self.config.family_rank
        return self.partition.full_dimension * rank + len(FOCUS_FAMILIES) * rank

    @torch.inference_mode()
    def update_from_half_batch_coordinates(
        self,
        plan: FocusProbePlan,
        first_half_coordinates: Tensor,
        second_half_coordinates: Tensor,
    ) -> tuple[FocusCrossSketch, ...]:
        """Update family EMAs from independent half-batch directional probes."""

        observations = plan.cross_sketches_from_half_batches(
            self.partition,
            first_half_coordinates,
            second_half_coordinates,
        )
        for observation in observations:
            self.family_state(observation.family).update(observation.first, observation.second)
        if self.persistent_numel > self.persistent_numel_cap:
            raise RuntimeError("matched FOCUS state exceeded its O(Pk) storage cap")
        return observations

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "partition_digest": self.partition.layout_digest,
            "full_dimension": self.partition.full_dimension,
            "config": asdict(self.config),
            "families": {
                family: self.family_state(family).state_dict() for family in FOCUS_FAMILIES
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        partition: CanonicalLoRAPartition,
        value: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> MatchedFocusState:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported matched FOCUS state schema")
        if value.get("partition_digest") != partition.layout_digest:
            raise ValueError("matched FOCUS state belongs to a different LoRA layout")
        if value.get("full_dimension") != partition.full_dimension:
            raise ValueError("matched FOCUS state has the wrong full dimension")
        config_value = value.get("config")
        if not isinstance(config_value, Mapping):
            raise TypeError("matched FOCUS state has no config mapping")
        config = FocusCovarianceConfig(**dict(config_value))
        families = value.get("families")
        if not isinstance(families, Mapping) or set(families) != set(FOCUS_FAMILIES):
            raise ValueError("matched FOCUS state must contain exactly the A/B families")
        family_states = {
            family: CrossSketchCovarianceEMA.from_state_dict(families[family], device=device)
            for family in FOCUS_FAMILIES
        }
        dtypes = {family_state.dtype for family_state in family_states.values()}
        if len(dtypes) != 1:
            raise ValueError("matched FOCUS family states use inconsistent dtypes")
        state = cls(partition, config, device=device, dtype=next(iter(dtypes)))
        for family in FOCUS_FAMILIES:
            expected = partition.family_dimension(family)
            if family_states[family].dimension != expected:
                raise ValueError(
                    f"matched FOCUS family-{family} dimension is "
                    f"{family_states[family].dimension}; expected {expected}"
                )
            if family_states[family].config != config:
                raise ValueError(f"matched FOCUS family-{family} config is inconsistent")
        state._families = family_states
        if state.persistent_numel > state.persistent_numel_cap:
            raise ValueError("serialized matched FOCUS state exceeds its storage cap")
        return state


def _random_matrix(
    rows: int,
    columns: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    generator_device = torch.device(generator.device)
    values = torch.randn(
        rows,
        columns,
        generator=generator,
        device=generator_device,
        dtype=torch.float32,
    )
    return values.to(device=device, dtype=dtype)


def _canonical_qr(matrix: Tensor) -> tuple[Tensor, Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("FOCUS raw basis must be tall enough for reduced QR")
    q_basis, reconstruction = torch.linalg.qr(matrix.float(), mode="reduced")
    diagonal = torch.diag(reconstruction)
    scale = max(float(matrix.norm(dim=0).amax().item()), 1.0)
    if bool((diagonal.abs() <= 1.0e-7 * scale).any()):
        raise RuntimeError("FOCUS raw directions are numerically rank deficient")
    signs = torch.where(
        diagonal < 0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    q_basis = q_basis * signs
    reconstruction = reconstruction * signs[:, None]
    return q_basis.to(dtype=matrix.dtype), reconstruction.to(dtype=matrix.dtype)


@torch.inference_mode()
def build_focus_probe_plan(
    partition: CanonicalLoRAPartition,
    state: MatchedFocusState,
    generator: torch.Generator,
    *,
    bootstrap_b_only: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> FocusProbePlan:
    """Build the deterministic q8 QR basis used for one FOCUS probe round.

    In steady state, each family receives two learned (or random fill)
    directions and two independent raw Gaussian scouts.  During standard-LoRA
    bootstrap, all eight raw directions lie in B because A's derivative is
    exactly zero while B is zero.
    """

    if state.partition.layout_digest != partition.layout_digest:
        raise ValueError("FOCUS state and requested LoRA partition do not match")
    if state.config.family_rank != 2:
        raise ValueError("matched FOCUS probe construction requires family rank two")
    if not dtype.is_floating_point:
        raise TypeError("FOCUS probe dtype must be floating point")
    target_device = state.device if device is None else torch.device(device)
    local_blocks: list[tuple[LoRAFamily, Tensor, Tensor, tuple[ProbeKind, ...]]] = []
    local_scout_columns: dict[LoRAFamily, tuple[int, int]] = {}

    if bootstrap_b_only:
        family: LoRAFamily = "B"
        dimension = partition.family_dimension(family)
        if dimension < 8:
            raise ValueError("LoRA B family needs at least eight coordinates for q8 bootstrap")
        raw = _random_matrix(dimension, 8, generator, target_device, dtype)
        q_basis, reconstruction = _canonical_qr(raw)
        kinds = cast(tuple[ProbeKind, ...], tuple("bootstrap" for _ in range(8)))
        local_blocks.append((family, q_basis, reconstruction, kinds))
        local_scout_columns[family] = (0, 1)
    else:
        for family in FOCUS_FAMILIES:
            dimension = partition.family_dimension(family)
            if dimension < 4:
                raise ValueError(f"LoRA family-{family} needs at least four coordinates for q8")
            learned = state.family_state(family).basis.to(device=target_device, dtype=dtype)
            if learned.shape[1] > 2:
                raise RuntimeError(f"FOCUS family-{family} state exceeds rank two")
            fill_count = 2 - learned.shape[1]
            fill = _random_matrix(dimension, fill_count, generator, target_device, dtype)
            scouts = _random_matrix(dimension, 2, generator, target_device, dtype)
            raw = torch.cat([learned, fill, scouts], dim=1)
            kinds = cast(
                tuple[ProbeKind, ...],
                tuple("active" for _ in range(learned.shape[1]))
                + tuple("fill" for _ in range(fill_count))
                + ("scout", "scout"),
            )
            q_basis, reconstruction = _canonical_qr(raw)
            local_blocks.append((family, q_basis, reconstruction, kinds))
            local_scout_columns[family] = (2, 3)

    total_directions = sum(block[1].shape[1] for block in local_blocks)
    if total_directions != 8:
        raise RuntimeError(f"matched FOCUS plan produced q={total_directions}, expected q=8")
    basis = torch.empty(
        partition.full_dimension,
        total_directions,
        device=target_device,
        dtype=dtype,
    )
    reconstruction = torch.zeros(
        total_directions,
        total_directions,
        device=target_device,
        dtype=dtype,
    )
    raw_families: list[LoRAFamily] = []
    raw_kinds: list[ProbeKind] = []
    scout_pairs: list[FocusScoutPair] = []
    offset = 0
    for family, local_basis, local_reconstruction, kinds in local_blocks:
        width = local_basis.shape[1]
        basis[:, offset : offset + width] = partition.embed(local_basis, family)
        reconstruction[offset : offset + width, offset : offset + width] = local_reconstruction
        raw_families.extend([family] * width)
        raw_kinds.extend(kinds)
        first_local, second_local = local_scout_columns[family]
        scout_pairs.append(
            FocusScoutPair(
                family=family,
                first_raw_column=offset + first_local,
                second_raw_column=offset + second_local,
            )
        )
        offset += width
    return FocusProbePlan(
        basis=basis.contiguous(),
        reconstruction=reconstruction.contiguous(),
        raw_families=tuple(raw_families),
        raw_kinds=tuple(raw_kinds),
        scout_pairs=tuple(scout_pairs),
        partition_digest=partition.layout_digest,
        bootstrap_b_only=bootstrap_b_only,
    )


__all__ = [
    "FOCUS_FAMILIES",
    "CanonicalLoRAPartition",
    "CrossSketchCovarianceEMA",
    "FocusCovarianceConfig",
    "FocusCrossSketch",
    "FocusProbePlan",
    "FocusScoutPair",
    "LoRAFamily",
    "LoRAFamilySpan",
    "MatchedFocusState",
    "build_focus_probe_plan",
    "canonical_lora_partition",
    "requires_b_only_bootstrap",
    "symmetric_cross_sketch_action",
]
