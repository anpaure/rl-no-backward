"""Validated configuration for the synthetic RLVR task and its rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import ChecksumSplits


def _require_plain_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Configuration for the complete 10-by-10 checksum dataset.

    Every checksum class has ten examples, so the three per-class counts must
    sum to ten.  The defaults create globally sized 80/10/10 splits.
    """

    split_seed: int = 0
    train_per_target: int = 8
    val_per_target: int = 1
    test_per_target: int = 1

    def __post_init__(self) -> None:
        _require_plain_int("split_seed", self.split_seed, minimum=0)
        _require_plain_int("train_per_target", self.train_per_target, minimum=0)
        _require_plain_int("val_per_target", self.val_per_target, minimum=0)
        _require_plain_int("test_per_target", self.test_per_target, minimum=0)
        per_target = self.train_per_target + self.val_per_target + self.test_per_target
        if per_target != 10:
            raise ValueError("train_per_target + val_per_target + test_per_target must equal 10")

    def build_splits(self) -> ChecksumSplits:
        """Materialize the configured deterministic, stratified splits."""

        # The local import keeps configuration independent of task construction
        # at module import time and avoids a circular import.
        from .task import make_checksum_splits

        splits: ChecksumSplits = make_checksum_splits(
            seed=self.split_seed,
            train_per_target=self.train_per_target,
            val_per_target=self.val_per_target,
            test_per_target=self.test_per_target,
        )
        return splits


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Task-level sampling settings shared by RLVR training methods."""

    group_size: int = 8
    max_new_tokens: int = 4
    temperature: float = 1.0

    def __post_init__(self) -> None:
        _require_plain_int("group_size", self.group_size, minimum=2)
        _require_plain_int("max_new_tokens", self.max_new_tokens, minimum=1)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
