"""Synthetic two-digit checksum task used for fast RLVR comparisons.

The task covers all 100 ordered pairs of decimal digits.  Its exact verifier
makes rewards cheap and deterministic, while its ten balanced target classes
make small train/validation/test splits straightforward to compare.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DIGITS: tuple[int, ...] = tuple(range(10))
CANDIDATE_ACTIONS: tuple[str, ...] = tuple(str(digit) for digit in DIGITS)
CODEBOOK_WORDS: tuple[str, ...] = ("amber", "cobalt", "jade", "ruby")
CODEBOOK_ACTIONS: tuple[int, ...] = (2, 0, 3, 1)

_PROMPT_TEMPLATE = """Compute the checksum for these two digits.
Rule: c = (3*a + 5*b + 1) mod 10
a = {a}
b = {b}
Reply with exactly one digit from 0 to 9."""


def _validate_digit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer digit")
    if value not in DIGITS:
        raise ValueError(f"{name} must be between 0 and 9 inclusive")


def checksum_target(a: int, b: int) -> int:
    """Return the verified target ``(3*a + 5*b + 1) mod 10``."""

    _validate_digit("a", a)
    _validate_digit("b", b)
    return (3 * a + 5 * b + 1) % 10


def format_prompt(a: int, b: int) -> str:
    """Format a stable prompt without including the answer."""

    _validate_digit("a", a)
    _validate_digit("b", b)
    return _PROMPT_TEMPLATE.format(a=a, b=b)


def codebook_context(a: int, b: int) -> int:
    """Map the 100 digit pairs evenly onto four observable contexts."""

    _validate_digit("a", a)
    _validate_digit("b", b)
    return (10 * a + b) % len(CODEBOOK_WORDS)


def codebook_target(a: int, b: int) -> int:
    """Return the hidden action associated with a pair's code word."""

    return CODEBOOK_ACTIONS[codebook_context(a, b)]


def format_codebook_prompt(a: int, b: int) -> str:
    """Format a contextual-bandit prompt without revealing the codebook."""

    word = CODEBOOK_WORDS[codebook_context(a, b)]
    return (
        "A hidden verifier maps each signal word to one action.\n"
        f"Signal word: {word}\n"
        "Choose exactly one action digit from 0 to 3."
    )


@dataclass(frozen=True, slots=True, order=True)
class ChecksumExample:
    """One immutable input pair and its mechanically derived target."""

    a: int
    b: int
    target: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", checksum_target(self.a, self.b))

    @property
    def prompt(self) -> str:
        return format_prompt(self.a, self.b)

    @property
    def example_id(self) -> str:
        return f"a{self.a}_b{self.b}"


@dataclass(frozen=True, slots=True)
class ChecksumSplits:
    """Deterministic train/validation/test partitions of all input pairs."""

    train: tuple[ChecksumExample, ...]
    val: tuple[ChecksumExample, ...]
    test: tuple[ChecksumExample, ...]

    def as_dict(self) -> dict[str, tuple[ChecksumExample, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def generate_examples() -> tuple[ChecksumExample, ...]:
    """Generate all 100 ordered pairs in lexicographic input order."""

    return tuple(ChecksumExample(a, b) for a in DIGITS for b in DIGITS)


def _validate_split_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def make_checksum_splits(
    *,
    seed: int = 0,
    train_per_target: int = 8,
    val_per_target: int = 1,
    test_per_target: int = 1,
) -> ChecksumSplits:
    """Create seeded splits stratified by the ten target values.

    Membership changes with ``seed``, but tuple order is always lexicographic so
    downstream runs cannot accidentally depend on hash or shuffle iteration.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    for name, count in (
        ("train_per_target", train_per_target),
        ("val_per_target", val_per_target),
        ("test_per_target", test_per_target),
    ):
        _validate_split_count(name, count)
    if train_per_target + val_per_target + test_per_target != 10:
        raise ValueError("per-target split counts must sum to 10")

    by_target: dict[int, list[ChecksumExample]] = {digit: [] for digit in DIGITS}
    for example in generate_examples():
        by_target[example.target].append(example)

    rng = random.Random(seed)
    train: list[ChecksumExample] = []
    val: list[ChecksumExample] = []
    test: list[ChecksumExample] = []
    for target in DIGITS:
        stratum = by_target[target]
        rng.shuffle(stratum)
        train_end = train_per_target
        val_end = train_end + val_per_target
        train.extend(stratum[:train_end])
        val.extend(stratum[train_end:val_end])
        test.extend(stratum[val_end:])

    return ChecksumSplits(
        train=tuple(sorted(train)),
        val=tuple(sorted(val)),
        test=tuple(sorted(test)),
    )


def parse_candidate_action(candidate: Any) -> int | None:
    """Parse a candidate into a digit, or return ``None`` when it is invalid.

    Generated text may contain surrounding whitespace, but explanations,
    multi-digit strings, booleans, and non-integral numeric values are rejected.
    """

    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return candidate if candidate in DIGITS else None
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    if len(stripped) != 1 or stripped not in CANDIDATE_ACTIONS:
        return None
    return int(stripped)


def score_candidate(example: ChecksumExample, candidate: Any) -> float:
    """Return the exact binary RLVR reward for one candidate completion."""

    action = parse_candidate_action(candidate)
    return float(action is not None and action == example.target)


def score_candidates(example: ChecksumExample, candidates: Sequence[Any]) -> tuple[float, ...]:
    """Score a rollout group while preserving candidate order."""

    return tuple(score_candidate(example, candidate) for candidate in candidates)


def group_leave_one_out_advantages(
    rewards: Sequence[float],
) -> tuple[float, ...]:
    """Subtract each sample's mean reward over the rest of its rollout group.

    For group size ``n``, this is equivalent to ``n / (n - 1)`` times the
    centered reward.  It sums to zero and avoids using an action in its own
    baseline.
    """

    if len(rewards) < 2:
        raise ValueError("leave-one-out advantages require at least two rewards")
    numeric_rewards: list[float] = []
    for reward in rewards:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError("rewards must contain only real numbers")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward):
            raise ValueError("rewards must be finite")
        numeric_rewards.append(numeric_reward)

    total = math.fsum(numeric_rewards)
    denominator = len(numeric_rewards) - 1
    return tuple(reward - (total - reward) / denominator for reward in numeric_rewards)
