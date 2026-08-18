from __future__ import annotations

from collections import Counter

import pytest

from rl_no_backward.task import (
    CANDIDATE_ACTIONS,
    CODEBOOK_ACTIONS,
    CODEBOOK_WORDS,
    ChecksumExample,
    checksum_target,
    codebook_context,
    codebook_target,
    format_codebook_prompt,
    format_prompt,
    generate_examples,
    group_leave_one_out_advantages,
    make_checksum_splits,
    parse_candidate_action,
    score_candidate,
    score_candidates,
)


def test_checksum_target_uses_declared_formula() -> None:
    assert checksum_target(0, 0) == 1
    assert checksum_target(1, 0) == 4
    assert checksum_target(0, 1) == 6
    assert checksum_target(9, 9) == 3

    with pytest.raises(ValueError):
        checksum_target(10, 0)
    with pytest.raises(TypeError):
        checksum_target(True, 0)


def test_complete_dataset_is_balanced_and_unique() -> None:
    examples = generate_examples()

    assert len(examples) == 100
    assert len({(example.a, example.b) for example in examples}) == 100
    assert Counter(example.target for example in examples) == Counter(
        {target: 10 for target in range(10)}
    )


def test_codebook_bandit_is_balanced_and_contextual() -> None:
    examples = generate_examples()

    assert CODEBOOK_WORDS == ("amber", "cobalt", "jade", "ruby")
    assert CODEBOOK_ACTIONS == (2, 0, 3, 1)
    assert Counter(codebook_context(e.a, e.b) for e in examples) == Counter(
        {context: 25 for context in range(4)}
    )
    assert Counter(codebook_target(e.a, e.b) for e in examples) == Counter(
        {action: 25 for action in range(4)}
    )
    prompt = format_codebook_prompt(0, 0)
    assert "amber" in prompt
    assert str(codebook_target(0, 0)) not in prompt


def test_splits_are_deterministic_disjoint_and_stratified() -> None:
    first = make_checksum_splits(seed=2026)
    repeated = make_checksum_splits(seed=2026)
    different_seed = make_checksum_splits(seed=2027)

    assert first == repeated
    assert first != different_seed
    assert tuple(map(len, (first.train, first.val, first.test))) == (80, 10, 10)

    partitions = [set(first.train), set(first.val), set(first.test)]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert set.union(*partitions) == set(generate_examples())

    for split, expected_per_target in (
        (first.train, 8),
        (first.val, 1),
        (first.test, 1),
    ):
        assert Counter(example.target for example in split) == Counter(
            {target: expected_per_target for target in range(10)}
        )


def test_prompt_format_is_stable_and_answer_free() -> None:
    expected = """Compute the checksum for these two digits.
Rule: c = (3*a + 5*b + 1) mod 10
a = 3
b = 4
Reply with exactly one digit from 0 to 9."""

    assert format_prompt(3, 4) == expected
    assert ChecksumExample(3, 4).prompt == expected


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("0", 0),
        (" 9\n", 9),
        (4, 4),
        ("10", None),
        ("answer: 4", None),
        (4.0, None),
        (True, None),
        (None, None),
    ],
)
def test_candidate_action_parsing(candidate: object, expected: int | None) -> None:
    assert parse_candidate_action(candidate) == expected


def test_candidate_actions_and_exact_rewards() -> None:
    example = ChecksumExample(1, 2)  # target = 4

    assert CANDIDATE_ACTIONS == tuple(str(digit) for digit in range(10))
    assert score_candidate(example, "4") == 1.0
    assert score_candidate(example, " 4\n") == 1.0
    assert score_candidate(example, "answer: 4") == 0.0
    assert score_candidates(example, ["4", "3", "invalid"]) == (1.0, 0.0, 0.0)


def test_group_leave_one_out_advantages() -> None:
    advantages = group_leave_one_out_advantages([1.0, 0.0, 0.0, 1.0])

    assert advantages == pytest.approx((2 / 3, -2 / 3, -2 / 3, 2 / 3))
    assert sum(advantages) == pytest.approx(0.0)
    assert group_leave_one_out_advantages([0.2, 0.8]) == pytest.approx((-0.6, 0.6))

    with pytest.raises(ValueError, match="at least two"):
        group_leave_one_out_advantages([1.0])
    with pytest.raises(ValueError, match="finite"):
        group_leave_one_out_advantages([1.0, float("nan")])
