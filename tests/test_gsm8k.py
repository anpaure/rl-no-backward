from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from rl_no_backward.gsm8k import (
    DifficultyFilter,
    GSM8KExample,
    build_messages,
    exact_match_reward,
    extract_model_answer,
    extract_reference_answer,
    filter_by_difficulty,
    format_prompt,
    load_gsm8k_split,
    load_gsm8k_subsets,
    make_gsm8k_subsets,
    normalize_numeric_answer,
    select_seeded_subset,
    verify_answer,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("1,500.00", "1500"),
        ("+001.50", "3/2"),
        ("−12", "-12"),
        (".125", "1/8"),
        ("2/4", "1/2"),
        (r"\frac{-3}{6}", "-1/2"),
        ("-1 1/2", "-3/2"),
        ("1e3", "1000"),
        ("$2,000", "2000"),
        (Decimal("0.20"), "1/5"),
        (Fraction(6, 8), "3/4"),
    ],
)
def test_numeric_normalization_is_exact(raw: object, canonical: str) -> None:
    assert normalize_numeric_answer(raw) == canonical  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["", "twelve", "12,34", "1/0", float("nan"), True])
def test_numeric_normalization_rejects_invalid_values(raw: object) -> None:
    assert normalize_numeric_answer(raw) is None  # type: ignore[arg-type]


def test_reference_extraction_requires_final_hash_marker() -> None:
    solution = "First compute 20 / 4 = 5.\nThen add 1.\n#### $6.00"

    assert extract_reference_answer(solution) == "6"
    assert extract_reference_answer("The answer is 6") is None
    assert extract_reference_answer("old #### 1\ncorrected #### 3/4") == "3/4"
    assert extract_reference_answer("reasoning\n#### not-a-number") is None


def test_model_extraction_prioritizes_boxed_then_final_answer_then_last_number() -> None:
    assert extract_model_answer(r"We saw 99. Therefore \boxed{\frac{3}{4}}. Check 100.") == "3/4"
    assert extract_model_answer("2 + 3 = 5. Final answer: -1,200.50. Check: 7") == "-2401/2"
    assert extract_model_answer("There were 8, then there were 11.") == "11"
    assert extract_model_answer("No numeric answer was produced.") is None

    # A valid boxed value wins even when a later final-answer phrase disagrees.
    contradictory = r"\boxed{12}\nFinal answer: 13"
    assert extract_model_answer(contradictory) == "12"


def test_exact_verifier_accepts_equivalent_numeric_forms_only() -> None:
    reference_solution = "Half of one whole.\n#### 0.5"

    assert verify_answer(r"Reasoning. \boxed{1/2}", reference_solution)
    assert verify_answer("Final answer: 0.500", "1/2")
    assert not verify_answer("Final answer: 0.51", reference_solution)
    assert not verify_answer("I cannot solve it.", reference_solution)
    assert exact_match_reward("Final answer: 1/2", reference_solution) == 1.0
    assert exact_match_reward("Final answer: 2", reference_solution) == 0.0


def _example(index: int, *, lines: int | None = None, answer: str | None = None) -> GSM8KExample:
    line_count = lines if lines is not None else index % 4 + 1
    final = answer if answer is not None else str((index + 1) * 10)
    reasoning = "\n".join(f"step {line}" for line in range(line_count))
    return GSM8KExample(
        question=f"Question {index}?",
        answer=f"{reasoning}\n#### {final}",
        split="train",
        source_index=index,
    )


def test_example_metadata_and_difficulty_filter_are_reference_derived() -> None:
    examples = (
        _example(0, lines=1, answer="5"),
        _example(1, lines=2, answer="50"),
        _example(2, lines=3, answer="500"),
        _example(3, lines=4, answer="5000"),
    )

    assert examples[1].canonical_answer == "50"
    assert examples[1].final_answer == "50"
    assert examples[1].reasoning_lines == 2
    assert examples[1].answer_magnitude == Fraction(50)

    difficulty = DifficultyFilter(
        min_reasoning_lines=2,
        max_reasoning_lines=3,
        min_answer_magnitude=10,
        max_answer_magnitude=500,
    )
    assert filter_by_difficulty(examples, difficulty) == examples[1:3]
    assert (
        filter_by_difficulty(
            examples,
            min_reasoning_lines=2,
            max_answer_magnitude=500,
        )
        == examples[1:3]
    )

    with pytest.raises(ValueError, match="minimum reasoning"):
        DifficultyFilter(min_reasoning_lines=3, max_reasoning_lines=2)
    with pytest.raises(ValueError, match="magnitude"):
        DifficultyFilter(min_answer_magnitude=-1)


def test_seeded_selection_is_fixed_and_rng_independent() -> None:
    examples = tuple(_example(index) for index in range(20))

    first = select_seeded_subset(examples, 7, seed=2026, namespace="train")
    repeated = select_seeded_subset(examples, 7, seed=2026, namespace="train")
    different_seed = select_seeded_subset(examples, 7, seed=2027, namespace="train")
    eval_namespace = select_seeded_subset(examples, 7, seed=2026, namespace="eval")

    assert first == repeated
    assert first != different_seed
    assert first != eval_namespace
    assert len(first) == len(set(first)) == 7
    with pytest.raises(ValueError, match="only 20"):
        select_seeded_subset(examples, 21, seed=0)


def test_train_eval_subsets_filter_then_select_independently() -> None:
    train = tuple(_example(index) for index in range(20))
    eval_examples = tuple(
        GSM8KExample(
            question=f"Eval {index}?",
            answer=f"one line\n#### {index + 1}",
            split="test",
            source_index=index,
        )
        for index in range(10)
    )
    subsets = make_gsm8k_subsets(
        train,
        eval_examples,
        train_size=5,
        eval_size=3,
        seed=123,
        difficulty=DifficultyFilter(max_reasoning_lines=2),
    )

    assert len(subsets.train) == 5
    assert len(subsets.eval) == 3
    assert all(example.reasoning_lines <= 2 for example in subsets.train + subsets.eval)
    assert subsets == make_gsm8k_subsets(
        train,
        eval_examples,
        train_size=5,
        eval_size=3,
        seed=123,
        difficulty=DifficultyFilter(max_reasoning_lines=2),
    )


def test_prompt_is_plain_or_uses_injected_chat_formatter() -> None:
    question = "Mia has 2 apples and gets 3 more. How many?"
    messages = build_messages(question)

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": question}
    plain = format_prompt(question)
    assert question in plain
    assert "Solution:" in plain
    assert "Final answer" in plain

    observed: list[object] = []

    def formatter(received: object) -> str:
        observed.append(received)
        return "<chat-template-output>"

    assert format_prompt(question, formatter) == "<chat-template-output>"
    assert observed == [messages]


def test_dataset_loading_is_lazy_and_fully_injectable_offline() -> None:
    calls: list[dict[str, object]] = []
    rows = {
        "train": [
            {"question": f"Train question {index}?", "answer": f"work\n#### {index}"}
            for index in range(6)
        ],
        "test": [
            {"question": f"Test question {index}?", "answer": f"work\n#### {index}"}
            for index in range(4)
        ],
    }

    def fake_loader(**kwargs: object) -> list[dict[str, str]]:
        calls.append(dict(kwargs))
        return rows[str(kwargs["split"])]

    loaded = load_gsm8k_split("train", dataset_loader=fake_loader, cache_dir="offline-cache")
    assert len(loaded) == 6
    assert loaded[0].split == "train"
    assert loaded[0].source_index == 0
    assert calls == [
        {
            "path": "openai/gsm8k",
            "name": "main",
            "split": "train",
            "cache_dir": "offline-cache",
        }
    ]

    calls.clear()
    subsets = load_gsm8k_subsets(
        train_size=3,
        eval_size=2,
        seed=9,
        dataset_loader=fake_loader,
    )
    assert len(subsets.train) == 3
    assert len(subsets.eval) == 2
    assert [call["split"] for call in calls] == ["train", "test"]


def test_malformed_dataset_rows_fail_with_context() -> None:
    def missing_answer(**_: object) -> list[dict[str, str]]:
        return [{"question": "Question?"}]

    with pytest.raises(ValueError, match="row 0.*answer"):
        load_gsm8k_split("train", dataset_loader=missing_answer)

    with pytest.raises(ValueError, match="numeric value"):
        GSM8KExample(question="Question?", answer="No final marker")
