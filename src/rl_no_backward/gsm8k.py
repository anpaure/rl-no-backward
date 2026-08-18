"""GSM8K data, prompting, and exact-reward utilities.

The helpers in this module deliberately do not import ``datasets`` or
``transformers`` at module import time.  Dataset loading is lazy, and prompt
formatting accepts a plain callable so training code can opt into a tokenizer's
chat template without coupling the task definition to a particular model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Protocol, TypeAlias

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
GSM8K_TRAIN_SPLIT = "train"
GSM8K_EVAL_SPLIT = "test"

GSM8K_SYSTEM_PROMPT = (
    "Solve the grade-school math problem carefully. Show your reasoning, then "
    "end with `Final answer: <number>`."
)

NumericInput: TypeAlias = str | int | float | Decimal | Fraction
DatasetLoader: TypeAlias = Callable[..., Iterable[Mapping[str, Any]]]


class ChatFormatter(Protocol):
    """Pure interface for applying a model-specific chat template."""

    def __call__(self, messages: Sequence[Mapping[str, str]], /) -> str: ...


_DECIMAL_SOURCE = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+"
_EXPONENT_SOURCE = rf"(?:{_DECIMAL_SOURCE})(?:[eE][+-]?\d+)?"
_LATEX_FRACTION_SOURCE = (
    rf"[+-]?\s*\\(?:d?frac|tfrac)\s*\{{\s*[+-]?{_EXPONENT_SOURCE}\s*\}}"
    rf"\s*\{{\s*[+-]?{_EXPONENT_SOURCE}\s*\}}"
)
_PLAIN_NUMBER_SOURCE = rf"[+-]?\s*(?:{_EXPONENT_SOURCE})(?:\s*/\s*[+-]?\s*(?:{_EXPONENT_SOURCE}))?"
_MIXED_FRACTION_SOURCE = (
    rf"[+-]?\s*(?:{_EXPONENT_SOURCE})\s+(?:{_EXPONENT_SOURCE})"
    rf"\s*/\s*(?:{_EXPONENT_SOURCE})"
)
_NUMBER_CANDIDATE_RE = re.compile(
    rf"(?<![\w.])(?:{_LATEX_FRACTION_SOURCE}|{_MIXED_FRACTION_SOURCE}|"
    rf"\$?{_PLAIN_NUMBER_SOURCE}\s*%?)(?![\w,])"
)
_LATEX_FRACTION_RE = re.compile(
    rf"^(?P<sign>[+-]?)\s*\\(?:d?frac|tfrac)\s*"
    rf"\{{\s*(?P<numerator>[+-]?{_EXPONENT_SOURCE})\s*\}}\s*"
    rf"\{{\s*(?P<denominator>[+-]?{_EXPONENT_SOURCE})\s*\}}$"
)
_MIXED_FRACTION_RE = re.compile(
    rf"^(?P<sign>[+-]?)(?P<whole>{_EXPONENT_SOURCE})\s+"
    rf"(?P<numerator>{_EXPONENT_SOURCE})\s*/\s*"
    rf"(?P<denominator>{_EXPONENT_SOURCE})$"
)
_PLAIN_FRACTION_RE = re.compile(
    rf"^(?P<numerator>[+-]?{_EXPONENT_SOURCE})\s*/\s*"
    rf"(?P<denominator>[+-]?{_EXPONENT_SOURCE})$"
)
_PLAIN_DECIMAL_RE = re.compile(rf"^[+-]?{_EXPONENT_SOURCE}$")
_FINAL_ANSWER_RE = re.compile(r"\b(?:the\s+)?final\s+answer\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\b")


def _strip_balanced_outer_braces(text: str) -> str:
    """Remove redundant outer braces without damaging fraction braces."""

    result = text.strip()
    while result.startswith("{") and result.endswith("}"):
        depth = 0
        balanced_at_end = False
        for index, character in enumerate(result):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth < 0:
                    return result
                if depth == 0:
                    balanced_at_end = index == len(result) - 1
                    break
        if not balanced_at_end:
            break
        result = result[1:-1].strip()
    return result


def _prepare_numeric_text(value: str) -> str:
    text = value.strip()
    text = text.replace("−", "-").replace("﹣", "-").replace("－", "-")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\$", "$")
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    text = _strip_balanced_outer_braces(text)
    text = text.strip(" \t\r\n`")
    text = re.sub(r"^[=:]\s*", "", text)
    text = text.rstrip(";:!?")
    if text.endswith(r"\%"):
        text = text[:-2].rstrip()
    elif text.endswith("%"):
        text = text[:-1].rstrip()
    # A dollar pair is a math delimiter; a single leading dollar is currency.
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    elif text.startswith("$"):
        text = text[1:].strip()
    return _strip_balanced_outer_braces(text)


def _decimal_fraction(text: str) -> Fraction | None:
    """Parse one signed decimal/scientific token exactly."""

    compact = re.sub(r"\s+", "", text)
    signless = compact.lstrip("+-")
    mantissa = re.split(r"[eE]", signless, maxsplit=1)[0]
    integer_part = mantissa.split(".", maxsplit=1)[0]
    if "," in integer_part and not re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer_part):
        return None
    compact = compact.replace(",", "")
    if not _PLAIN_DECIMAL_RE.fullmatch(compact):
        return None
    try:
        return Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _parse_numeric_answer(value: NumericInput) -> Fraction | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value) if value.is_finite() else None
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Decimal(str(...)) reflects the human-facing float rather than its
        # binary expansion, which is the useful behavior for answer checking.
        return Fraction(Decimal(str(value)))
    if not isinstance(value, str):
        return None

    text = _prepare_numeric_text(value)
    latex_match = _LATEX_FRACTION_RE.fullmatch(text)
    if latex_match:
        numerator = _decimal_fraction(latex_match.group("numerator"))
        denominator = _decimal_fraction(latex_match.group("denominator"))
        if numerator is None or denominator in (None, 0):
            return None
        fraction = numerator / denominator
        return -fraction if latex_match.group("sign") == "-" else fraction

    mixed_match = _MIXED_FRACTION_RE.fullmatch(text)
    if mixed_match:
        whole = _decimal_fraction(mixed_match.group("whole"))
        numerator = _decimal_fraction(mixed_match.group("numerator"))
        denominator = _decimal_fraction(mixed_match.group("denominator"))
        if whole is None or numerator is None or denominator in (None, 0):
            return None
        fraction = whole + numerator / denominator
        return -fraction if mixed_match.group("sign") == "-" else fraction

    fraction_match = _PLAIN_FRACTION_RE.fullmatch(text)
    if fraction_match:
        numerator = _decimal_fraction(fraction_match.group("numerator"))
        denominator = _decimal_fraction(fraction_match.group("denominator"))
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    return _decimal_fraction(text)


def normalize_numeric_answer(value: NumericInput) -> str | None:
    """Return an exact, reduced representation of a numeric answer.

    Comma grouping, Unicode signs, decimal/scientific notation, ordinary or
    LaTeX fractions, and mixed fractions are accepted.  Equivalent values share
    a canonical representation: for example, ``"1,500.0"`` becomes ``"1500"``
    and both ``"0.5"`` and ``"2/4"`` become ``"1/2"``.
    """

    parsed = _parse_numeric_answer(value)
    if parsed is None:
        return None
    if parsed.denominator == 1:
        return str(parsed.numerator)
    return f"{parsed.numerator}/{parsed.denominator}"


def extract_reference_answer(solution: str) -> str | None:
    """Extract the canonical answer after the final GSM8K ``####`` marker."""

    if not isinstance(solution, str):
        return None
    marker_index = solution.rfind("####")
    if marker_index < 0:
        return None
    tail = solution[marker_index + 4 :].strip()
    direct = normalize_numeric_answer(tail)
    if direct is not None:
        return direct
    candidates = _numeric_candidates(tail)
    return candidates[0] if candidates else None


def _boxed_contents(text: str) -> tuple[str, ...]:
    contents: list[str] = []
    for match in _BOXED_RE.finditer(text):
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            continue
        if text[cursor] != "{":
            line_end = text.find("\n", cursor)
            contents.append(text[cursor : line_end if line_end >= 0 else len(text)])
            continue
        depth = 0
        start = cursor + 1
        for end in range(cursor, len(text)):
            character = text[end]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[start:end])
                    break
                if depth < 0:
                    break
    return tuple(contents)


def _numeric_candidates(text: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for match in _NUMBER_CANDIDATE_RE.finditer(text):
        answer = normalize_numeric_answer(match.group(0))
        if answer is not None:
            normalized.append(answer)
    return tuple(normalized)


def extract_model_answer(completion: str) -> str | None:
    """Extract a model's intended answer using explicit signals first.

    The last valid ``\\boxed{...}`` value has highest priority.  Next comes the
    first number following the last ``final answer`` phrase.  With neither
    signal present, the final numeric value in the completion is used.
    """

    if not isinstance(completion, str) or not completion.strip():
        return None

    boxed_answers: list[str] = []
    for content in _boxed_contents(completion):
        answer = normalize_numeric_answer(content)
        if answer is None:
            nested_candidates = _numeric_candidates(content)
            answer = nested_candidates[-1] if nested_candidates else None
        if answer is not None:
            boxed_answers.append(answer)
    if boxed_answers:
        return boxed_answers[-1]

    final_markers = tuple(_FINAL_ANSWER_RE.finditer(completion))
    if final_markers:
        tail = completion[final_markers[-1].end() :]
        candidates = _numeric_candidates(tail)
        if candidates:
            return candidates[0]

    candidates = _numeric_candidates(completion)
    return candidates[-1] if candidates else None


def _canonical_reference(reference: str | NumericInput | GSM8KExample) -> str | None:
    if isinstance(reference, GSM8KExample):
        return reference.canonical_answer
    if isinstance(reference, str) and "####" in reference:
        return extract_reference_answer(reference)
    return normalize_numeric_answer(reference)


def verify_answer(completion: str, reference: str | NumericInput | GSM8KExample) -> bool:
    """Return exact numeric agreement between a completion and a reference."""

    predicted = extract_model_answer(completion)
    expected = _canonical_reference(reference)
    return predicted is not None and expected is not None and predicted == expected


def exact_match_reward(
    completion: str,
    reference: str | NumericInput | GSM8KExample,
) -> float:
    """Return the binary RLVR reward for one GSM8K completion."""

    return float(verify_answer(completion, reference))


def _reasoning_line_count(solution: str) -> int:
    marker_index = solution.rfind("####")
    reasoning = solution[:marker_index] if marker_index >= 0 else solution
    return sum(bool(line.strip()) for line in reasoning.splitlines())


@dataclass(frozen=True, slots=True)
class GSM8KExample:
    """One validated GSM8K row with deterministic difficulty metadata."""

    question: str
    answer: str
    split: str = ""
    source_index: int = -1
    canonical_answer: str = field(init=False)
    reasoning_lines: int = field(init=False)
    answer_magnitude: Fraction = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        canonical = extract_reference_answer(self.answer)
        if canonical is None:
            raise ValueError("answer must contain a numeric value after '####'")
        if not isinstance(self.split, str):
            raise TypeError("split must be a string")
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int):
            raise TypeError("source_index must be an integer")
        object.__setattr__(self, "question", self.question.strip())
        object.__setattr__(self, "canonical_answer", canonical)
        object.__setattr__(self, "reasoning_lines", _reasoning_line_count(self.answer))
        object.__setattr__(self, "answer_magnitude", abs(Fraction(canonical)))

    @property
    def final_answer(self) -> str:
        """Alias useful in generic task code."""

        return self.canonical_answer

    @property
    def example_id(self) -> str:
        payload = f"{self.question}\0{self.answer}".encode()
        return hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class DifficultyFilter:
    """Inclusive deterministic bounds over reference-derived metadata."""

    min_reasoning_lines: int = 0
    max_reasoning_lines: int | None = None
    min_answer_magnitude: NumericInput = 0
    max_answer_magnitude: NumericInput | None = None
    _min_magnitude: Fraction = field(init=False, repr=False, compare=True)
    _max_magnitude: Fraction | None = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("min_reasoning_lines", self.min_reasoning_lines),
            ("max_reasoning_lines", self.max_reasoning_lines),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.max_reasoning_lines is not None
            and self.min_reasoning_lines > self.max_reasoning_lines
        ):
            raise ValueError("minimum reasoning lines cannot exceed maximum")

        min_magnitude = _parse_numeric_answer(self.min_answer_magnitude)
        max_magnitude = (
            None
            if self.max_answer_magnitude is None
            else _parse_numeric_answer(self.max_answer_magnitude)
        )
        if min_magnitude is None:
            raise ValueError("min_answer_magnitude must be numeric")
        if self.max_answer_magnitude is not None and max_magnitude is None:
            raise ValueError("max_answer_magnitude must be numeric or None")
        if min_magnitude < 0 or (max_magnitude is not None and max_magnitude < 0):
            raise ValueError("answer magnitude bounds must be non-negative")
        if max_magnitude is not None and min_magnitude > max_magnitude:
            raise ValueError("minimum answer magnitude cannot exceed maximum")
        object.__setattr__(self, "_min_magnitude", min_magnitude)
        object.__setattr__(self, "_max_magnitude", max_magnitude)

    def matches(self, example: GSM8KExample) -> bool:
        if example.reasoning_lines < self.min_reasoning_lines:
            return False
        if (
            self.max_reasoning_lines is not None
            and example.reasoning_lines > self.max_reasoning_lines
        ):
            return False
        if example.answer_magnitude < self._min_magnitude:
            return False
        return self._max_magnitude is None or example.answer_magnitude <= self._max_magnitude


def filter_by_difficulty(
    examples: Iterable[GSM8KExample],
    difficulty: DifficultyFilter | None = None,
    *,
    min_reasoning_lines: int = 0,
    max_reasoning_lines: int | None = None,
    min_answer_magnitude: NumericInput = 0,
    max_answer_magnitude: NumericInput | None = None,
) -> tuple[GSM8KExample, ...]:
    """Filter examples in source order using only reference-derived metadata."""

    if difficulty is not None and (
        min_reasoning_lines != 0
        or max_reasoning_lines is not None
        or min_answer_magnitude != 0
        or max_answer_magnitude is not None
    ):
        raise ValueError("pass a DifficultyFilter or individual bounds, not both")
    active_filter = difficulty or DifficultyFilter(
        min_reasoning_lines=min_reasoning_lines,
        max_reasoning_lines=max_reasoning_lines,
        min_answer_magnitude=min_answer_magnitude,
        max_answer_magnitude=max_answer_magnitude,
    )
    return tuple(example for example in examples if active_filter.matches(example))


def build_messages(question: str) -> tuple[dict[str, str], ...]:
    """Build model-neutral chat messages without answer leakage."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return (
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    )


def format_prompt(question: str, formatter: ChatFormatter | None = None) -> str:
    """Format a plain prompt, or delegate chat tokens to a pure formatter.

    A training caller can pass, for example, a small wrapper around
    ``tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True)``.  This module never guesses special tokens.
    """

    messages = build_messages(question)
    if formatter is not None:
        rendered = formatter(messages)
        if not isinstance(rendered, str):
            raise TypeError("formatter must return a string")
        return rendered
    return f"{GSM8K_SYSTEM_PROMPT}\n\nProblem:\n{messages[1]['content']}\n\nSolution:\n"


def _default_dataset_loader(**kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Loading GSM8K requires the 'datasets' package; install the project dependencies"
        ) from error
    return load_dataset(**kwargs)


def load_gsm8k_split(
    split: str,
    *,
    dataset_loader: DatasetLoader | None = None,
    **load_kwargs: Any,
) -> tuple[GSM8KExample, ...]:
    """Lazily load and validate one ``openai/gsm8k`` ``main`` split."""

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    reserved = {"path", "name", "split"}.intersection(load_kwargs)
    if reserved:
        joined = ", ".join(sorted(reserved))
        raise ValueError(f"dataset coordinates are fixed; do not pass {joined}")
    loader = dataset_loader or _default_dataset_loader
    rows = loader(
        path=GSM8K_DATASET_ID,
        name=GSM8K_DATASET_CONFIG,
        split=split,
        **load_kwargs,
    )
    examples: list[GSM8KExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"GSM8K row {index} is not a mapping")
        try:
            question = row["question"]
            answer = row["answer"]
        except KeyError as error:
            raise ValueError(f"GSM8K row {index} is missing {error.args[0]!r}") from error
        examples.append(
            GSM8KExample(
                question=question,
                answer=answer,
                split=split,
                source_index=index,
            )
        )
    return tuple(examples)


def _validate_subset_size(size: int, available: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 0:
        raise ValueError("size must be non-negative")
    if size > available:
        raise ValueError(f"requested {size} examples, but only {available} are available")


def select_seeded_subset(
    examples: Sequence[GSM8KExample],
    size: int,
    *,
    seed: int,
    namespace: str = "",
) -> tuple[GSM8KExample, ...]:
    """Select a fixed subset by stable SHA-256 ranking.

    Hash ranking avoids dependence on global RNG state, Python's salted hashes,
    or implementation details of a particular ``random`` release.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    _validate_subset_size(size, len(examples))

    ranked: list[tuple[bytes, int, GSM8KExample]] = []
    for index, example in enumerate(examples):
        key = f"{seed}\0{namespace}\0{example.example_id}".encode()
        ranked.append((hashlib.sha256(key).digest(), index, example))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[:size])


@dataclass(frozen=True, slots=True)
class GSM8KSubsets:
    """Small, fixed official-train and official-test subsets."""

    train: tuple[GSM8KExample, ...]
    eval: tuple[GSM8KExample, ...]

    def as_dict(self) -> dict[str, tuple[GSM8KExample, ...]]:
        return {"train": self.train, "eval": self.eval}


def make_gsm8k_subsets(
    train_examples: Sequence[GSM8KExample],
    eval_examples: Sequence[GSM8KExample],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
    difficulty: DifficultyFilter | None = None,
) -> GSM8KSubsets:
    """Filter and deterministically select independent train/eval subsets."""

    train_pool = filter_by_difficulty(train_examples, difficulty)
    eval_pool = filter_by_difficulty(eval_examples, difficulty)
    return GSM8KSubsets(
        train=select_seeded_subset(train_pool, train_size, seed=seed, namespace="train"),
        eval=select_seeded_subset(eval_pool, eval_size, seed=seed, namespace="eval"),
    )


def load_gsm8k_subsets(
    *,
    train_size: int,
    eval_size: int,
    seed: int,
    difficulty: DifficultyFilter | None = None,
    dataset_loader: DatasetLoader | None = None,
    **load_kwargs: Any,
) -> GSM8KSubsets:
    """Load official splits lazily, then create fixed small experiment subsets."""

    train_examples = load_gsm8k_split(
        GSM8K_TRAIN_SPLIT,
        dataset_loader=dataset_loader,
        **load_kwargs,
    )
    eval_examples = load_gsm8k_split(
        GSM8K_EVAL_SPLIT,
        dataset_loader=dataset_loader,
        **load_kwargs,
    )
    return make_gsm8k_subsets(
        train_examples,
        eval_examples,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
        difficulty=difficulty,
    )


# Explicit aliases make call sites read naturally without hiding semantics.
extract_gsm8k_reference_answer = extract_reference_answer
extract_gsm8k_model_answer = extract_model_answer
verify_gsm8k_answer = verify_answer
score_gsm8k_answer = exact_match_reward
filter_gsm8k_by_difficulty = filter_by_difficulty


__all__ = [
    "GSM8K_DATASET_CONFIG",
    "GSM8K_DATASET_ID",
    "GSM8K_EVAL_SPLIT",
    "GSM8K_SYSTEM_PROMPT",
    "GSM8K_TRAIN_SPLIT",
    "ChatFormatter",
    "DifficultyFilter",
    "GSM8KExample",
    "GSM8KSubsets",
    "build_messages",
    "exact_match_reward",
    "extract_gsm8k_model_answer",
    "extract_gsm8k_reference_answer",
    "extract_model_answer",
    "extract_reference_answer",
    "filter_by_difficulty",
    "filter_gsm8k_by_difficulty",
    "format_prompt",
    "load_gsm8k_split",
    "load_gsm8k_subsets",
    "make_gsm8k_subsets",
    "normalize_numeric_answer",
    "score_gsm8k_answer",
    "select_seeded_subset",
    "verify_answer",
    "verify_gsm8k_answer",
]
