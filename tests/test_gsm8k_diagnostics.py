from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rl_no_backward.gsm8k_diagnostics import (
    FINITE_DIFFERENCE_MUS,
    audit_sequence_forward_only_source,
    build_parser,
    fisher_trace_comparison,
    random_subspace_capture,
    reward_group_statistics,
    vector_agreement,
    write_gsm8k_diagnostic_results,
)


def test_vector_agreement_reports_cosine_and_relative_l2_error() -> None:
    result = vector_agreement(
        torch.tensor([3.0, 4.0]),
        torch.tensor([3.0, 3.0]),
    )

    assert result["reference_nonzero"] is True
    assert result["estimate_nonzero"] is True
    assert result["absolute_l2_error"] == pytest.approx(1.0)
    assert result["relative_l2_error"] == pytest.approx(0.2)
    assert result["cosine_similarity"] == pytest.approx(21.0 / (5.0 * math.sqrt(18.0)))


def test_vector_agreement_marks_zero_reference_metrics_undefined() -> None:
    result = vector_agreement(torch.zeros(3), torch.zeros(3))

    assert result["reference_nonzero"] is False
    assert result["estimate_nonzero"] is False
    assert result["relative_l2_error"] is None
    assert result["cosine_similarity"] is None


def test_random_subspace_capture_uses_squared_gradient_norm() -> None:
    gradient = torch.tensor([3.0, 4.0, 12.0])
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    result = random_subspace_capture(gradient, basis)

    assert result["parameter_count"] == 3
    assert result["directions"] == 2
    assert result["projected_gradient_l2_norm"] == pytest.approx(5.0)
    assert result["full_gradient_l2_norm"] == pytest.approx(13.0)
    assert result["captured_squared_norm_fraction"] == pytest.approx(25.0 / 169.0)
    assert result["expected_isotropic_squared_norm_fraction"] == pytest.approx(2.0 / 3.0)


def test_random_subspace_capture_rejects_nonorthonormal_columns() -> None:
    with pytest.raises(ValueError, match="not orthonormal"):
        random_subspace_capture(
            torch.ones(2),
            torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
        )


def test_fisher_trace_distinguishes_token_and_completion_outer_products() -> None:
    scores = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 2.0], [99.0, 99.0]],
                [[1.0, 1.0], [99.0, 99.0], [99.0, 99.0]],
            ]
        ]
    )
    mask = torch.tensor([[[True, True, False], [True, False, False]]])

    result = fisher_trace_comparison(scores, mask)

    assert result["correct_token_fisher_trace"] == pytest.approx(2.25)
    assert result["legacy_completion_outer_fisher_trace"] == pytest.approx(1.625)
    assert result["legacy_to_correct_trace_ratio"] == pytest.approx(1.625 / 2.25)
    assert result["legacy_completion_score_normalization"] == "mean_token_score"


def test_reward_group_statistics_reports_exact_and_zero_groups() -> None:
    rewards = torch.tensor([[0.0, 0.1, 0.1], [0.2, 0.2, 0.2]])
    exact = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    result = reward_group_statistics(rewards, exact_rewards=exact)

    assert result["samples"] == 6
    assert result["reward_mean"] == pytest.approx(0.8 / 6.0)
    assert result["zero_reward_fraction"] == pytest.approx(1.0 / 6.0)
    assert result["zero_advantage_groups"] == 1
    assert result["zero_advantage_group_fraction"] == pytest.approx(0.5)
    assert result["exact_reward_rate"] == pytest.approx(1.0 / 6.0)


def test_sequence_forward_only_source_passes_reverse_mode_audit() -> None:
    result = audit_sequence_forward_only_source()

    assert result.passed, result.violations
    assert result.violations == ()
    assert Path(result.source_path).name == "sequence_forward_only.py"


def test_diagnostic_writer_emits_standard_json(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "gsm8k_diagnostic.json"
    report = {
        "mus": FINITE_DIFFERENCE_MUS,
        "tensor": torch.tensor([1.0, 2.0]),
        "undefined": float("nan"),
    }

    output = write_gsm8k_diagnostic_results(destination, report)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == destination
    assert payload == {
        "mus": [0.25, 0.5, 1.0, 2.0],
        "tensor": [1.0, 2.0],
        "undefined": None,
    }


def test_cli_exposes_output_model_and_max_tokens() -> None:
    arguments = build_parser().parse_args(
        [
            "--output",
            "result.json",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--max-tokens",
            "17",
        ]
    )

    assert arguments.output == Path("result.json")
    assert arguments.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert arguments.max_tokens == 17
