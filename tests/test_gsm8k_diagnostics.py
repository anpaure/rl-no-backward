from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_no_backward.gsm8k_diagnostics import (
    DEFAULT_ADAPTER_LAYERS,
    DEFAULT_ADAPTER_RANK,
    DEFAULT_ATTENTION_IMPLEMENTATION,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_USE_FROZEN_PREFIX_SCORING,
    FINITE_DIFFERENCE_MUS,
    SEARCH_DIRECTIONS,
    _model_runtime_metadata,
    _prefix_cache_metadata,
    _teacher_forced_call_counters,
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


def test_cli_defaults_match_pinned_optimized_hf_diagnostic() -> None:
    arguments = build_parser().parse_args(["--output", "result.json"])

    assert arguments.model == DEFAULT_MODEL_NAME
    assert arguments.model_revision == DEFAULT_MODEL_REVISION
    assert arguments.dataset_revision == DEFAULT_DATASET_REVISION
    assert arguments.attention_implementation == DEFAULT_ATTENTION_IMPLEMENTATION
    assert arguments.adapter_rank == DEFAULT_ADAPTER_RANK
    assert arguments.adapter_layers == DEFAULT_ADAPTER_LAYERS
    assert arguments.max_tokens == DEFAULT_MAX_TOKENS == 512
    assert arguments.directions == SEARCH_DIRECTIONS == 8
    assert arguments.use_frozen_prefix_scoring is DEFAULT_USE_FROZEN_PREFIX_SCORING is True


def test_cli_exposes_output_model_max_tokens_and_directions() -> None:
    arguments = build_parser().parse_args(
        [
            "--output",
            "result.json",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--max-tokens",
            "17",
            "--directions",
            "8",
            "--model-revision",
            "model-commit",
            "--dataset-revision",
            "dataset-commit",
            "--attention-implementation",
            "eager",
            "--adapter-rank",
            "4",
            "--adapter-layers",
            "2",
            "--no-frozen-prefix-scoring",
        ]
    )

    assert arguments.output == Path("result.json")
    assert arguments.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert arguments.max_tokens == 17
    assert arguments.directions == 8
    assert arguments.model_revision == "model-commit"
    assert arguments.dataset_revision == "dataset-commit"
    assert arguments.attention_implementation == "eager"
    assert arguments.adapter_rank == 4
    assert arguments.adapter_layers == 2
    assert arguments.use_frozen_prefix_scoring is False


def test_teacher_forced_call_counters_distinguish_cached_prefix_build() -> None:
    uncached = SimpleNamespace(environment_samples=16, frozen_prefix_cache=None)
    cached = SimpleNamespace(
        environment_samples=16,
        frozen_prefix_cache=SimpleNamespace(full_prefix_calls=1),
    )

    assert _teacher_forced_call_counters(
        uncached,  # type: ignore[arg-type]
        policy_evaluations=8,
        scoring_micro_batch_size=4,
        include_prefix_build=True,
    ) == {"forward_calls": 32, "full_prefix_calls": 32, "suffix_calls": 32}
    assert _teacher_forced_call_counters(
        cached,  # type: ignore[arg-type]
        policy_evaluations=8,
        scoring_micro_batch_size=4,
        include_prefix_build=True,
    ) == {"forward_calls": 33, "full_prefix_calls": 1, "suffix_calls": 32}


def test_runtime_and_prefix_metadata_record_resolved_configuration() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="model-commit",
            _attn_implementation="flash_attention_2",
        )
    )
    bundle = SimpleNamespace(model=model)
    runtime = _model_runtime_metadata(
        bundle,  # type: ignore[arg-type]
        requested_revision="model-commit",
        requested_attention_implementation="flash_attention_2",
    )
    assert runtime["revision_matches_requested"] is True
    assert runtime["attention_matches_requested"] is True
    assert "installed_flash_attn_version" in runtime

    structure = SimpleNamespace(
        total_layers=28,
        first_adapted_layer=24,
        suffix_layers=(object(), object(), object(), object()),
    )
    rollout = SimpleNamespace(
        frozen_prefix_cache=SimpleNamespace(
            structure=structure,
            batch_size=16,
            sequence_length=700,
            full_prefix_calls=1,
        ),
        frozen_prefix_fallback_reason=None,
    )
    behavior = {"forward_calls": 2, "full_prefix_calls": 1, "suffix_calls": 1}
    oracle = {"forward_calls": 1, "full_prefix_calls": 0, "suffix_calls": 1}
    probes = {"forward_calls": 64, "full_prefix_calls": 0, "suffix_calls": 64}
    prefix = _prefix_cache_metadata(
        rollout,  # type: ignore[arg-type]
        requested=True,
        behavior_counters=behavior,
        oracle_counters=oracle,
        finite_difference_counters=probes,
    )

    assert prefix["active"] is True
    assert prefix["frozen_decoder_layers"] == 24
    assert prefix["replayed_suffix_layers"] == 4
    assert prefix["cached_examples"] == 16
    assert prefix["total_teacher_forced_scoring"] == {
        "forward_calls": 67,
        "full_prefix_calls": 1,
        "suffix_calls": 66,
    }
