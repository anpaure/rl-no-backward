"""Real-model GSM8K diagnostics for projected forward-only sequence RL.

This command intentionally performs one reverse-mode calculation: it forms the
exact gradient of the fixed-rollout GRPO surrogate as an *external diagnostic
oracle*.  It never applies that gradient or updates the model.  Every finite-
difference quantity is obtained through the production inference-only sequence
path, and :func:`audit_sequence_forward_only_source` statically verifies that
the path contains no reverse-mode calls.

Run the diagnostic on a CUDA machine with, for example::

    python -m rl_no_backward.gsm8k_diagnostics \
        --output artifacts/gsm8k_diagnostics.json \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --max-tokens 64

Only the official GSM8K training split is loaded.  The calibration examples,
two rollout prompts, sampled completions, and random search subspace are all
selected with fixed seeds so the saved JSON is an auditable experiment
artifact rather than a synthetic unit check.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .common import make_search_basis
from .diagnostics import SourceAuditResult, audit_forward_only_source
from .gsm8k import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_TRAIN_SPLIT,
    DifficultyFilter,
    GSM8KExample,
    exact_match_reward,
    extract_model_answer,
    filter_by_difficulty,
    format_prompt,
    load_gsm8k_split,
    select_seeded_subset,
)
from .gsm8k_experiment import GSM8KExperimentConfig, shaped_gsm8k_reward
from .model import (
    ModelBundle,
    load_model_bundle,
    parameter_vector,
    set_adapter_grad_enabled,
)
from .sequence_forward_only import directional_sequence_score_statistics
from .sequence_policy import (
    CompletionSample,
    SequenceRolloutBatch,
    clipped_grpo_surrogate,
    generate_sequence_rollouts,
    teacher_forced_token_log_probs,
)
from .task import CANDIDATE_ACTIONS

FINITE_DIFFERENCE_MUS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
DIAGNOSTIC_SEED = 2026
CALIBRATION_EXAMPLES = 8
ROLLOUT_BATCH_SIZE = 2
ROLLOUT_GROUP_SIZE = 4
SEARCH_DIRECTIONS = 4
DEFAULT_MAX_TOKENS = 64


def _finite_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    numeric = value.detach().to(dtype=torch.float64)
    if not torch.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def vector_agreement(
    reference: Tensor,
    estimate: Tensor,
    *,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | bool | None]:
    """Return stable cosine and relative-L2 agreement for two vectors.

    ``None`` is used when a metric is mathematically undefined (for example,
    cosine similarity with a zero vector).  This keeps degenerate all-zero
    reward rollouts explicit rather than manufacturing a perfect score or
    emitting non-standard JSON ``NaN`` values.
    """

    if zero_tolerance < 0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be non-negative and finite")
    exact = _finite_tensor("reference", reference).reshape(-1)
    approximate = _finite_tensor("estimate", estimate).reshape(-1)
    if exact.shape != approximate.shape:
        raise ValueError("reference and estimate must have the same number of entries")

    exact_norm = float(torch.linalg.vector_norm(exact).item())
    estimate_norm = float(torch.linalg.vector_norm(approximate).item())
    error_norm = float(torch.linalg.vector_norm(approximate - exact).item())
    reference_nonzero = exact_norm > zero_tolerance
    estimate_nonzero = estimate_norm > zero_tolerance
    cosine: float | None = None
    if reference_nonzero and estimate_nonzero:
        raw_cosine = float(torch.dot(exact, approximate).item()) / (exact_norm * estimate_norm)
        cosine = max(-1.0, min(1.0, raw_cosine))
    relative_error = error_norm / exact_norm if reference_nonzero else None
    return {
        "reference_nonzero": reference_nonzero,
        "estimate_nonzero": estimate_nonzero,
        "reference_l2_norm": exact_norm,
        "estimate_l2_norm": estimate_norm,
        "absolute_l2_error": error_norm,
        "relative_l2_error": relative_error,
        "cosine_similarity": cosine,
    }


def random_subspace_capture(
    gradient: Tensor,
    basis: Tensor,
    *,
    orthonormal_tolerance: float = 2e-5,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | int | None]:
    """Measure how much squared gradient norm an orthonormal basis captures."""

    full = _finite_tensor("gradient", gradient).reshape(-1)
    subspace = _finite_tensor("basis", basis)
    if subspace.ndim != 2 or subspace.shape[0] != full.numel() or subspace.shape[1] == 0:
        raise ValueError("basis must have shape [gradient entries, directions >= 1]")
    if subspace.shape[1] > subspace.shape[0]:
        raise ValueError("basis cannot have more columns than rows")
    gram = subspace.T @ subspace
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    orthonormal_error = float((gram - identity).abs().max().item())
    if orthonormal_error > orthonormal_tolerance:
        raise ValueError(f"basis columns are not orthonormal (max error {orthonormal_error:.3g})")

    coordinates = subspace.T @ full
    full_squared_norm = float(torch.dot(full, full).item())
    projected_squared_norm = float(torch.dot(coordinates, coordinates).item())
    capture = (
        projected_squared_norm / full_squared_norm
        if full_squared_norm > zero_tolerance**2
        else None
    )
    # Roundoff can put a mathematically bounded ratio microscopically outside [0, 1].
    if capture is not None:
        capture = max(0.0, min(1.0, capture))
    return {
        "parameter_count": int(subspace.shape[0]),
        "directions": int(subspace.shape[1]),
        "full_gradient_l2_norm": math.sqrt(full_squared_norm),
        "projected_gradient_l2_norm": math.sqrt(max(projected_squared_norm, 0.0)),
        "captured_squared_norm_fraction": capture,
        "expected_isotropic_squared_norm_fraction": float(subspace.shape[1] / subspace.shape[0]),
        "basis_orthonormality_max_error": orthonormal_error,
    }


def fisher_trace_comparison(
    directional_token_scores: Tensor,
    response_mask: Tensor,
    *,
    legacy_length_normalize: bool = True,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | bool | None | str]:
    """Compare token-local Fisher trace with the old completion-outer trace.

    The sampled sequence KL averages token-local KL within each completion, so
    its local curvature is

    ``mean_bg(sum_t score_t score_t.T / response_length)``.

    The legacy approximation first sums (or mean-pools) token scores and then
    takes one outer product.  It therefore contains cross-token terms that are
    not part of the KL curvature.  Traces are sufficient to expose the scale
    distortion without serializing either Fisher matrix.
    """

    scores = _finite_tensor("directional_token_scores", directional_token_scores)
    if scores.ndim != 4:
        raise ValueError("directional_token_scores must have shape [B, G, T, D]")
    if not isinstance(response_mask, Tensor):
        raise TypeError("response_mask must be a torch.Tensor")
    if response_mask.dtype != torch.bool:
        raise TypeError("response_mask must have boolean dtype")
    if tuple(response_mask.shape) != tuple(scores.shape[:3]):
        raise ValueError("response_mask must match the first three score dimensions")
    if not bool(response_mask.any(dim=-1).all()):
        raise ValueError("every completion must contain at least one valid token")

    mask = response_mask.to(device=scores.device).unsqueeze(-1)
    masked_scores = scores.masked_fill(~mask, 0.0)
    lengths = response_mask.sum(dim=-1).to(device=scores.device, dtype=scores.dtype)
    token_squared_norms = masked_scores.square().sum(dim=-1)
    correct_trace = float((token_squared_norms.sum(dim=-1) / lengths).mean().item())

    completion_scores = masked_scores.sum(dim=2)
    normalization = "mean_token_score" if legacy_length_normalize else "summed_token_score"
    if legacy_length_normalize:
        completion_scores = completion_scores / lengths.unsqueeze(-1)
    legacy_trace = float(completion_scores.square().sum(dim=-1).mean().item())
    ratio = legacy_trace / correct_trace if correct_trace > zero_tolerance else None
    return {
        "correct_token_fisher_trace": correct_trace,
        "legacy_completion_outer_fisher_trace": legacy_trace,
        "legacy_to_correct_trace_ratio": ratio,
        "legacy_completion_score_normalization": normalization,
        "correct_trace_nonzero": correct_trace > zero_tolerance,
    }


def reward_group_statistics(
    rewards: Tensor,
    *,
    exact_rewards: Tensor | None = None,
    zero_tolerance: float = 1e-12,
) -> dict[str, float | int | None]:
    """Summarize shaped rewards and prompt groups with zero GRPO advantage."""

    shaped = _finite_tensor("rewards", rewards)
    if shaped.ndim != 2 or shaped.shape[0] == 0 or shaped.shape[1] < 2:
        raise ValueError("rewards must have shape [B >= 1, G >= 2]")
    within_group_span = shaped.amax(dim=1) - shaped.amin(dim=1)
    zero_groups = within_group_span <= zero_tolerance
    payload: dict[str, float | int | None] = {
        "prompt_groups": int(shaped.shape[0]),
        "group_size": int(shaped.shape[1]),
        "samples": int(shaped.numel()),
        "reward_mean": float(shaped.mean().item()),
        "reward_std_population": float(shaped.std(unbiased=False).item()),
        "reward_min": float(shaped.min().item()),
        "reward_max": float(shaped.max().item()),
        "zero_reward_fraction": float((shaped.abs() <= zero_tolerance).double().mean().item()),
        "nonzero_reward_fraction": float((shaped.abs() > zero_tolerance).double().mean().item()),
        "zero_advantage_groups": int(zero_groups.sum().item()),
        "zero_advantage_group_fraction": float(zero_groups.double().mean().item()),
    }
    if exact_rewards is None:
        payload["exact_reward_rate"] = None
    else:
        exact = _finite_tensor("exact_rewards", exact_rewards)
        if exact.shape != shaped.shape:
            raise ValueError("exact_rewards must have the same shape as rewards")
        if bool(((exact < 0) | (exact > 1)).any()):
            raise ValueError("exact_rewards must lie in [0, 1]")
        payload["exact_reward_rate"] = float(exact.mean().item())
    return payload


def audit_sequence_forward_only_source(
    source_path: str | Path | None = None,
) -> SourceAuditResult:
    """Audit the sequence forward-only implementation for reverse-mode calls."""

    path = (
        Path(source_path)
        if source_path is not None
        else Path(__file__).with_name("sequence_forward_only.py")
    )
    return audit_forward_only_source(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_gsm8k_diagnostic_results(
    destination: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Write a diagnostic report as deterministic, standards-compliant JSON."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _chat_prompts(tokenizer: object, examples: Sequence[GSM8KExample]) -> list[str]:
    def render(messages: Sequence[Mapping[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    return [format_prompt(example.question, render) for example in examples]


def _reward_callback(examples: Sequence[GSM8KExample], shaping_weight: float) -> Any:
    def reward(sample: CompletionSample) -> float:
        return shaped_gsm8k_reward(
            sample.completion,
            examples[sample.prompt_index],
            shaping_weight,
        )

    return reward


def _exact_backprop_grpo_gradient(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    *,
    clip_epsilon: float,
    scoring_micro_batch_size: int,
) -> tuple[Tensor, float, bool]:
    """Return the exact ascent gradient without applying a parameter update."""

    bundle.model.eval()
    parameters = bundle.trainable_parameters
    if not parameters:
        raise ValueError("the diagnostic requires at least one adapter parameter")
    before = parameter_vector(bundle).float().clone()
    set_adapter_grad_enabled(bundle, True)
    for parameter in parameters:
        parameter.grad = None
    try:
        new_token_log_probs = teacher_forced_token_log_probs(
            bundle,
            rollout,
            micro_batch_size=scoring_micro_batch_size,
        )
        surrogate = clipped_grpo_surrogate(new_token_log_probs, rollout, clip_epsilon)
        surrogate.backward()
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("backprop did not produce every adapter gradient")
        gradient = torch.cat(
            [parameter.grad.detach().float().reshape(-1) for parameter in parameters]
        )
        surrogate_value = float(surrogate.detach().item())
    finally:
        for parameter in parameters:
            parameter.grad = None
        set_adapter_grad_enabled(bundle, False)
    unchanged = bool(torch.equal(parameter_vector(bundle).float(), before))
    return gradient, surrogate_value, unchanged


def _rollout_rows(
    examples: Sequence[GSM8KExample], rollout: SequenceRolloutBatch
) -> tuple[list[dict[str, Any]], Tensor]:
    exact_groups: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    for prompt_index, example in enumerate(examples):
        completions: list[dict[str, Any]] = []
        exact_group: list[float] = []
        for group_index, completion in enumerate(rollout.completions[prompt_index]):
            exact = exact_match_reward(completion, example)
            exact_group.append(exact)
            completions.append(
                {
                    "group_index": group_index,
                    "text": completion,
                    "extracted_answer": extract_model_answer(completion),
                    "shaped_reward": float(rollout.rewards[prompt_index, group_index].item()),
                    "exact_reward": exact,
                    "response_tokens": int(
                        rollout.response_mask[prompt_index, group_index].sum().item()
                    ),
                }
            )
        exact_groups.append(exact_group)
        rows.append(
            {
                "example_id": example.example_id,
                "source_index": example.source_index,
                "question": example.question,
                "reference_answer": example.canonical_answer,
                "completions": completions,
            }
        )
    return rows, torch.tensor(exact_groups, dtype=torch.float64)


def _load_fixed_train_examples(
    config: GSM8KExperimentConfig,
) -> tuple[tuple[GSM8KExample, ...], tuple[GSM8KExample, ...], int]:
    difficulty = DifficultyFilter(
        min_reasoning_lines=config.min_reasoning_lines,
        max_reasoning_lines=config.max_reasoning_lines,
        max_answer_magnitude=config.max_answer_magnitude,
    )
    official_train = load_gsm8k_split(GSM8K_TRAIN_SPLIT)
    filtered_train = filter_by_difficulty(official_train, difficulty)
    needed = CALIBRATION_EXAMPLES + ROLLOUT_BATCH_SIZE
    selected = select_seeded_subset(
        filtered_train,
        needed,
        seed=DIAGNOSTIC_SEED,
        namespace="real-model-fd-diagnostic",
    )
    return (
        selected[:CALIBRATION_EXAMPLES],
        selected[CALIBRATION_EXAMPLES:],
        len(official_train),
    )


def run_gsm8k_diagnostics(
    *,
    model_name: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Run the fixed real-Qwen GSM8K finite-difference diagnostic on CUDA."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    defaults = GSM8KExperimentConfig()
    selected_model = defaults.model_name if model_name is None else model_name
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise ValueError("model_name must be a non-empty string")
    if not torch.cuda.is_available():
        raise RuntimeError("the real-model diagnostic requires a CUDA device")

    started = time.perf_counter()
    source_audit = audit_sequence_forward_only_source()
    calibration_examples, rollout_examples, official_train_size = _load_fixed_train_examples(
        defaults
    )

    # Match the Qwen chat formatting used by the benchmark before calibration.
    from transformers import AutoTokenizer

    calibration_tokenizer = AutoTokenizer.from_pretrained(selected_model)
    if calibration_tokenizer.pad_token_id is None:
        calibration_tokenizer.pad_token = calibration_tokenizer.eos_token
    calibration_tokenizer.padding_side = "left"
    calibration_prompts = _chat_prompts(calibration_tokenizer, calibration_examples)

    bundle = load_model_bundle(
        model_name=selected_model,
        calibration_prompts=calibration_prompts,
        candidates=CANDIDATE_ACTIONS[:4],
        adapter_rank=defaults.adapter_rank,
        adapter_layers=defaults.adapter_layers,
        adapter_scale=defaults.adapter_scale,
        dtype=defaults.dtype,
        device=defaults.device,
    )
    prompts = _chat_prompts(bundle.tokenizer, rollout_examples)
    rollout = generate_sequence_rollouts(
        bundle,
        prompts,
        _reward_callback(rollout_examples, defaults.numeric_shaping_weight),
        group_size=ROLLOUT_GROUP_SIZE,
        max_new_tokens=max_tokens,
        temperature=defaults.sampling_temperature,
        max_prompt_tokens=defaults.max_prompt_tokens,
        seed=DIAGNOSTIC_SEED,
        scoring_micro_batch_size=defaults.scoring_micro_batch_size,
    )
    rollout_rows, exact_rewards = _rollout_rows(rollout_examples, rollout)
    reward_statistics = reward_group_statistics(
        rollout.rewards.detach().cpu(),
        exact_rewards=exact_rewards,
    )

    center = parameter_vector(bundle).float().clone()
    exact_gradient, surrogate_value, parameters_unchanged = _exact_backprop_grpo_gradient(
        bundle,
        rollout,
        clip_epsilon=0.2,
        scoring_micro_batch_size=defaults.scoring_micro_batch_size,
    )
    generator = torch.Generator(device=bundle.device).manual_seed(DIAGNOSTIC_SEED + 1)
    basis = make_search_basis(
        center.numel(),
        SEARCH_DIRECTIONS,
        generator,
        bundle.device,
    )
    exact_projected_coordinates = basis.T @ exact_gradient
    capture = random_subspace_capture(exact_gradient, basis)

    finite_difference_results: list[dict[str, Any]] = []
    for mu in FINITE_DIFFERENCE_MUS:
        statistics, policy_evaluations = directional_sequence_score_statistics(
            bundle,
            rollout,
            center,
            basis,
            mu,
            length_normalize=True,
            scoring_micro_batch_size=defaults.scoring_micro_batch_size,
        )
        reconstructed_gradient = basis @ statistics.gradient
        finite_difference_results.append(
            {
                "mu": mu,
                "policy_evaluations": policy_evaluations,
                "teacher_forced_examples": policy_evaluations * rollout.environment_samples,
                "scored_tokens": policy_evaluations * rollout.valid_response_tokens,
                "projected_gradient_coordinates": statistics.gradient,
                "agreement_with_exact_projected_gradient": vector_agreement(
                    exact_projected_coordinates,
                    statistics.gradient,
                ),
                "agreement_with_full_backprop_gradient": vector_agreement(
                    exact_gradient,
                    reconstructed_gradient,
                ),
                "fisher": fisher_trace_comparison(
                    statistics.directional_token_scores,
                    rollout.response_mask,
                    legacy_length_normalize=True,
                ),
            }
        )

    eligible = [
        result
        for result in finite_difference_results
        if result["agreement_with_exact_projected_gradient"]["relative_l2_error"] is not None
    ]
    best_mu = (
        min(
            eligible,
            key=lambda result: result["agreement_with_exact_projected_gradient"][
                "relative_l2_error"
            ],
        )["mu"]
        if eligible
        else None
    )
    final_parameters_unchanged = bool(torch.equal(parameter_vector(bundle).float(), center))
    elapsed = time.perf_counter() - started
    device_index = bundle.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "real_model_gsm8k_projected_finite_difference",
        "purpose": (
            "Backprop is used once as a non-updating oracle; the compared production "
            "optimizer path remains forward-only."
        ),
        "model": {
            "name": bundle.model_name,
            "dtype": defaults.dtype,
            "device": str(bundle.device),
            "adapter_rank": defaults.adapter_rank,
            "adapter_layers": defaults.adapter_layers,
            "adapter_parameter_count": bundle.parameter_count,
            "adapter_names": bundle.adapter_names,
        },
        "dataset": {
            "id": GSM8K_DATASET_ID,
            "config": GSM8K_DATASET_CONFIG,
            "split": GSM8K_TRAIN_SPLIT,
            "official_train_only": True,
            "official_train_examples": official_train_size,
            "selection_seed": DIAGNOSTIC_SEED,
            "calibration_example_ids": [example.example_id for example in calibration_examples],
            "rollout_example_ids": [example.example_id for example in rollout_examples],
        },
        "rollout": {
            "batch_size": rollout.batch_size,
            "group_size": rollout.group_size,
            "environment_samples": rollout.environment_samples,
            "max_new_tokens": max_tokens,
            "valid_response_tokens": rollout.valid_response_tokens,
            "sampling_temperature": rollout.sampling_temperature,
            "numeric_shaping_weight": defaults.numeric_shaping_weight,
            "reward_statistics": reward_statistics,
            "samples": rollout_rows,
        },
        "exact_backprop_oracle": {
            "objective": "length-normalized token-clipped GRPO surrogate at policy center",
            "clip_epsilon": 0.2,
            "surrogate_value": surrogate_value,
            "gradient_l2_norm": float(exact_gradient.norm().item()),
            "projected_gradient_coordinates": exact_projected_coordinates,
            "backward_calls": 1,
            "optimizer_steps": 0,
            "parameters_unchanged_after_backward": parameters_unchanged,
        },
        "random_subspace": capture,
        "finite_difference": {
            "mus": FINITE_DIFFERENCE_MUS,
            "directions": SEARCH_DIRECTIONS,
            "best_mu_by_projected_relative_l2_error": best_mu,
            "results": finite_difference_results,
        },
        "source_audit": asdict(source_audit),
        "integrity": {
            "parameters_unchanged_after_all_probes": final_parameters_unchanged,
            "source_audit_passed": source_audit.passed,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device_index),
            "hostname": platform.node(),
        },
        "elapsed_seconds": elapsed,
    }
    report["passed"] = bool(
        source_audit.passed and parameters_unchanged and final_parameters_unchanged
    )
    return _json_safe(report)


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone real-model diagnostic argument parser."""

    defaults = GSM8KExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact path")
    parser.add_argument(
        "--model",
        default=defaults.model_name,
        help=f"Hugging Face Qwen model (default: {defaults.model_name})",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_integer,
        default=DEFAULT_MAX_TOKENS,
        help=f"maximum generated response tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gsm8k_diagnostics(
        model_name=arguments.model,
        max_tokens=arguments.max_tokens,
    )
    output = write_gsm8k_diagnostic_results(arguments.output, report)
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised on the remote GPU
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_EXAMPLES",
    "DEFAULT_MAX_TOKENS",
    "DIAGNOSTIC_SEED",
    "FINITE_DIFFERENCE_MUS",
    "ROLLOUT_BATCH_SIZE",
    "ROLLOUT_GROUP_SIZE",
    "SEARCH_DIRECTIONS",
    "audit_sequence_forward_only_source",
    "build_parser",
    "fisher_trace_comparison",
    "main",
    "random_subspace_capture",
    "reward_group_statistics",
    "run_gsm8k_diagnostics",
    "vector_agreement",
    "write_gsm8k_diagnostic_results",
]
