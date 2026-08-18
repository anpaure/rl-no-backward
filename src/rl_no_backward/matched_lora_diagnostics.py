"""Isolated finite-difference diagnostic for the matched standard-LoRA policy.

This module is a measurement tool, not a trainer.  It deliberately computes an
exact reverse-mode gradient, then compares its coordinates in the production
forward-only search basis with the center-policy token-score estimator used by
the trainer.  The policy is restored byte-for-byte before returning.

The forward-only trainer never imports this module.  Run it as its own process::

    python -m rl_no_backward.matched_lora_diagnostics --config CONFIG --output REPORT

The first real-model invocation can cache the complete fixed rollout.  Later
invocations consume that cache without generating new candidates, which makes
comparisons across finite-difference radii exact and cheap to reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .common import make_search_basis
from .matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
)
from .matched_lora_backprop import matched_grpo_streaming_backward
from .model import ModelBundle, parameter_vector, set_adapter_grad_enabled, set_parameter_vector
from .sequence_policy import (
    SequenceRolloutBatch,
    central_difference_score_statistics,
    teacher_forced_token_log_probs,
)
from .standard_lora import (
    StandardLoRAConfig,
    frozen_base_parameter_digest,
    lora_parameter_layout,
    lora_state_digest,
)

REAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
REAL_LORA_PARAMETER_COUNT = 1_089_536
ROLLOUT_CACHE_SCHEMA = "rl-no-backward-matched-fixed-rollout-v1"
DIAGNOSTIC_SCHEMA = "rl-no-backward-matched-lora-fd-diagnostic-v1"


@dataclass(frozen=True, slots=True)
class GradientAgreementThresholds:
    """Numerical gates applied independently to every finite-difference radius."""

    minimum_cosine_similarity: float = 0.95
    maximum_relative_l2_error: float = 0.25
    maximum_absolute_error: float | None = None
    minimum_passing_mu_count: int = 1
    required_mu: float | None = None
    basis_orthonormality_tolerance: float = 2.0e-5

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_cosine_similarity <= 1.0:
            raise ValueError("minimum_cosine_similarity must lie in [-1, 1]")
        if not math.isfinite(self.maximum_relative_l2_error) or self.maximum_relative_l2_error <= 0:
            raise ValueError("maximum_relative_l2_error must be positive and finite")
        if self.maximum_absolute_error is not None and (
            not math.isfinite(self.maximum_absolute_error) or self.maximum_absolute_error <= 0
        ):
            raise ValueError("maximum_absolute_error must be positive and finite or None")
        if (
            isinstance(self.minimum_passing_mu_count, bool)
            or not isinstance(self.minimum_passing_mu_count, int)
            or self.minimum_passing_mu_count < 1
        ):
            raise ValueError("minimum_passing_mu_count must be a positive integer")
        if self.required_mu is not None and (
            not math.isfinite(self.required_mu) or self.required_mu <= 0
        ):
            raise ValueError("required_mu must be positive and finite or None")
        if (
            not math.isfinite(self.basis_orthonormality_tolerance)
            or self.basis_orthonormality_tolerance <= 0
        ):
            raise ValueError("basis_orthonormality_tolerance must be positive and finite")


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _semantic_digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _git_output(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _source_receipt(
    *,
    expected_commit: str | None,
    allow_dirty: bool,
) -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    if commit is None or status is None:
        raise RuntimeError("diagnostic requires a Git worktree with a resolvable source commit")
    dirty_entries = status.splitlines() if status else []
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"diagnostic source commit {commit} differs from expected {expected_commit}"
        )
    if dirty_entries and not allow_dirty:
        raise RuntimeError("diagnostic requires a clean source worktree")
    return {
        "commit": commit,
        "expected_commit": expected_commit,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_receipt(device: torch.device) -> dict[str, Any]:
    packages = {
        distribution: _installed_version(distribution)
        for distribution in (
            "accelerate",
            "datasets",
            "flash-attn",
            "peft",
            "torch",
            "transformers",
            "trl",
            "vllm",
        )
    }
    hardware: dict[str, Any] = {
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "resolved_device": f"cuda:{index}",
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": [properties.major, properties.minor],
            }
        )
    return {"packages": packages, "hardware": hardware}


def _finite_statistics(values: Tensor) -> dict[str, int | float | bool | None]:
    numeric = values.detach().float().reshape(-1)
    finite_mask = torch.isfinite(numeric)
    finite = numeric[finite_mask]
    count = int(numeric.numel())
    finite_count = int(finite.numel())
    if finite_count:
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        mean = float(finite.mean().item())
        standard_deviation = float(finite.std(unbiased=False).item())
        l2_norm = float(finite.norm().item())
        max_abs = float(finite.abs().max().item())
    else:
        minimum = maximum = mean = standard_deviation = l2_norm = max_abs = None
    return {
        "count": count,
        "finite_count": finite_count,
        "non_finite_count": count - finite_count,
        "finite_fraction": finite_count / count if count else 0.0,
        "all_finite": finite_count == count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "l2_norm": l2_norm,
        "max_abs": max_abs,
    }


def _coordinate_agreement(
    exact: Tensor,
    finite_difference: Tensor,
    thresholds: GradientAgreementThresholds,
) -> dict[str, Any]:
    if exact.shape != finite_difference.shape or exact.ndim != 1:
        raise ValueError("gradient coordinate vectors must have the same rank-one shape")
    error = finite_difference.float() - exact.float()
    exact_norm = float(exact.float().norm().item())
    finite_norm = float(finite_difference.float().norm().item())
    all_finite = bool(
        torch.isfinite(exact).all().item()
        and torch.isfinite(finite_difference).all().item()
        and torch.isfinite(error).all().item()
    )
    nonzero = exact_norm > 1.0e-20 and finite_norm > 1.0e-20
    if all_finite and nonzero:
        cosine: float | None = float(
            torch.nn.functional.cosine_similarity(
                exact.float(),
                finite_difference.float(),
                dim=0,
            ).item()
        )
        relative_l2: float | None = float(error.norm().item() / exact_norm)
        maximum_absolute_error: float | None = float(error.abs().max().item())
    else:
        cosine = relative_l2 = maximum_absolute_error = None
    passed = bool(
        all_finite
        and nonzero
        and cosine is not None
        and cosine >= thresholds.minimum_cosine_similarity
        and relative_l2 is not None
        and relative_l2 <= thresholds.maximum_relative_l2_error
        and (
            thresholds.maximum_absolute_error is None
            or (
                maximum_absolute_error is not None
                and maximum_absolute_error <= thresholds.maximum_absolute_error
            )
        )
    )
    return {
        "passed": passed,
        "cosine_similarity": cosine,
        "relative_l2_error": relative_l2,
        "maximum_absolute_error": maximum_absolute_error,
        "exact_projected_norm": exact_norm,
        "finite_difference_norm": finite_norm,
        "exact_coordinates": [float(value) for value in exact.detach().float().cpu()],
        "finite_difference_coordinates": [
            float(value) for value in finite_difference.detach().float().cpu()
        ],
        "exact_coordinate_statistics": _finite_statistics(exact),
        "finite_difference_coordinate_statistics": _finite_statistics(finite_difference),
        "error_statistics": _finite_statistics(error),
    }


def _validate_mu_values(mu_values: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in mu_values)
    if not values:
        raise ValueError("at least one finite-difference radius is required")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("finite-difference radii must be positive and finite")
    if len(set(values)) != len(values):
        raise ValueError("finite-difference radii must be unique")
    return values


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_matches_bundle(parameter_device: torch.device, bundle_device: torch.device) -> bool:
    """Treat an unindexed bundle device as the configured device type's current device."""

    if parameter_device.type != bundle_device.type:
        return False
    return bundle_device.index is None or parameter_device.index == bundle_device.index


def _rollout_receipts(
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
) -> dict[str, Any]:
    return {
        "batch_size": rollout.batch_size,
        "group_size": rollout.group_size,
        "environment_samples": rollout.environment_samples,
        "valid_response_tokens": rollout.valid_response_tokens,
        "prompt_input_ids_digest": _tensor_digest(rollout.prompt_input_ids),
        "response_input_ids_digest": _tensor_digest(rollout.response_input_ids),
        "response_mask_digest": _tensor_digest(rollout.response_mask),
        "hf_old_token_log_probs_digest": _tensor_digest(rollout.old_token_log_probs),
        "sampler_token_log_probs_digest": _tensor_digest(sampler_token_log_probs),
        "rewards_digest": _tensor_digest(rollout.rewards),
        "advantages_digest": _tensor_digest(rollout.advantages),
    }


def fixed_rollout_finite_difference_diagnostic(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    objective_config: MatchedGRPOObjectiveConfig,
    *,
    directions: int,
    basis_seed: int,
    mu_values: Sequence[float] = (0.5, 1.0, 2.0),
    scoring_micro_batch_size: int = 16,
    prompt_groups_per_micro_batch: int = 1,
    thresholds: GradientAgreementThresholds | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare exact and finite-difference coordinates without updating the policy.

    ``basis_seed`` must be the seed of the production FO direction generator at
    the diagnosed step.  For the first matched-run step this is ``seed + 70_000``.
    One basis is constructed and reused for every radius, so radius is the only
    changing numerical variable.
    """

    radii = _validate_mu_values(mu_values)
    gates = thresholds or GradientAgreementThresholds()
    if gates.minimum_passing_mu_count > len(radii):
        raise ValueError("minimum_passing_mu_count exceeds the number of radii")
    if gates.required_mu is not None and gates.required_mu not in radii:
        raise ValueError("required_mu must be present in the diagnosed radii")
    for name, value in (
        ("directions", directions),
        ("scoring_micro_batch_size", scoring_micro_batch_size),
        ("prompt_groups_per_micro_batch", prompt_groups_per_micro_batch),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(basis_seed, bool) or not isinstance(basis_seed, int):
        raise TypeError("basis_seed must be an integer")
    if rollout.frozen_prefix_cache is not None:
        raise ValueError("the all-layer LoRA diagnostic requires full-prefix scoring")
    if sampler_token_log_probs.shape != rollout.old_token_log_probs.shape:
        raise ValueError("sampler log-probability shape mismatch")
    if sampler_token_log_probs.requires_grad:
        raise ValueError("sampler log probabilities must be detached")
    if not all(
        _device_matches_bundle(parameter.device, bundle.device)
        for parameter in bundle.trainable_parameters
    ):
        raise ValueError("all adapter tensors must be on the bundle device")

    center = parameter_vector(bundle).detach().float().clone()
    if directions > center.numel():
        raise ValueError("directions exceed the adapter parameter count")
    adapter_digest_before = lora_state_digest(bundle.model)
    base_digest_before = frozen_base_parameter_digest(bundle.model)
    original_requires_grad = tuple(
        parameter.requires_grad for parameter in bundle.trainable_parameters
    )
    original_gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in bundle.trainable_parameters
    )
    original_training = bundle.model.training
    generator = torch.Generator(device=bundle.device).manual_seed(basis_seed)
    basis = make_search_basis(
        center.numel(),
        directions,
        generator,
        bundle.device,
    )
    gram_error = basis.T @ basis - torch.eye(
        directions,
        dtype=basis.dtype,
        device=basis.device,
    )
    basis_max_orthonormality_error = float(gram_error.abs().max().item())

    exact_gradient: Tensor | None = None
    exact_objective: float | None = None
    missing_gradient_parameters: list[str] = []
    comparisons: list[dict[str, Any]] = []
    exact_seconds = 0.0
    try:
        bundle.model.eval()
        set_adapter_grad_enabled(bundle, True)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        _synchronize(bundle.device)
        started = time.perf_counter()
        exact_objective, backward_calls = matched_grpo_streaming_backward(
            bundle,
            rollout,
            sampler_token_log_probs,
            objective_config,
            prompt_groups_per_micro_batch=prompt_groups_per_micro_batch,
        )
        _synchronize(bundle.device)
        exact_seconds = time.perf_counter() - started
        gradient_chunks: list[Tensor] = []
        for name, parameter in zip(
            bundle.adapter_names,
            bundle.trainable_parameters,
            strict=True,
        ):
            if parameter.grad is None:
                missing_gradient_parameters.append(name)
                gradient_chunks.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
            else:
                # The streaming BP routine minimizes -surrogate.
                gradient_chunks.append(-parameter.grad.detach().float().reshape(-1))
        exact_gradient = torch.cat(gradient_chunks)
        for parameter in bundle.trainable_parameters:
            parameter.grad = None
        set_adapter_grad_enabled(bundle, False)
        exact_coordinates = basis.T @ exact_gradient

        inference_weights = detached_inference_correction(
            rollout.old_token_log_probs,
            sampler_token_log_probs,
            rollout.response_mask,
            objective_config,
        )
        with torch.inference_mode():
            for mu in radii:
                positive_logps: list[Tensor] = []
                negative_logps: list[Tensor] = []
                _synchronize(bundle.device)
                started = time.perf_counter()
                for direction_index in range(directions):
                    direction = basis[:, direction_index]
                    set_parameter_vector(bundle, center + mu * direction)
                    positive = teacher_forced_token_log_probs(
                        bundle,
                        rollout,
                        micro_batch_size=scoring_micro_batch_size,
                    )
                    positive_logps.append(positive)

                    set_parameter_vector(bundle, center - mu * direction)
                    negative = teacher_forced_token_log_probs(
                        bundle,
                        rollout,
                        micro_batch_size=scoring_micro_batch_size,
                    )
                    negative_logps.append(negative)
                _synchronize(bundle.device)
                elapsed = time.perf_counter() - started
                finite = central_difference_score_statistics(
                    torch.stack(positive_logps, dim=-1),
                    torch.stack(negative_logps, dim=-1),
                    rollout,
                    mu,
                    length_normalize=True,
                    sampling_weights=inference_weights,
                ).gradient.float()
                agreement = _coordinate_agreement(exact_coordinates, finite, gates)
                comparisons.append(
                    {
                        "mu": mu,
                        "seconds": elapsed,
                        "coordinate_estimator": "center_policy_token_score_statistics",
                        **agreement,
                    }
                )
    finally:
        set_parameter_vector(bundle, center)
        for parameter, requires_grad, gradient in zip(
            bundle.trainable_parameters,
            original_requires_grad,
            original_gradients,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
            parameter.grad = None if gradient is None else gradient
        bundle.model.train(original_training)

    if exact_gradient is None or exact_objective is None:
        raise RuntimeError("the exact-gradient diagnostic did not complete")
    adapter_digest_after = lora_state_digest(bundle.model)
    base_digest_after = frozen_base_parameter_digest(bundle.model)
    passing_mu_count = sum(bool(comparison["passed"]) for comparison in comparisons)
    required_mu_comparison = next(
        (
            comparison
            for comparison in comparisons
            if gates.required_mu is not None and comparison["mu"] == gates.required_mu
        ),
        None,
    )
    required_mu_passed = bool(
        gates.required_mu is None
        or (required_mu_comparison is not None and required_mu_comparison["passed"])
    )
    exact_stats = _finite_statistics(exact_gradient)
    basis_passed = (
        math.isfinite(basis_max_orthonormality_error)
        and basis_max_orthonormality_error <= gates.basis_orthonormality_tolerance
    )
    integrity = {
        "adapter_state_digest_before": adapter_digest_before,
        "adapter_state_digest_after": adapter_digest_after,
        "adapter_state_restored": adapter_digest_before == adapter_digest_after,
        "frozen_base_parameter_digest_before": base_digest_before,
        "frozen_base_parameter_digest_after": base_digest_after,
        "frozen_base_unchanged": base_digest_before == base_digest_after,
        "requires_grad_flags_before": list(original_requires_grad),
        "requires_grad_flags_restored": tuple(
            parameter.requires_grad for parameter in bundle.trainable_parameters
        )
        == original_requires_grad,
        "gradient_buffers_restored": all(
            (before is None and parameter.grad is None)
            or (
                before is not None
                and parameter.grad is not None
                and torch.equal(before, parameter.grad)
            )
            for before, parameter in zip(
                original_gradients,
                bundle.trainable_parameters,
                strict=True,
            )
        ),
    }
    finite_exact = bool(exact_stats["all_finite"])
    passed = bool(
        passing_mu_count >= gates.minimum_passing_mu_count
        and required_mu_passed
        and basis_passed
        and finite_exact
        and not missing_gradient_parameters
        and integrity["adapter_state_restored"]
        and integrity["frozen_base_unchanged"]
        and integrity["requires_grad_flags_restored"]
        and integrity["gradient_buffers_restored"]
    )
    viable = [
        comparison for comparison in comparisons if comparison["relative_l2_error"] is not None
    ]
    recommended = (
        min(viable, key=lambda value: float(value["relative_l2_error"]))["mu"] if viable else None
    )
    layout: dict[str, Any]
    try:
        layout = lora_parameter_layout(bundle.model).as_dict()
    except (TypeError, ValueError):
        layout = {
            "names": list(bundle.adapter_names),
            "parameter_count": center.numel(),
        }
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "passed": passed,
        "process_id": os.getpid(),
        "reverse_mode_scope": "isolated_non_updating_diagnostic_process_only",
        "context": dict(context or {}),
        "objective": asdict(objective_config),
        "parameter_layout": layout,
        "directions": directions,
        "basis_seed": basis_seed,
        "basis_digest": _tensor_digest(basis),
        "basis_max_orthonormality_error": basis_max_orthonormality_error,
        "basis_passed": basis_passed,
        "mu_values": list(radii),
        "thresholds": asdict(gates),
        "passing_mu_count": passing_mu_count,
        "required_mu_passed": required_mu_passed,
        "recommended_mu_by_relative_l2": recommended,
        "exact_objective_at_center": exact_objective,
        "exact_gradient_full_l2_norm": float(exact_gradient.norm().item()),
        "exact_gradient_statistics": exact_stats,
        "exact_gradient_missing_parameter_count": len(missing_gradient_parameters),
        "exact_gradient_missing_parameters": missing_gradient_parameters,
        "exact_gradient_seconds": exact_seconds,
        "exact_backward_calls": backward_calls,
        "finite_difference_policy_evaluations": 2 * directions * len(radii),
        "comparisons": comparisons,
        "rollout": _rollout_receipts(rollout, sampler_token_log_probs),
        "integrity": integrity,
    }


def save_fixed_rollout_cache(
    path: str | Path,
    rollout: SequenceRolloutBatch,
    sampler_token_log_probs: Tensor,
    metadata: Mapping[str, Any],
) -> Path:
    """Save a safe tensor/primitive replay artifact for one immutable rollout."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite fixed-rollout cache {destination}")
    if rollout.frozen_prefix_cache is not None:
        raise ValueError("frozen-prefix caches are not portable rollout state")
    if sampler_token_log_probs.shape != rollout.old_token_log_probs.shape:
        raise ValueError("sampler log-probability shape mismatch")
    receipts = _rollout_receipts(rollout, sampler_token_log_probs)
    cache_metadata = dict(metadata)
    semantic_receipt = {
        "metadata": cache_metadata,
        "receipts": receipts,
    }
    payload = {
        "schema": ROLLOUT_CACHE_SCHEMA,
        "metadata": cache_metadata,
        "semantic_digest": _semantic_digest(semantic_receipt),
        "receipts": receipts,
        "prompts": rollout.prompts,
        "completions": rollout.completions,
        "prompt_input_ids": rollout.prompt_input_ids.detach().cpu(),
        "prompt_attention_mask": rollout.prompt_attention_mask.detach().cpu(),
        "response_input_ids": rollout.response_input_ids.detach().cpu(),
        "response_mask": rollout.response_mask.detach().cpu(),
        "old_token_log_probs": rollout.old_token_log_probs.detach().float().cpu(),
        "sampler_token_log_probs": sampler_token_log_probs.detach().float().cpu(),
        "rewards": rollout.rewards.detach().float().cpu(),
        "advantages": rollout.advantages.detach().float().cpu(),
        "sampling_temperature": float(rollout.sampling_temperature),
        "pad_token_id": int(rollout.pad_token_id),
        "eos_token_ids": rollout.eos_token_ids,
        "frozen_prefix_fallback_reason": rollout.frozen_prefix_fallback_reason,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_fixed_rollout_cache(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[SequenceRolloutBatch, Tensor, dict[str, Any]]:
    """Load and semantically validate a cache written by :func:`save_fixed_rollout_cache`."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != ROLLOUT_CACHE_SCHEMA:
        raise ValueError("unsupported fixed-rollout cache")
    metadata = payload.get("metadata")
    receipts = payload.get("receipts")
    if not isinstance(metadata, Mapping) or not isinstance(receipts, Mapping):
        raise TypeError("fixed-rollout cache metadata/receipts are malformed")
    expected_semantic_digest = _semantic_digest(
        {"metadata": dict(metadata), "receipts": dict(receipts)}
    )
    if payload.get("semantic_digest") != expected_semantic_digest:
        raise ValueError("fixed-rollout cache semantic digest is invalid")
    target = torch.device(device)
    rollout = SequenceRolloutBatch(
        prompts=tuple(payload["prompts"]),
        completions=tuple(tuple(group) for group in payload["completions"]),
        prompt_input_ids=payload["prompt_input_ids"].to(target),
        prompt_attention_mask=payload["prompt_attention_mask"].to(target),
        response_input_ids=payload["response_input_ids"].to(target),
        response_mask=payload["response_mask"].to(target),
        old_token_log_probs=payload["old_token_log_probs"].to(target).detach(),
        rewards=payload["rewards"].to(target),
        advantages=payload["advantages"].to(target),
        sampling_temperature=float(payload["sampling_temperature"]),
        pad_token_id=int(payload["pad_token_id"]),
        eos_token_ids=tuple(int(value) for value in payload["eos_token_ids"]),
        frozen_prefix_cache=None,
        frozen_prefix_fallback_reason=payload.get("frozen_prefix_fallback_reason"),
    )
    sampler = payload["sampler_token_log_probs"].to(target).detach()
    actual_receipts = _rollout_receipts(rollout, sampler)
    if actual_receipts != dict(receipts):
        raise ValueError("fixed-rollout tensors do not match their receipts")
    return rollout, sampler, dict(metadata)


def _validate_real_parameterization(config: Any, bundle: ModelBundle) -> None:
    expected_lora = StandardLoRAConfig()
    if config.model_name != REAL_MODEL_NAME:
        raise ValueError(f"diagnostic is locked to {REAL_MODEL_NAME}")
    if config.lora != expected_lora:
        raise ValueError("diagnostic requires all-28-block q_proj/v_proj PEFT LoRA r8/a16")
    layout = lora_parameter_layout(bundle.model)
    if layout.parameter_count != REAL_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"diagnostic found {layout.parameter_count:,} LoRA parameters; "
            f"expected {REAL_LORA_PARAMETER_COUNT:,}"
        )
    if tuple(bundle.adapter_names) != layout.names:
        raise RuntimeError("bundle adapter order differs from the canonical LoRA layout")


def _real_fixed_rollout(
    config: Any,
    bundle: ModelBundle,
    *,
    seed: int,
    step: int,
    cache_path: Path | None,
    scratch_root: Path,
) -> tuple[SequenceRolloutBatch, Tensor, dict[str, Any]]:
    from .evaluation_manifest import load_evaluation_split_manifest

    adapter_digest = lora_state_digest(bundle.model)
    objective_payload = asdict(config.objective)
    manifest_digest = load_evaluation_split_manifest(config.evaluation_manifest).manifest_sha256
    if cache_path is not None and cache_path.is_file():
        rollout, sampler, metadata = load_fixed_rollout_cache(
            cache_path,
            device=bundle.device,
        )
        expected = {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "dataset_revision": config.dataset_revision,
            "evaluation_manifest_sha256": manifest_digest,
            "adapter_state_digest": adapter_digest,
            "objective": objective_payload,
            "seed": seed,
            "step": step,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"fixed-rollout cache does not match this diagnostic: {mismatches}")
        return rollout, sampler, {**metadata, "source": "cached_fixed_rollout"}

    # These imports are intentionally local.  Importing this diagnostic never
    # pulls vLLM or the matched runner into the production FO process.
    from .matched_lora_runner import (
        _load_data,
        build_matched_rollout,
        build_prompt_schedule,
    )
    from .vllm_lora_rollout import (
        ReloadableLoRAGenerator,
        create_standard_lora_vllm_engine,
    )

    train_examples, _, manifest = _load_data(config)
    train_by_id = {example.example_id: example for example in train_examples}
    schedule = build_prompt_schedule(train_examples, config, seed=seed)
    if not 1 <= step <= len(schedule):
        raise ValueError(f"step must lie in [1, {len(schedule)}]")
    schedule_entry = schedule[step - 1]
    examples = tuple(train_by_id[example_id] for example_id in schedule_entry["example_ids"])
    engine = create_standard_lora_vllm_engine(
        config.model_name,
        revision=config.model_revision,
        dtype=config.dtype,
        max_model_len=config.max_prompt_tokens + config.max_new_tokens,
        max_lora_rank=config.lora.rank,
        kv_cache_memory_bytes=config.vllm_kv_cache_memory_bytes,
        enforce_eager=config.vllm_enforce_eager,
        flash_attn_version=config.vllm_flash_attn_version,
        max_num_seqs=config.responses_per_step,
        seed=seed,
    )
    generator = ReloadableLoRAGenerator(engine)
    reload_receipt = generator.sync(
        bundle.model,
        scratch_root / "vllm_lora_exports",
        version=1,
        policy_version=f"fd-diagnostic/seed={seed}/step={step}",
    )
    matched = build_matched_rollout(
        bundle,
        generator,
        examples,
        config,
        rollout_seed=int(schedule_entry["rollout_seed"]),
    )
    metadata = {
        "source": "generated_once_with_matched_vllm_hf_rollout_path",
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "dataset_revision": config.dataset_revision,
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "adapter_state_digest": adapter_digest,
        "objective": objective_payload,
        "seed": seed,
        "step": step,
        "rollout_seed": int(schedule_entry["rollout_seed"]),
        "prompt_example_ids": list(schedule_entry["example_ids"]),
        "runner_rollout_token_digest": matched.provenance["rollout_token_digest"],
        "runner_sampler_logprob_digest": matched.provenance["behavior_logprob_digest"],
        "runner_hf_old_logprob_digest": matched.provenance["hf_old_logprob_digest"],
        "vllm_hf_parity": matched.parity,
        "vllm_reload_state_digest": reload_receipt.state_digest,
        "vllm_reload_adapter_model_sha256": reload_receipt.adapter_model_sha256,
    }
    if cache_path is not None:
        save_fixed_rollout_cache(
            cache_path,
            matched.rollout,
            matched.sampler_token_log_probs,
            metadata,
        )
    return matched.rollout, matched.sampler_token_log_probs, metadata


def run_real_model_diagnostic(
    *,
    config_path: str | Path,
    seed: int,
    step: int,
    mu_values: Sequence[float],
    rollout_cache: str | Path | None,
    minimum_cosine_similarity: float | None = None,
    maximum_relative_l2_error: float | None = None,
    maximum_absolute_error: float | None = None,
    require_all_mu: bool = False,
    expected_source_commit: str | None = None,
    allow_dirty_source: bool = False,
) -> dict[str, Any]:
    """Load Qwen/GSM8K once and run the non-updating diagnostic in this process."""

    from .evaluation_manifest import load_evaluation_split_manifest
    from .matched_lora_runner import _load_bundle, load_matched_config

    source_receipt = _source_receipt(
        expected_commit=expected_source_commit,
        allow_dirty=allow_dirty_source,
    )
    config = load_matched_config(config_path)
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not present in the matched config")
    bundle, model_snapshot, initialization_digest = _load_bundle(config, seed=seed)
    _validate_real_parameterization(config, bundle)
    if lora_state_digest(bundle.model) != initialization_digest:
        raise RuntimeError("diagnostic did not load the committed shared initialization")
    cache = None if rollout_cache is None else Path(rollout_cache)
    scratch_root = (
        cache.parent / f".{cache.stem}-scratch"
        if cache is not None
        else Path("artifacts/diagnostics/matched_lora_fd_scratch")
    )
    rollout, sampler, rollout_metadata = _real_fixed_rollout(
        config,
        bundle,
        seed=seed,
        step=step,
        cache_path=cache,
        scratch_root=scratch_root,
    )
    min_cosine = (
        config.projected_gradient_min_cosine
        if minimum_cosine_similarity is None
        else minimum_cosine_similarity
    )
    max_relative = (
        config.projected_gradient_max_relative_l2
        if maximum_relative_l2_error is None
        else maximum_relative_l2_error
    )
    thresholds = GradientAgreementThresholds(
        minimum_cosine_similarity=min_cosine,
        maximum_relative_l2_error=max_relative,
        maximum_absolute_error=maximum_absolute_error,
        minimum_passing_mu_count=(len(tuple(mu_values)) if require_all_mu else 1),
        required_mu=config.forward.finite_difference_mu,
    )
    basis_seed = seed + 70_000
    if step != 1:
        raise ValueError(
            "only step 1 is supported: later production bases require replaying prior FO updates"
        )
    report = fixed_rollout_finite_difference_diagnostic(
        bundle,
        rollout,
        sampler,
        config.objective,
        directions=config.forward.directions,
        basis_seed=basis_seed,
        mu_values=mu_values,
        scoring_micro_batch_size=config.forward.scoring_micro_batch_size,
        prompt_groups_per_micro_batch=config.backprop.prompt_groups_per_micro_batch,
        thresholds=thresholds,
        context={
            "config_path": str(Path(config_path).resolve()),
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "model_snapshot": model_snapshot,
            "shared_initialization_digest": initialization_digest,
            "lora": config.lora.as_dict(),
            "seed": seed,
            "step": step,
            "production_fo_basis_seed_formula": "seed + 70000",
            "rollout_metadata": rollout_metadata,
        },
    )
    if report["parameter_layout"]["parameter_count"] != REAL_LORA_PARAMETER_COUNT:
        raise RuntimeError("diagnostic report lost the locked LoRA parameter count")
    manifest = load_evaluation_split_manifest(config.evaluation_manifest)
    report["source"] = source_receipt
    report["runtime"] = _runtime_receipt(bundle.device)
    report["dataset"] = {
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": config.dataset_revision,
        "evaluation_manifest": str(Path(config.evaluation_manifest).resolve()),
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "locked_test_evaluated": False,
    }
    report["production_forward_only_settings"] = {
        "directions": config.forward.directions,
        "finite_difference_mu": config.forward.finite_difference_mu,
        "basis_seed": basis_seed,
        "basis_seed_formula": "seed + 70000",
        "scoring_micro_batch_size": config.forward.scoring_micro_batch_size,
    }
    return report


def _write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--mu", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--minimum-cosine", type=float)
    parser.add_argument("--maximum-relative-l2", type=float)
    parser.add_argument("--maximum-absolute-error", type=float)
    parser.add_argument("--require-all-mu", action="store_true")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args(argv)
    report = run_real_model_diagnostic(
        config_path=args.config,
        seed=args.seed,
        step=args.step,
        mu_values=args.mu,
        rollout_cache=args.rollout_cache,
        minimum_cosine_similarity=args.minimum_cosine,
        maximum_relative_l2_error=args.maximum_relative_l2,
        maximum_absolute_error=args.maximum_absolute_error,
        require_all_mu=args.require_all_mu,
        expected_source_commit=args.expected_source_commit,
        allow_dirty_source=args.allow_dirty_source,
    )
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


__all__ = [
    "DIAGNOSTIC_SCHEMA",
    "GradientAgreementThresholds",
    "fixed_rollout_finite_difference_diagnostic",
    "load_fixed_rollout_cache",
    "run_real_model_diagnostic",
    "save_fixed_rollout_cache",
]


if __name__ == "__main__":  # pragma: no cover
    main()
