from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

import rl_no_backward.plotting as plotting_module
from rl_no_backward.plotting import (
    aggregate_metrics,
    attach_locked_test_results,
    bootstrap_mean_ci,
    find_gsm8k_fd_diagnostic,
    load_matched_results,
    load_results,
    main,
    paired_method_comparisons,
    plot_finite_difference_diagnostics,
    plot_results,
    summarize_results,
)
from rl_no_backward.validate_artifacts import validate_benchmark_artifacts


def _write_matched_metadata(
    root: Path,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    source_commit = "0123456789abcdef0123456789abcdef01234567"
    model_revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    lora = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "layer_indices": list(range(28)),
        "bias": "none",
    }
    config = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": model_revision,
        "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
        "methods": list(methods),
        "seeds": list(seeds),
        "steps": 2,
        "batch_size": 2,
        "group_size": 64,
        "eval_interval": 1,
        "run_test_evaluation": False,
        "test_size": 0,
        "wandb_mode": "offline",
        "record_rollout_provenance": True,
        "dtype": "bfloat16",
        "device": "cuda",
        "attention_implementation": "flash_attention_2",
        "rollout_backend": "vllm_lora",
        "vllm_flash_attn_version": 2,
        "vllm_batch_invariant": False,
        "vllm_enable_v1_multiprocessing": False,
        "vllm_allow_insecure_serialization": False,
        "expected_lora_parameter_count": 1_089_536,
        "lora": lora,
        "max_prompt_tokens": 64,
        "max_new_tokens": 32,
        "eval_batch_size": 8,
        "vllm_kv_cache_memory_bytes": 1024,
        "evaluation_manifest": "configs/gsm8k_standard_lora_eval_manifest.json",
        "dev_source_index_receipt": "configs/gsm8k_standard_lora_dev_source_indices.json",
        "touched_test_exclusions": "configs/gsm8k_touched_test_exclusions.json",
    }
    oracle = {
        "schema": "rl-no-backward-matched-lora-gradient-oracle-v1",
        "passed": True,
        "adapter_parameter_count": 1_089_536,
        "source_commit": source_commit,
        "model_snapshot": f"/models/snapshots/{model_revision}",
        "seed": seeds[0],
        "frozen_base_parameter_digest_before": "1" * 64,
        "frozen_base_parameter_digest_after": "1" * 64,
        "behavior_logprob_digest": "2" * 64,
        "rollout_token_digest": "3" * 64,
        "shared_initialization_digest": "4" * 64,
        "reverse_mode_scope": "dedicated non-updating oracle child only",
        "thresholds": {
            "minimum_cosine_similarity": 0.9,
            "maximum_relative_l2_error": 0.45,
        },
        "report": {
            "cosine_similarity": 0.95,
            "relative_l2_error": 0.25,
            "directions": 8,
            "finite_difference_mu": 10.0,
            "exact_projected_norm": 0.2,
            "finite_difference_norm": 0.19,
            "parameter_integrity": True,
            "parameter_digest_before": "5" * 64,
            "parameter_digest_after": "5" * 64,
        },
        "vllm_reload_receipt": {
            "parameter_count": 1_089_536,
            "state_digest": "4" * 64,
        },
    }
    manifest_payload = json.loads(
        (
            Path(__file__).parents[1] / "configs" / "gsm8k_standard_lora_eval_manifest.json"
        ).read_text(encoding="utf-8")
    )
    dev_source_payload = json.loads(
        (
            Path(__file__).parents[1] / "configs" / "gsm8k_standard_lora_dev_source_indices.json"
        ).read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "implementation": "matched custom, TRL-equivalent",
        "config": config,
        "git_commit": source_commit,
        "git_dirty": False,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": model_revision,
        "resolved_model_snapshot": f"/models/snapshots/{model_revision}",
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
        "adapter_parameter_count": 1_089_536,
        "lora_parameterization": lora,
        "provenance": {
            "source": {"commit": source_commit},
            "model": {
                "id": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": model_revision,
            },
            "dataset": {
                "id": "openai/gsm8k",
                "revision": "740312add88f781978c0658806c59bc2815b9866",
            },
        },
        "objective_contract": {
            "ppo_denominator": "frozen_hf_old_policy_token_logprobs",
            "sampler_correction": "detached_old_hf_over_q_vllm",
            "advantages": "trl_group_centered_sample_std",
            "clip": "token_local",
            "aggregation": "equal_completion_length_normalized",
        },
        "memory_measurement_scope": {
            "allocator": "PyTorch CUDA allocator",
            "scope": "combined HF scorer/trainer plus colocated vLLM engine per child process",
            "vllm_worker_excluded": False,
        },
        "rollout_backend": "vllm_0.22_standard_peft_lora_load_inplace",
        "evaluation_backend": "same_vllm_0.22_standard_peft_lora_engine",
        "vllm_attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "hf_attention_implementation": "flash_attention_2",
        "test_example_ids": [],
        "train_example_ids": ["train-0", "train-1"],
        "val_example_ids": ["dev-0", "dev-1"],
        "excluded_test_example_ids": ["excluded-0"],
        "locked_test_rows_materialized": False,
        "locked_test_accessed": False,
        "locked_test_evaluated": False,
        "evaluation_manifest_sha256": manifest_payload["manifest_sha256"],
        "evaluation_manifest_locked_test_ids_sha256": manifest_payload["locked_test_ids_sha256"],
        "evaluation_manifest_path": config["evaluation_manifest"],
        "dev_source_index_receipt_sha256": dev_source_payload["receipt_sha256"],
        "dev_source_index_receipt_path": config["dev_source_index_receipt"],
        "projected_gradient_oracle": oracle,
    }
    (root / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_synthetic_runs(root: Path, *, include_base: bool = False) -> None:
    methods = ("base", "bp_grpo", "fo_npg") if include_base else ("bp_grpo", "fo_npg")
    _write_matched_metadata(
        root,
        methods=methods,
        seeds=(0, 1, 2),
    )
    if include_base:
        base_record = {
            "kind": "evaluation",
            "split": "validation",
            "method": "base",
            "seed": 0,
            "step": 0,
            "environment_samples": 0,
            "wall_time_seconds": 0.0,
            "forward_calls": 0,
            "backward_calls": 0,
            "peak_gpu_memory_bytes": 2 * 2**30,
            "val_accuracy": 0.4,
            "best_val_accuracy": 0.4,
            "best_step": 0,
        }
        base_path = root / "raw" / "base_seed0.jsonl"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    for method_index, method in enumerate(("bp_grpo", "fo_npg")):
        for seed in range(3):
            path = root / "raw" / f"{method}_seed{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, object]] = []
            for step in range(3):
                samples = step * 128
                wall_time = step * (3.0 + 1.5 * method_index) + seed * 0.05
                accuracy = 0.22 + 0.12 * step - 0.025 * method_index + 0.01 * seed
                expected_reward = accuracy + 0.08
                if step:
                    rollout_seed = 20_000 + seed * 1_000 + step
                    token_digest = f"{10_000 * method_index + 100 * seed + step:064x}"
                    behavior_digest = f"{20_000 * method_index + 100 * seed + step:064x}"
                    combined = hashlib.sha256()
                    combined.update(b"rl-no-backward-rollout-v1/combined\0")
                    combined.update(struct.pack("<Q", rollout_seed))
                    combined.update(bytes.fromhex(token_digest))
                    combined.update(bytes.fromhex(behavior_digest))
                    records.append(
                        {
                            "kind": "train_step",
                            "method": method,
                            "seed": seed,
                            "step": step,
                            # These are deliberately per-step values; the loader
                            # must prefer the cumulative fields below.
                            "environment_samples": 128,
                            "forward_calls": 7 + 4 * method_index,
                            "backward_calls": 2 if method == "bp_grpo" else 0,
                            "cumulative_environment_samples": samples,
                            "cumulative_forward_calls": step * (7 + 4 * method_index),
                            "cumulative_backward_calls": step * (2 if method == "bp_grpo" else 0),
                            "cumulative_teacher_forced_examples": step * (400 + 120 * method_index),
                            "policy_sync_seconds": 0.1,
                            "rollout_and_old_score_seconds": 1.0 + 0.2 * method_index,
                            "optimizer_seconds": 0.5 + 0.4 * method_index,
                            "wall_time_seconds": wall_time,
                            "peak_gpu_memory_bytes": (3 + method_index) * 2**30,
                            "rollout_exact_reward": (
                                0.18 + 0.08 * step - 0.015 * method_index + 0.005 * seed
                            ),
                            "empirical_kl": 0.005 * step * (method_index + 1),
                            "surrogate_improvement": (
                                0.0025 * step * (1.0 - 0.15 * method_index) + 0.0001 * seed
                            ),
                            "step_norm": 0.01 * step * (method_index + 1),
                            "projected_gradient_norm": (
                                0.06 * step * (1.0 + 0.2 * method_index) + 0.002 * seed
                            ),
                            "zero_advantage_fraction": 0.25 / step,
                            "accepted": not (method == "fo_npg" and seed == 2 and step == 1),
                            "rollout_provenance_version": "rl-no-backward-rollout-v1",
                            "rollout_digest_algorithm": "sha256",
                            "rollout_seed": rollout_seed,
                            "rollout_token_digest": token_digest,
                            "behavior_logprob_digest": behavior_digest,
                            "rollout_digest": combined.hexdigest(),
                        }
                    )
                if method == "bp_grpo":
                    evaluation: dict[str, object] = {
                        "kind": "evaluation",
                        "split": "validation",
                        "method": method,
                        "seed": seed,
                        "step": step,
                        "environment_samples": samples,
                        "wall_time_seconds": wall_time,
                        "forward_calls": step * 7,
                        "backward_calls": step * 2,
                        "teacher_forced_examples": step * 400,
                        "peak_gpu_memory_bytes": 3 * 2**30,
                        "test_accuracy": accuracy,
                        "val_accuracy": accuracy,
                        "test_expected_reward": expected_reward,
                    }
                else:
                    # Nested future-facing GSM8K/W&B-style names exercise the
                    # normalisation of slashes and namespaces.
                    evaluation = {
                        "kind": "evaluation",
                        "split": "validation",
                        "method": method,
                        "seed": seed,
                        "step": step,
                        "environment_samples": samples,
                        "wall_time_seconds": wall_time,
                        "forward_calls": step * 11,
                        "backward_calls": 0,
                        "teacher_forced_examples": step * 520,
                        "peak_gpu_memory_bytes": 4 * 2**30,
                        "record_type": "eval",
                        "val_accuracy": accuracy,
                        "config": {"method": method, "seed": seed},
                        "optimizer_step": step,
                        "progress": {
                            "environment_samples": samples,
                            "wall_time_seconds": wall_time,
                            "forward_calls": step * 11,
                            "backward_calls": 0,
                            "teacher_forced_examples": step * 520,
                            "peak_gpu_memory_bytes": 4 * 2**30,
                        },
                        "eval": {
                            "gsm8k": {"exact_match": accuracy},
                            "expected_reward": expected_reward,
                        },
                    }
                records.append(evaluation)
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
    _write_synthetic_support_artifacts(root)


def _write_synthetic_support_artifacts(root: Path) -> None:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    identities = [
        (method, seed)
        for method in config["methods"]
        for seed in (config["seeds"][:1] if method == "base" else config["seeds"])
    ]
    for index, (method, seed) in enumerate(identities):
        stem = f"gsm8k_{method}_seed{seed}"
        checkpoint = root / "checkpoints" / f"{stem}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"placeholder": torch.tensor([index], dtype=torch.int64)}, checkpoint)
        samples = root / "samples" / f"{stem}.json"
        samples.parent.mkdir(parents=True, exist_ok=True)
        samples.write_text(json.dumps([{"example_id": "dev-0", "score": 1.0}]), encoding="utf-8")
        wandb = root / "wandb" / f"offline-run-20260101_00000{index}-run{index}"
        (wandb / "logs").mkdir(parents=True, exist_ok=True)
        (wandb / f"run-run{index}.wandb").write_bytes(b"offline wandb payload")
        (wandb / "logs" / "debug.log").write_text(
            "run started, returning control to user process\nfinishing run\n",
            encoding="utf-8",
        )
    validation = validate_benchmark_artifacts(root)
    assert validation.passed, validation
    (root / "artifact_validation.json").write_text(
        json.dumps(validation.to_dict(), sort_keys=True), encoding="utf-8"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(
        json.dumps(
            row,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _build_locked_bundle(root: Path) -> tuple[pd.DataFrame, Path, Path, Path]:
    training = root / "training"
    locked = root / "locked"
    _write_synthetic_runs(training, include_base=True)
    metadata_path = training / "metadata.json"
    validation_path = training / "artifact_validation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    for method in config["methods"]:
        for seed in config["seeds"][:1] if method == "base" else config["seeds"]:
            source_raw = training / "raw" / f"{method}_seed{seed}.jsonl"
            source_raw.rename(training / "raw" / f"gsm8k_{method}_seed{seed}.jsonl")
    repo = Path(__file__).parents[1]
    manifest_path = repo / "configs" / "gsm8k_standard_lora_eval_manifest.json"
    source_path = repo / "configs" / "gsm8k_standard_lora_locked_source_indices.json"
    audit_path = source_path.with_suffix(".audit.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dev_source_path = repo / config["dev_source_index_receipt"]
    quarantine_path = repo / config["touched_test_exclusions"]
    dev_source = json.loads(dev_source_path.read_text(encoding="utf-8"))
    training_evaluation_inputs = {
        "development_source_receipt_relpath": config["dev_source_index_receipt"],
        "development_source_receipt_file_sha256": _file_sha256(dev_source_path),
        "development_source_receipt_sha256": dev_source["receipt_sha256"],
        "development_source_indices_sha256": source["dev_source_indices_sha256"],
        "development_ids_sha256": manifest["dev_ids_sha256"],
        "quarantine_relpath": config["touched_test_exclusions"],
        "quarantine_file_sha256": _file_sha256(quarantine_path),
        "excluded_test_ids_sha256": manifest["excluded_test_ids_sha256"],
        "development_row_loading": "Dataset.select(committed_dev_source_indices)",
    }
    prior_question_evidence = sorted(
        (
            {
                "metadata_file_sha256": evidence["metadata_file_sha256"],
                "metadata_relpath": evidence["committed_metadata_source"],
                "evidence_relpath": evidence["committed_question_source"],
                "evidence_file_sha256": evidence["question_sample_file_sha256"],
                "question_projection_sha256": evidence["question_projection_sha256"],
                "row_count": evidence["row_count"],
                "access_scope": (
                    "example_id and question only; all other values lexically skipped"
                ),
            }
            for evidence in audit["access_sources"]["prior_exposure_question_samples"]
        ),
        key=lambda evidence: evidence["metadata_file_sha256"],
    )
    source_by_id = {
        example_id: entry["source_index"]
        for example_id, entry in zip(
            manifest["locked_test_example_ids"], source["entries"], strict=True
        )
    }
    identities = [
        (method, seed)
        for seed_index, seed in enumerate(config["seeds"])
        for method in (
            config["methods"]
            if seed_index == 0
            else [method for method in config["methods"] if method != "base"]
        )
    ]
    checkpoints: list[dict[str, object]] = []
    selections: dict[tuple[str, int], dict[str, object]] = {}
    for index, (method, seed) in enumerate(identities):
        selected_step = 0 if method == "base" else 2
        selection_accuracy = (
            0.4
            if method == "base"
            else 0.22 + 0.12 * 2 + 0.01 * seed
            if method == "bp_grpo"
            else 0.22 + 0.12 * 2 - 0.025 + 0.01 * seed
        )
        state = {"synthetic.lora": torch.full((1_089_536,), index, dtype=torch.float16)}
        from rl_no_backward.standard_lora import lora_state_digest

        state_digest = lora_state_digest(state)
        checkpoint_path = training / "checkpoints" / f"gsm8k_{method}_seed{seed}.pt"
        torch.save(
            {
                "method": method,
                "seed": seed,
                "selected_step": selected_step,
                "selection_val_accuracy": selection_accuracy,
                "lora_state": state,
            },
            checkpoint_path,
        )
        selection = {
            "schema": "rl-no-backward-validation-selection-v1",
            "method": method,
            "seed": seed,
            "selected_step": selected_step,
            "selection_split": "development",
            "selection_metric": "exact_match",
            "tie_breaker": "latest_checkpoint",
            "selection_val_accuracy": selection_accuracy,
            "selected_lora_state_digest": state_digest,
            "checkpoint_path": str(checkpoint_path),
        }
        selection_path = training / "selection" / f"gsm8k_{method}_seed{seed}.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
        raw_path = training / "raw" / f"gsm8k_{method}_seed{seed}.jsonl"
        raw_records = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_development_count = sum(
            record.get("kind") == "evaluation" and record.get("split") == "validation"
            for record in raw_records
        )
        learning_gate: dict[str, object] = {
            "schema": "rl-no-backward-matched-learning-gate-v2",
            "passed": True,
            "hard_structural_checks_passed": True,
            "failed_hard_structural_checks": [],
            "hard_structural_checks": {"synthetic_structure": True},
        }
        if method == "base":
            learning_gate["role"] = "non-updating baseline"
        else:
            learning_gate.update(
                method=method,
                seed=seed,
                best_dev_accuracy=selection_accuracy,
            )
        learning_gate_path = training / "learning_gate" / f"gsm8k_{method}_seed{seed}.json"
        learning_gate_path.parent.mkdir(parents=True, exist_ok=True)
        learning_gate_path.write_text(json.dumps(learning_gate, sort_keys=True), encoding="utf-8")
        layout = [
            {
                "name": "synthetic.lora",
                "shape": [1_089_536],
                "dtype": "float16",
            }
        ]
        checkpoint = {
            "method": method,
            "seed": seed,
            "selected_step": selected_step,
            "selection_val_accuracy": selection_accuracy,
            "selected_lora_state_digest": state_digest,
            "lora_parameter_count": 1_089_536,
            "lora_layout_sha256": _canonical_sha256(layout),
            "selection_relpath": str(selection_path.relative_to(training)),
            "selection_file_sha256": _file_sha256(selection_path),
            "checkpoint_relpath": str(checkpoint_path.relative_to(training)),
            "checkpoint_file_sha256": _file_sha256(checkpoint_path),
            "raw_relpath": str(raw_path.relative_to(training)),
            "raw_file_sha256": _file_sha256(raw_path),
            "raw_development_evaluation_count": raw_development_count,
            "recomputed_selection_rule": "maximum val_accuracy, latest step on exact tie",
            "learning_gate_relpath": str(learning_gate_path.relative_to(training)),
            "learning_gate_file_sha256": _file_sha256(learning_gate_path),
        }
        checkpoints.append(checkpoint)
        selections[(method, seed)] = selection
    checkpoint_set_sha256 = _canonical_sha256(checkpoints)
    plan_unsigned: dict[str, object] = {
        "schema": "rl-no-backward-locked-evaluation-plan-v1",
        "training_source_commit": metadata["git_commit"],
        "evaluator_source_commit": "abcdef0123456789abcdef0123456789abcdef01",
        "benchmark_metadata_sha256": _file_sha256(metadata_path),
        "artifact_validation_sha256": _file_sha256(validation_path),
        "model": {
            "id": config["model_name"],
            "revision": config["model_revision"],
            "resolved_snapshot": metadata["resolved_model_snapshot"],
            "dtype": config["dtype"],
            "attention_implementation": config["attention_implementation"],
        },
        "dataset": {
            "id": "openai/gsm8k",
            "config": "main",
            "revision": config["dataset_revision"],
            "official_test_count": manifest["official_test_count"],
            "locked_test_count": manifest["locked_test_count"],
        },
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "locked_test_ids_sha256": manifest["locked_test_ids_sha256"],
        "manifest_file_sha256": _file_sha256(manifest_path),
        "source_index_receipt_sha256": source["receipt_sha256"],
        "source_index_file_sha256": _file_sha256(source_path),
        "source_sealing_audit_file_sha256": _file_sha256(audit_path),
        "official_question_ids_sha256": source["official_question_ids_sha256"],
        "locked_source_indices_sha256": source["locked_source_indices_sha256"],
        "source_sealing_audit_sha256": audit["audit_sha256"],
        "training_evaluation_inputs": training_evaluation_inputs,
        "prior_question_evidence": prior_question_evidence,
        "row_loading": (
            "Dataset.select(committed_locked_source_indices), then authorized opaque-ID reorder"
        ),
        "selection_split": "development",
        "selection_metric": "exact_match",
        "generation": {
            "backend": "vllm_0.22_standard_peft_lora_load_inplace",
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
            "greedy": True,
            "max_prompt_tokens": config["max_prompt_tokens"],
            "max_new_tokens": config["max_new_tokens"],
            "eval_batch_size": config["eval_batch_size"],
        },
        "methods": config["methods"],
        "seeds": config["seeds"],
        "base_policy_semantics": {
            "evaluated_once": True,
            "seed": config["seeds"][0],
            "role": "shared non-updating reference for every trained seed",
        },
        "checkpoints": checkpoints,
        "checkpoint_set_sha256": checkpoint_set_sha256,
    }
    plan = {**plan_unsigned, "plan_sha256": _canonical_sha256(plan_unsigned)}
    locked.mkdir(parents=True)
    (locked / "plan.json").write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    ledger = {
        "schema": "rl-no-backward-locked-consumption-v1",
        "status": "started",
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_set_sha256": checkpoint_set_sha256,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "locked_test_ids_sha256": manifest["locked_test_ids_sha256"],
        "output": str(locked),
        "started_unix_seconds": 1.0,
    }
    ledger_path = training / "locked_test_consumed.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    result_rows: list[dict[str, object]] = []
    for version, identity in enumerate(identities, start=1):
        method, seed = identity
        correct_count = 300 if method == "base" else (340 if method == "bp_grpo" else 320) + seed
        samples: list[dict[str, object]] = []
        for row_index, example_id in enumerate(manifest["locked_test_example_ids"]):
            samples.append(
                {
                    "example_id": example_id,
                    "source_index": source_by_id[example_id],
                    "completion": "Final answer: 1",
                    "predicted_answer": "1",
                    "correct": row_index < correct_count,
                    "response_tokens": 4,
                    "finish_reason": "stop",
                }
            )
        samples_relpath = f"samples/gsm8k_{method}_seed{seed}.jsonl"
        samples_sha256 = _write_canonical_jsonl(locked / samples_relpath, samples)
        prediction_sha256 = _canonical_sha256(
            [
                {
                    "example_id": sample["example_id"],
                    "predicted_answer": sample["predicted_answer"],
                    "correct": sample["correct"],
                }
                for sample in samples
            ]
        )
        selection = selections[identity]
        checkpoint = checkpoints[version - 1]
        policy_version = f"locked/{method}/seed={seed}/selected-step={selection['selected_step']}"
        result_rows.append(
            {
                "method": method,
                "seed": seed,
                "selected_step": selection["selected_step"],
                "selection_val_accuracy": selection["selection_val_accuracy"],
                "selected_lora_state_digest": selection["selected_lora_state_digest"],
                "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
                "policy_sync_seconds": 0.1,
                "evaluation_seconds": 1.0,
                "accuracy": correct_count / 679,
                "correct_count": correct_count,
                "example_count": 679,
                "mean_response_tokens": 4.0,
                "finish_reason_counts": {"stop": 679},
                "truncation_fraction": 0.0,
                "eos_terminated_fraction": 1.0,
                "samples_relpath": samples_relpath,
                "samples_sha256": samples_sha256,
                "prediction_sha256": prediction_sha256,
                "vllm_lora_reload_receipt": {
                    "version": version,
                    "policy_version": policy_version,
                    "state_digest": selection["selected_lora_state_digest"],
                    "adapter_model_sha256": f"{version:064x}",
                    "parameter_count": 1_089_536,
                    "adapter_path": f"/transient/{version}",
                    "active_lora_ids": [1],
                    "prefix_cache_reset": True,
                    "load_inplace": True,
                    "adapter_path_transient": True,
                    "durable_hash_fields": ["state_digest", "adapter_model_sha256"],
                },
                "vllm_policy_probe": {
                    "policy_version": policy_version,
                    "state_digest": selection["selected_lora_state_digest"],
                    "token_id": 1,
                    "selected_token_logprob": -0.1,
                },
            }
        )
    result_unsigned = {
        "schema": "rl-no-backward-locked-evaluation-result-v1",
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_set_sha256": checkpoint_set_sha256,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "locked_test_ids_sha256": manifest["locked_test_ids_sha256"],
        "source_index_receipt_sha256": source["receipt_sha256"],
        "consumption_ledger_sha256": _file_sha256(ledger_path),
        "resolved_model_snapshot": metadata["resolved_model_snapshot"],
        "locked_rows_materialized": True,
        "locked_rows_materialized_after_authorization": True,
        "locked_test_example_count": 679,
        "total_wall_time_seconds": 7.0,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 200,
        "results": result_rows,
        "runtime": {},
    }
    results = {
        **result_unsigned,
        "result_receipt_sha256": _canonical_sha256(result_unsigned),
    }
    results_path = locked / "results.json"
    results_path.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    complete = {
        "schema": "rl-no-backward-locked-evaluation-result-v1",
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "result_receipt_sha256": results["result_receipt_sha256"],
    }
    (locked / "COMPLETE.json").write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")
    return load_matched_results(training), results_path, manifest_path, source_path


def _reseal_result_and_complete(results_path: Path, payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("result_receipt_sha256", None)
    payload = {**unsigned, "result_receipt_sha256": _canonical_sha256(unsigned)}
    results_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    complete = {
        "schema": "rl-no-backward-locked-evaluation-result-v1",
        "status": "complete",
        "plan_sha256": payload["plan_sha256"],
        "result_receipt_sha256": payload["result_receipt_sha256"],
    }
    (results_path.parent / "COMPLETE.json").write_text(
        json.dumps(complete, sort_keys=True), encoding="utf-8"
    )


def _write_validation_and_test_run(root: Path) -> None:
    _write_matched_metadata(root, methods=("bp_grpo",), seeds=(0,))
    path = root / "raw" / "bp_grpo_seed0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 0,
            "environment_samples": 0,
            "wall_time_seconds": 0.0,
            "val_accuracy": 0.2,
            "val_exact_reward": 0.2,
            "val_shaped_reward": 0.3,
        },
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 10,
            "environment_samples": 100,
            "wall_time_seconds": 5.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "val_accuracy": 0.4,
            "val_exact_reward": 0.4,
            "val_shaped_reward": 0.5,
        },
        {
            "kind": "evaluation",
            "split": "test",
            "method": "bp_grpo",
            "seed": 0,
            "step": 10,
            "environment_samples": 100,
            "wall_time_seconds": 5.0,
            "test_accuracy": 0.9,
            "test_exact_reward": 0.9,
            "test_shaped_reward": 0.95,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_bootstrap_is_deterministic_and_ignores_nonfinite_values() -> None:
    first = bootstrap_mean_ci([0.1, 0.3, 0.5, np.nan], samples=500, seed=17)
    second = bootstrap_mean_ci([0.1, 0.3, 0.5, np.inf], samples=500, seed=17)

    assert first == second
    assert first.mean == pytest.approx(0.3)
    assert first.ci_low <= first.mean <= first.ci_high
    assert first.count == 3

    singleton = bootstrap_mean_ci([0.4], samples=500, seed=17)
    assert singleton.mean == pytest.approx(0.4)
    assert np.isnan(singleton.ci_low)
    assert np.isnan(singleton.ci_high)


def test_recursive_loader_normalises_controlled_and_gsm8k_metrics(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    # An interrupted final append is tolerated without discarding prior rows.
    damaged = tmp_path / "raw" / "fo_npg_seed0.jsonl"
    with damaged.open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":')

    with pytest.warns(UserWarning, match="skipping invalid JSON"):
        frame = load_results(tmp_path)

    assert set(frame["method"]) == {"bp_grpo", "fo_npg"}
    evaluations = frame[frame["kind"] == "evaluation"]
    assert len(evaluations) == 18
    assert evaluations["accuracy"].notna().all()
    assert evaluations["expected_reward"].notna().all()
    assert any("gsm8k_exact_match" in source for source in evaluations["score_source"])

    final_forward = frame[
        (frame["method"] == "fo_npg") & (frame["kind"] == "train_step") & (frame["step"] == 2)
    ]
    assert set(final_forward["environment_samples"]) == {256.0}
    assert set(final_forward["forward_calls"]) == {22.0}
    assert final_forward["rollout_exact_reward"].notna().all()
    assert final_forward["surrogate_improvement"].notna().all()
    assert final_forward["step_norm"].notna().all()
    assert final_forward["projected_gradient_norm"].notna().all()


def test_matched_loader_rejects_ambiguous_or_contaminated_artifacts(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    frame = load_matched_results(tmp_path / "raw")

    assert set(frame["method"]) == {"bp_grpo", "fo_npg"}
    assert set(frame["run_id"]) == {
        *(f"matched::bp_grpo::seed={seed}" for seed in range(3)),
        *(f"matched::fo_npg::seed={seed}" for seed in range(3)),
    }

    contaminant = tmp_path / "raw" / "discarded_residual_seed0.jsonl"
    contaminant.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could contaminate"):
        load_matched_results(tmp_path)

    contaminant.unlink()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["implementation"] = "discarded residual core"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to mix pilot results"):
        load_matched_results(tmp_path)


def test_matched_loader_live_validator_rejects_truncated_jsonl(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    with (tmp_path / "raw" / "fo_npg_seed2.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"train_step"')

    with pytest.raises(ValueError, match="live artifact validation.*incomplete or malformed"):
        load_matched_results(tmp_path)


def test_matched_labels_include_optional_focus_method(tmp_path: Path) -> None:
    methods = ("base", "bp_grpo", "fo_npg", "fo_focus_npg")
    _write_matched_metadata(tmp_path, methods=methods, seeds=(0,))
    raw = tmp_path / "raw"
    raw.mkdir()
    for method in methods:
        record = {
            "kind": "evaluation",
            "split": "validation",
            "method": method,
            "seed": 0,
            "step": 0,
            "environment_samples": 0,
            "val_accuracy": 0.5,
            "best_val_accuracy": 0.5,
            "best_step": 0,
        }
        (raw / f"gsm8k_{method}_seed0.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    summary = summarize_results(load_results(raw), bootstrap_samples=10)

    assert list(summary["display_name"]) == ["Base", "BP-GRPO", "FO-NPG", "FO-FOCUS-NPG"]
    assert set(summary["uncertainty_status"]) == {"single_seed_no_interval"}
    assert summary["final_accuracy_ci_low"].isna().all()
    assert set(summary["ci_description"]) == {
        "point estimate only (n=1); no across-seed uncertainty interval"
    }


def test_locked_test_receipt_attaches_by_method_and_seed_without_rewriting_dev(
    tmp_path: Path,
) -> None:
    frame, receipt, manifest, source = _build_locked_bundle(tmp_path)

    attached = attach_locked_test_results(
        frame,
        receipt,
        evaluation_manifest_path=manifest,
        source_index_receipt_path=source,
    )
    summary = summarize_results(attached, bootstrap_samples=100, bootstrap_seed=7)

    assert len(attached[attached["evaluation_split"] == "test"]) == 7
    assert set(summary["performance_sources"]) == {"locked_test_selected_checkpoint"}
    assert summary.loc[summary["method"] == "bp_grpo", "locked_test_accuracy_mean"].item() == (
        pytest.approx(341 / 679)
    )
    assert summary.loc[
        summary["method"] == "bp_grpo", "final_step_validation_accuracy_mean"
    ].item() == pytest.approx(0.47)
    assert set(summary.loc[summary["method"] != "base", "locked_test_accuracy_count"]) == {3}
    assert summary.loc[summary["method"] == "base", "uncertainty_status"].item() == (
        "single_seed_no_interval"
    )
    assert np.isnan(summary.loc[summary["method"] == "base", "locked_test_accuracy_ci_low"].item())

    tampered = json.loads(receipt.read_text(encoding="utf-8"))
    tampered["results"][0]["selection_val_accuracy"] += 0.01
    receipt.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        attach_locked_test_results(
            frame,
            receipt,
            evaluation_manifest_path=manifest,
            source_index_receipt_path=source,
        )


def test_locked_test_receipt_from_another_benchmark_is_rejected(tmp_path: Path) -> None:
    frame, _, manifest, source = _build_locked_bundle(tmp_path / "benchmark-a")
    _, foreign_receipt, _, _ = _build_locked_bundle(tmp_path / "benchmark-b")

    with pytest.raises(ValueError, match="different training benchmark"):
        attach_locked_test_results(
            frame,
            foreign_receipt,
            evaluation_manifest_path=manifest,
            source_index_receipt_path=source,
        )


def test_locked_sample_path_traversal_is_rejected_after_outer_reseal(tmp_path: Path) -> None:
    frame, receipt, manifest, source = _build_locked_bundle(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["results"][0]["samples_relpath"] = "../../foreign-samples.jsonl"
    _reseal_result_and_complete(receipt, payload)

    with pytest.raises(ValueError, match="locked sample path is not canonical"):
        attach_locked_test_results(
            frame,
            receipt,
            evaluation_manifest_path=manifest,
            source_index_receipt_path=source,
        )


def test_locked_attachment_rejects_legacy_v1_source_seal(tmp_path: Path) -> None:
    frame, receipt, manifest, source = _build_locked_bundle(tmp_path)
    legacy_source = tmp_path / "legacy-source-index.json"
    legacy_payload = json.loads(source.read_text(encoding="utf-8"))
    legacy_payload["schema"] = "rl-no-backward-gsm8k-locked-source-index-v1"
    legacy_source.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy_source.with_suffix(".audit.json").write_bytes(
        source.with_suffix(".audit.json").read_bytes()
    )

    with pytest.raises(ValueError, match="unsupported locked source receipt schema"):
        attach_locked_test_results(
            frame,
            receipt,
            evaluation_manifest_path=manifest,
            source_index_receipt_path=legacy_source,
        )


def test_loader_and_summary_preserve_phase_timing_and_allocator_peaks(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "fo_pg_seed0.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "kind": "train_step",
            "method": "fo_pg",
            "seed": 0,
            "step": 1,
            "environment_samples": 8,
            "wall_time_seconds": 5.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "policy_sync_seconds": 0.5,
            "rollout_and_old_score_seconds": 1.25,
            "optimizer_seconds": 2.5,
            "training_phase_seconds": 4.25,
            "peak_gpu_memory_allocated_bytes": 3 * 2**30,
            "peak_gpu_memory_reserved_bytes": 4 * 2**30,
        },
        {
            "kind": "evaluation",
            "method": "fo_pg",
            "seed": 0,
            "step": 1,
            "environment_samples": 8,
            "wall_time_seconds": 7.0,
            "forward_calls": 5,
            "full_prefix_calls": 1,
            "suffix_calls": 4,
            "evaluation_seconds": 2.0,
            "peak_gpu_memory_allocated_bytes": 3 * 2**30,
            "peak_gpu_memory_reserved_bytes": 4 * 2**30,
            "val_accuracy": 0.5,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    frame = load_results(tmp_path)
    train = frame[frame["kind"] == "train_step"].iloc[0]
    assert train["rollout_and_old_score_seconds"] == pytest.approx(1.25)
    assert train["optimizer_seconds"] == pytest.approx(2.5)
    assert train["full_prefix_calls"] == 1
    assert train["suffix_calls"] == 4
    assert train["peak_gpu_memory_bytes"] == 3 * 2**30
    assert train["peak_gpu_memory_allocated_bytes"] == 3 * 2**30
    assert train["peak_gpu_memory_reserved_bytes"] == 4 * 2**30

    summary = summarize_results(frame, bootstrap_samples=10)
    assert summary["total_rollout_and_old_score_seconds_mean"].item() == pytest.approx(1.25)
    assert summary["total_policy_sync_seconds_mean"].item() == pytest.approx(0.5)
    assert summary["total_optimizer_seconds_mean"].item() == pytest.approx(2.5)
    assert summary["total_training_phase_seconds_mean"].item() == pytest.approx(4.25)
    assert summary["total_evaluation_seconds_mean"].item() == pytest.approx(2.0)
    assert summary["final_full_prefix_calls_mean"].item() == 1
    assert summary["final_suffix_calls_mean"].item() == 4
    assert summary["peak_gpu_memory_allocated_bytes_mean"].item() == 3 * 2**30
    assert summary["peak_gpu_memory_reserved_bytes_mean"].item() == 4 * 2**30


def test_training_phase_total_is_unavailable_when_any_component_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "bp_grpo_seed0.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "kind": "train_step",
            "method": "bp_grpo",
            "seed": 0,
            "step": 1,
            "policy_sync_seconds": 0.1,
            "rollout_and_old_score_seconds": 1.0,
            # Missing optimizer time must not be treated as zero.
        },
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 1,
            "val_accuracy": 0.5,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = summarize_results(load_results(tmp_path), bootstrap_samples=10)

    assert np.isnan(summary["total_training_phase_seconds_mean"].item())


def test_aggregation_and_summary_have_bootstrap_ci_and_auc(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    frame = load_results(tmp_path)
    aggregate = aggregate_metrics(
        frame,
        metric="accuracy",
        by="environment_samples",
        bootstrap_samples=300,
        bootstrap_seed=9,
    )
    final = aggregate[aggregate["environment_samples"] == 256]

    assert set(final["count"]) == {3}
    assert (final["ci_low"] <= final["mean"]).all()
    assert (final["mean"] <= final["ci_high"]).all()

    summary = summarize_results(frame, bootstrap_samples=300, bootstrap_seed=9)
    assert list(summary["method"]) == ["bp_grpo", "fo_npg"]
    assert set(summary["runs"]) == {3}
    assert set(summary["final_accuracy_count"]) == {3}
    assert set(summary["ci_description"]) == {"95% percentile bootstrap across independent seeds"}
    assert summary["final_accuracy_mean"].notna().all()
    assert summary["normalised_auc_environment_samples_mean"].between(0, 1).all()
    assert summary.loc[summary["method"] == "fo_npg", "final_backward_calls_mean"].item() == 0

    paired = paired_method_comparisons(frame, bootstrap_samples=300, bootstrap_seed=9)
    assert paired["method"].tolist() == ["fo_npg"]
    assert paired["primary_accuracy_delta_count"].item() == 3
    assert paired["training_time_ratio_mean"].item() == pytest.approx(4.4 / 3.2)


def test_validation_curve_excludes_same_step_test_but_summary_prefers_test(
    tmp_path: Path,
) -> None:
    _write_validation_and_test_run(tmp_path)
    frame = load_results(tmp_path)

    evaluations = frame[frame["kind"] == "evaluation"]
    assert list(evaluations["evaluation_split"]) == ["validation", "validation", "test"]

    curve = aggregate_metrics(
        frame,
        metric="accuracy",
        by="environment_samples",
        bootstrap_samples=100,
    )
    final_curve = curve[curve["environment_samples"] == 100]
    assert len(final_curve) == 1
    assert final_curve["mean"].item() == pytest.approx(0.4)

    summary = summarize_results(frame, bootstrap_samples=100)
    assert summary["final_accuracy_mean"].item() == pytest.approx(0.9)
    assert summary["locked_test_accuracy_mean"].item() == pytest.approx(0.9)
    assert summary["final_step_validation_accuracy_mean"].item() == pytest.approx(0.4)
    assert summary["final_expected_reward_mean"].item() == pytest.approx(0.95)
    assert summary["primary_metric_sources"].item() == "test_accuracy"
    assert summary["normalised_auc_environment_samples_mean"].item() == pytest.approx(0.3)


def test_summary_reports_saved_best_checkpoint_without_rewriting_learning_curve(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw" / "bp_grpo_seed0.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 0,
            "environment_samples": 0,
            "val_accuracy": 0.50,
            "best_val_accuracy": 0.50,
            "best_step": 0,
        },
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 10,
            "environment_samples": 100,
            "val_accuracy": 0.75,
            "best_val_accuracy": 0.75,
            "best_step": 10,
        },
        {
            "kind": "evaluation",
            "method": "bp_grpo",
            "seed": 0,
            "step": 20,
            "environment_samples": 200,
            "val_accuracy": 0.60,
            "best_val_accuracy": 0.75,
            "best_step": 10,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    frame = load_results(tmp_path)
    curve = aggregate_metrics(
        frame,
        metric="accuracy",
        by="environment_samples",
        bootstrap_samples=20,
    )
    assert curve.loc[curve["environment_samples"] == 200, "mean"].item() == pytest.approx(0.60)

    summary = summarize_results(frame, bootstrap_samples=20)
    assert summary["endpoint_accuracy_mean"].item() == pytest.approx(0.60)
    assert summary["final_accuracy_mean"].item() == pytest.approx(0.75)
    assert summary["selected_checkpoint_validation_accuracy_mean"].item() == pytest.approx(0.75)
    assert summary["final_step_validation_accuracy_mean"].item() == pytest.approx(0.60)
    assert summary["selected_step_mean"].item() == pytest.approx(10)
    assert summary["performance_sources"].item() == "validation_selected_checkpoint"


def test_plot_results_writes_readable_png_pdf_and_summary(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    output = tmp_path / "report"
    artifacts = plot_results(
        tmp_path / "raw",
        output,
        bootstrap_samples=200,
        bootstrap_seed=42,
    )

    expected_stems = {
        "learning_curves_samples",
        "learning_curves_wall_time",
        "final_performance",
        "compute_memory_tradeoffs",
        "paired_method_comparisons",
        "training_reward_and_dev_accuracy",
        "grpo_surrogate_objective_change",
        "optimizer_diagnostics",
    }
    assert artifacts["summary_csv"] == output / "summary.csv"
    assert artifacts["paired_comparisons_csv"] == output / "paired_comparisons.csv"
    for stem in expected_stems:
        for extension in ("png", "pdf"):
            path = artifacts[f"{stem}_{extension}"]
            assert path.exists()
            assert path.stat().st_size > 1_000
    image = plt.imread(artifacts["learning_curves_samples_png"])
    assert image.ndim == 3
    assert min(image.shape[:2]) > 400
    reward_image = plt.imread(artifacts["training_reward_and_dev_accuracy_png"])
    surrogate_image = plt.imread(artifacts["grpo_surrogate_objective_change_png"])
    assert min(reward_image.shape[:2]) > 400
    assert min(surrogate_image.shape[:2]) > 400

    summary = pd.read_csv(output / "summary.csv")
    assert list(summary["method"]) == ["bp_grpo", "fo_npg"]
    assert list(summary["display_name"]) == ["BP-GRPO", "FO-NPG"]
    assert "auc_environment_samples_mean" in summary
    assert "peak_gpu_memory_bytes_mean" in summary
    assert "peak_gpu_memory_allocated_bytes_mean" in summary
    assert "peak_gpu_memory_reserved_bytes_mean" in summary
    assert set(summary["memory_measurement_scope"]) == {
        "combined in-process HF scorer/trainer + colocated vLLM engine"
    }
    assert set(summary["forward_call_accounting_scope"]) == {
        "physical audit invocations; not FLOP-equivalent"
    }
    phase = summary.set_index("method")["total_training_phase_seconds_mean"]
    assert phase["bp_grpo"] == pytest.approx(3.2)
    assert phase["fo_npg"] == pytest.approx(4.4)


def test_optimizer_gradient_norm_panel_is_shape_only_and_warns_against_magnitude_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_synthetic_runs(tmp_path)
    frame = load_matched_results(tmp_path / "raw")
    captured: dict[str, plt.Figure] = {}

    def capture_figure(
        figure: plt.Figure,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Path]:
        assert stem == "optimizer_diagnostics"
        captured["figure"] = figure
        return {
            "png": output_dir / f"{stem}.png",
            "pdf": output_dir / f"{stem}.pdf",
        }

    monkeypatch.setattr(plotting_module, "_save_figure", capture_figure)
    with plt.rc_context(plotting_module.PLOT_STYLE):
        plotting_module._plot_optimizer_diagnostics(
            frame,
            tmp_path,
            bootstrap_samples=50,
            bootstrap_seed=9,
        )

    figure = captured["figure"]
    gradient_axis = figure.axes[0]
    assert gradient_axis.get_title(loc="left") == ("Gradient-norm trace · within-seed normalized")
    assert plotting_module.GRADIENT_NORM_SCOPE_NOTE in {
        text.get_text() for text in gradient_axis.texts
    }
    assert "raw magnitudes are not comparable" in plotting_module.GRADIENT_NORM_SCOPE_NOTE
    first_values = {
        line.get_label(): float(np.asarray(line.get_ydata(), dtype=float)[0])
        for line in gradient_axis.lines
    }
    assert first_values == {"BP-GRPO": pytest.approx(1.0), "FO-NPG": pytest.approx(1.0)}
    plt.close(figure)


def test_cli_main_accepts_bootstrap_controls(tmp_path: Path) -> None:
    _write_synthetic_runs(tmp_path)
    output = tmp_path / "cli-report"

    assert (
        main(
            [
                str(tmp_path / "raw"),
                str(output),
                "--bootstrap-samples",
                "50",
                "--bootstrap-seed",
                "3",
            ]
        )
        == 0
    )
    assert (output / "summary.csv").exists()


def _write_fd_diagnostic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic": "real_model_gsm8k_projected_finite_difference",
        "finite_difference": {
            "results": [
                {
                    "mu": mu,
                    "agreement_with_exact_projected_gradient": {
                        "cosine_similarity": cosine,
                        "relative_l2_error": relative_error,
                    },
                    "fisher": {"legacy_to_correct_trace_ratio": ratio},
                }
                for mu, cosine, relative_error, ratio in (
                    (0.25, 0.979, 0.236, 0.019),
                    (0.5, 0.886, 0.504, 0.021),
                    (1.0, 0.992, 0.131, 0.023),
                    (2.0, 0.999, 0.036, 0.022),
                )
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finite_difference_diagnostic_plot_is_public_and_readable(tmp_path: Path) -> None:
    diagnostic = tmp_path / "gsm8k_fd.json"
    _write_fd_diagnostic(diagnostic)

    paths = plot_finite_difference_diagnostics(diagnostic, tmp_path / "figures")

    assert set(paths) == {"png", "pdf"}
    assert all(path.exists() and path.stat().st_size > 1_000 for path in paths.values())
    image = plt.imread(paths["png"])
    assert image.ndim == 3
    assert min(image.shape[:2]) > 400


def test_headline_plot_rejects_legacy_residual_diagnostic_and_requires_bound_oracle(
    tmp_path: Path,
) -> None:
    final_run = tmp_path / "artifacts" / "final_gsm8k"
    _write_synthetic_runs(final_run)
    diagnostic = tmp_path / "artifacts" / "diagnostics" / "gsm8k_fd.json"
    _write_fd_diagnostic(diagnostic)

    assert find_gsm8k_fd_diagnostic(final_run) == diagnostic
    artifacts = plot_results(
        final_run,
        tmp_path / "report",
        bootstrap_samples=30,
        bootstrap_seed=4,
    )

    assert "projected_gradient_oracle_png" not in artifacts
    with pytest.raises(ValueError, match="matched-LoRA gradient oracle exactly bound"):
        plot_results(
            final_run,
            tmp_path / "report-legacy-diagnostic",
            bootstrap_samples=30,
            bootstrap_seed=4,
            finite_difference_diagnostic=diagnostic,
        )

    metadata = json.loads((final_run / "metadata.json").read_text(encoding="utf-8"))
    oracle = final_run / "diagnostics" / "projected_gradient_seed0.json"
    oracle.parent.mkdir()
    oracle.write_text(json.dumps(metadata["projected_gradient_oracle"]), encoding="utf-8")
    explicit = plot_results(
        final_run,
        tmp_path / "report-matched-oracle",
        bootstrap_samples=30,
        bootstrap_seed=4,
        finite_difference_diagnostic=oracle,
    )
    assert explicit["projected_gradient_oracle_png"].exists()
    assert explicit["projected_gradient_oracle_pdf"].exists()
