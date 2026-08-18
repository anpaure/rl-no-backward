from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
import yaml

from rl_no_backward.validate_artifacts import main, validate_benchmark_artifacts


def _config() -> dict[str, Any]:
    return {
        "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
        "methods": ["base", "bp_grpo", "fo_npg"],
        "seeds": [0, 1],
        "steps": 4,
        "batch_size": 2,
        "group_size": 3,
        "scoring_micro_batch_size": 6,
        "eval_interval": 2,
        "run_test_evaluation": True,
        "test_size": 3,
        "wandb_mode": "offline",
        "expected_lora_parameter_count": 30,
        "forward": {
            "directions": 8,
            "line_search_steps": 2,
            "scoring_micro_batch_size": 6,
        },
        "focus": {"family_rank": 2},
    }


def _evaluation(
    method: str,
    seed: int,
    step: int,
    *,
    environment_samples: int,
    forward_calls: int,
    backward_calls: int,
    split: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "evaluation",
        "method": method,
        "seed": seed,
        "step": step,
        "wall_time_seconds": float(step * 10),
        "environment_samples": environment_samples,
        "generated_tokens": environment_samples * 5,
        "scored_tokens": environment_samples
        * (90 if method == "fo_focus_npg" else 18 if method == "fo_npg" else 10),
        "forward_calls": forward_calls,
        "full_prefix_calls": forward_calls,
        "suffix_calls": forward_calls,
        "backward_calls": backward_calls,
        "teacher_forced_examples": (environment_samples * 18 if method == "fo_focus_npg" else 0),
        "peak_gpu_memory_bytes": 1024 + step,
        "peak_gpu_memory_allocated_bytes": 1024 + step,
        "peak_gpu_memory_reserved_bytes": 2048 + step,
        "evaluation_seconds": 1.0,
        "val_accuracy": 0.25 + 0.01 * step,
    }
    if split is not None:
        record.update(
            {
                "split": split,
                "selected_step": 0 if method == "base" else 2,
                "selection_val_accuracy": 0.27,
                "test_accuracy": 0.3,
            }
        )
    return record


def _records(method: str, seed: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    is_base = method == "base"
    is_forward = method in {"fo_npg", "fo_focus_npg"}
    is_focus = method == "fo_focus_npg"
    records = [
        _evaluation(
            method,
            seed,
            0,
            environment_samples=0,
            forward_calls=0,
            backward_calls=0,
        )
    ]
    if not is_base:
        samples_per_step = config["batch_size"] * config["group_size"]
        for step in range(1, config["steps"] + 1):
            backward_calls = 0 if is_forward else step
            forward_calls = step * (18 if is_focus else 8 if is_forward else 3)
            environment_samples = step * samples_per_step
            records.append(
                {
                    "kind": "train_step",
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "wall_time_seconds": float(step * 10 - 1),
                    "environment_samples": environment_samples,
                    "generated_tokens": environment_samples * 5,
                    "scored_tokens": environment_samples
                    * (90 if is_focus else 18 if is_forward else 10),
                    "forward_calls": forward_calls,
                    "full_prefix_calls": forward_calls,
                    "suffix_calls": forward_calls,
                    "backward_calls": backward_calls,
                    "teacher_forced_examples": (environment_samples * 18 if is_focus else 0),
                    "peak_gpu_memory_bytes": 1024 + step,
                    "peak_gpu_memory_allocated_bytes": 1024 + step,
                    "peak_gpu_memory_reserved_bytes": 2048 + step,
                    "rollout_and_old_score_seconds": 2.0,
                    "optimizer_seconds": 3.0,
                    "derivative_variance": None,
                    **(
                        {
                            "focus_bootstrap_b_only": step == 1,
                            "focus_a_rank": min(step - 1, 2),
                            "focus_b_rank": min(step, 2),
                            "focus_a_update_count": step - 1,
                            "focus_b_update_count": step,
                            "focus_state_numel": 8,
                            "focus_state_numel_cap": 64,
                            "focus_first_half_prompts": 1,
                            "focus_second_half_prompts": 1,
                            "focus_cross_sketch_count": 1 if step == 1 else 2,
                            "focus_state_update_policy_evaluations": 0,
                            "line_search_trials": 1,
                            "policy_evaluations": 17,
                            "old_policy_rescore_forward_calls": 1,
                            "old_policy_rescore_teacher_forced_examples": 6,
                            "old_policy_rescore_scored_tokens": 30,
                        }
                        if is_focus
                        else {}
                    ),
                }
            )
            if step % config["eval_interval"] == 0:
                records.append(
                    _evaluation(
                        method,
                        seed,
                        step,
                        environment_samples=environment_samples,
                        forward_calls=forward_calls,
                        backward_calls=backward_calls,
                    )
                )
    if config["run_test_evaluation"]:
        final_samples = (
            0 if is_base else config["steps"] * config["batch_size"] * config["group_size"]
        )
        final_forward = (
            0 if is_base else config["steps"] * (18 if is_focus else 8 if is_forward else 3)
        )
        final_backward = 0 if is_base or is_forward else config["steps"]
        records.append(
            _evaluation(
                method,
                seed,
                0 if is_base else config["steps"],
                environment_samples=final_samples,
                forward_calls=final_forward,
                backward_calls=final_backward,
                split="test",
            )
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _expected_trials(config: dict[str, Any]) -> list[tuple[str, int]]:
    trials: list[tuple[str, int]] = []
    for method in config["methods"]:
        seeds = config["seeds"][:1] if method == "base" else config["seeds"]
        trials.extend((method, seed) for seed in seeds)
    return trials


def _build_complete_artifacts(
    root: Path,
    *,
    include_focus: bool = False,
) -> tuple[Path, dict[str, Any]]:
    output = root / "benchmark"
    config = _config()
    if include_focus:
        config["methods"].append("fo_focus_npg")
    metadata = {
        "config": config,
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "provenance": {
            "model": {
                "id": config["model_name"],
                "revision": "model-snapshot-abc123",
            },
            "dataset": {
                "id": "DigitalLearningGmbH/MATH-lighteval",
                "fingerprint": "dataset-fingerprint-123",
            },
        },
        "train_example_ids": ["train-0", "train-1"],
        "val_example_ids": ["val-0", "val-1"],
        "test_example_ids": ["test-0", "test-1", "test-2"],
        "excluded_test_example_ids": ["excluded-0"],
    }
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for index, (method, seed) in enumerate(_expected_trials(config)):
        stem = f"math_{method}_seed{seed}"
        _write_jsonl(output / "raw" / f"{stem}.jsonl", _records(method, seed, config))
        checkpoint = output / "checkpoints" / f"{stem}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"tiny deterministic checkpoint")
        samples = output / "samples" / f"{stem}.json"
        samples.parent.mkdir(parents=True, exist_ok=True)
        samples.write_text(
            json.dumps([{"example_id": "test-0", "score": 1.0}], allow_nan=False) + "\n",
            encoding="utf-8",
        )

        wandb = output / "wandb" / f"offline-run-20260101_00000{index}-run{index}"
        (wandb / "logs").mkdir(parents=True, exist_ok=True)
        (wandb / f"run-run{index}.wandb").write_bytes(b"offline wandb payload")
        (wandb / "logs" / "debug.log").write_text(
            "run started, returning control to user process\nfinishing run\n",
            encoding="utf-8",
        )
    return output, metadata


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_complete_synthetic_sweep_passes_library_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, _ = _build_complete_artifacts(tmp_path)

    result = validate_benchmark_artifacts(output)

    assert result.status == "complete"
    assert result.passed
    assert result.expected_runs == 5
    assert result.validated_runs == 5
    assert result.incomplete == ()
    assert result.errors == ()
    assert main([str(output)]) == 0
    assert capsys.readouterr().out.startswith("COMPLETE:")


def test_optional_focus_run_is_validated_as_forward_only(tmp_path: Path) -> None:
    output, _ = _build_complete_artifacts(tmp_path, include_focus=True)

    result = validate_benchmark_artifacts(output)

    assert result.status == "complete"
    assert result.expected_runs == 7
    assert result.validated_runs == 7
    assert result.errors == ()

    focus_raw = output / "raw" / "math_fo_focus_npg_seed1.jsonl"
    records = _load_records(focus_raw)
    records[-1]["backward_calls"] = 1
    _write_jsonl(focus_raw, records)
    broken = validate_benchmark_artifacts(output)
    assert broken.status == "invalid"
    assert any("fo_focus_npg/seed-1 is forward-only" in message for message in broken.errors)


@pytest.mark.parametrize(
    ("train_index", "key", "value", "expected_message"),
    [
        (1, "focus_bootstrap_b_only", True, "one B-only bootstrap"),
        (1, "focus_state_update_policy_evaluations", 1, "used extra policy evaluations"),
        (1, "focus_a_update_count", 0, "A-state update count"),
        (1, "focus_b_update_count", 1, "B-state update count"),
        (1, "focus_a_rank", 3, "exceeds the configured rank cap"),
        (1, "focus_first_half_prompts", 0, "first prompt half has the wrong size"),
        (1, "focus_state_numel_cap", 63, "state cap does not match"),
        (1, "focus_state_numel", 65, "state exceeds"),
        (1, "focus_cross_sketch_count", 1, "cross-sketch family count"),
        (1, "policy_evaluations", 18, "not q8 probes plus line search"),
        (1, "forward_calls", 37, "forward-call delta includes unaccounted"),
    ],
)
def test_focus_publication_audit_rejects_inconsistent_raw_telemetry(
    tmp_path: Path,
    train_index: int,
    key: str,
    value: Any,
    expected_message: str,
) -> None:
    output, _ = _build_complete_artifacts(tmp_path, include_focus=True)
    focus_raw = output / "raw" / "math_fo_focus_npg_seed0.jsonl"
    records = _load_records(focus_raw)
    train_records = [record for record in records if record.get("kind") == "train_step"]
    train_records[train_index][key] = value
    _write_jsonl(focus_raw, records)

    result = validate_benchmark_artifacts(output)

    assert result.status == "invalid"
    assert any(expected_message in message for message in result.errors)


def test_focus_publication_audit_requires_q8_in_embedded_metadata(tmp_path: Path) -> None:
    output, metadata = _build_complete_artifacts(tmp_path, include_focus=True)
    metadata["config"]["forward"]["directions"] = 7
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = validate_benchmark_artifacts(output)

    assert result.status == "invalid"
    assert any("FOCUS requires q=8" in message for message in result.errors)


def test_trailing_partial_jsonl_is_clearly_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    raw = output / "raw" / "math_fo_npg_seed1.jsonl"
    with raw.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "train_step"')

    result = validate_benchmark_artifacts(output)

    assert result.status == "incomplete"
    assert not result.passed
    assert any("incomplete or malformed" in message for message in result.incomplete)
    assert main([str(output)]) == 1
    assert capsys.readouterr().out.startswith("INCOMPLETE:")


def test_missing_run_artifacts_and_unfinished_wandb_do_not_silently_pass(tmp_path: Path) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    (output / "checkpoints" / "math_fo_npg_seed1.pt").unlink()
    (output / "samples" / "math_fo_npg_seed1.json").unlink()
    unfinished = max((output / "wandb").glob("offline-run-*"))
    (unfinished / "logs" / "debug.log").write_text(
        "run started, returning control to user process\n",
        encoding="utf-8",
    )

    result = validate_benchmark_artifacts(output)

    assert result.status == "incomplete"
    assert any("missing expected checkpoint" in message for message in result.incomplete)
    assert any("missing expected samples" in message for message in result.incomplete)
    assert any("not finished cleanly" in message for message in result.incomplete)


def test_selection_counter_and_backward_invariants_fail(tmp_path: Path) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    raw = output / "raw" / "math_fo_npg_seed0.jsonl"
    records = _load_records(raw)
    test_record = next(record for record in records if record.get("split") == "test")
    test_record.pop("selected_step")
    test_record.pop("selection_val_accuracy")
    train_records = [record for record in records if record["kind"] == "train_step"]
    train_records[1]["forward_calls"] = train_records[0]["forward_calls"] - 1
    train_records[2]["backward_calls"] = 1
    _write_jsonl(raw, records)

    result = validate_benchmark_artifacts(output)

    assert result.status == "invalid"
    joined = "\n".join(result.errors)
    assert "missing integer selected_step" in joined
    assert "missing a finite selection metric" in joined
    assert "counter 'forward_calls' decreases" in joined
    assert "forward-only" in joined and "backward_calls=1" in joined


def test_opt_in_rollout_provenance_is_required_on_every_train_record(tmp_path: Path) -> None:
    output, metadata = _build_complete_artifacts(tmp_path)
    metadata["config"]["record_rollout_provenance"] = True
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for method_index, (method, seed) in enumerate(_expected_trials(metadata["config"])):
        raw = output / "raw" / f"math_{method}_seed{seed}.jsonl"
        records = _load_records(raw)
        for record in records:
            if record["kind"] != "train_step":
                continue
            step = record["step"]
            identity = 1_000_000 * method_index + 1_000 * seed + step
            rollout_seed = 20_000 + seed * 1_000 + step
            token_digest = f"{identity + 10:064x}"
            logprob_digest = f"{identity + 20:064x}"
            combined = hashlib.sha256()
            combined.update(b"rl-no-backward-rollout-v1/combined\0")
            combined.update(struct.pack("<Q", rollout_seed))
            combined.update(bytes.fromhex(token_digest))
            combined.update(bytes.fromhex(logprob_digest))
            record.update(
                {
                    "rollout_provenance_version": "rl-no-backward-rollout-v1",
                    "rollout_digest_algorithm": "sha256",
                    "rollout_seed": rollout_seed,
                    "rollout_token_digest": token_digest,
                    "behavior_logprob_digest": logprob_digest,
                    "rollout_digest": combined.hexdigest(),
                }
            )
        _write_jsonl(raw, records)

    assert validate_benchmark_artifacts(output).status == "complete"

    damaged = output / "raw" / "math_fo_npg_seed1.jsonl"
    damaged_records = _load_records(damaged)
    target = next(record for record in damaged_records if record["kind"] == "train_step")
    target.pop("behavior_logprob_digest")
    target["rollout_seed"] += 1
    _write_jsonl(damaged, damaged_records)

    result = validate_benchmark_artifacts(output)

    assert result.status == "invalid"
    joined = "\n".join(result.errors)
    assert "behavior_logprob_digest is not a lowercase SHA-256 digest" in joined
    assert "does not match the deterministic scheduled seed" in joined


def test_evaluation_counts_environment_budget_and_explicit_test_are_enforced(
    tmp_path: Path,
) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    raw = output / "raw" / "math_bp_grpo_seed0.jsonl"
    records = _load_records(raw)
    records = [
        record for record in records if not (record["kind"] == "train_step" and record["step"] == 4)
    ]
    test_record = next(record for record in records if record.get("split") == "test")
    test_record.pop("split")
    test_record["environment_samples"] = 18
    _write_jsonl(raw, records)

    result = validate_benchmark_artifacts(output)

    assert result.status == "incomplete"
    joined = "\n".join((*result.incomplete, *result.errors))
    assert "3/4 expected train-step records" in joined
    assert "missing its explicit official-test evaluation" in joined
    assert "environment budget is incomplete" in joined


def test_provenance_nonfinite_values_and_test_overlap_are_rejected(tmp_path: Path) -> None:
    output, metadata = _build_complete_artifacts(tmp_path)
    metadata["provenance"]["model"].pop("revision")
    metadata["provenance"]["dataset"].pop("fingerprint")
    metadata["test_example_ids"][0] = "train-0"
    metadata["telemetry"] = {"bad": float("nan")}
    # YAML can represent NaN, allowing the recursive finite-value check to be
    # exercised without writing non-standard JSON.
    metadata_yaml = output / "metadata.yaml"
    metadata_yaml.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")

    result = validate_benchmark_artifacts(output, metadata_path=metadata_yaml)

    assert result.status == "invalid"
    joined = "\n".join(result.errors)
    assert "immutable model revision" in joined
    assert "dataset revision/fingerprint" in joined
    assert "official-test IDs overlap excluded IDs" in joined
    assert "non-finite numeric value" in joined


def test_external_config_is_checked_against_embedded_lock(tmp_path: Path) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    external = _config()
    external["steps"] = 5
    config_path = tmp_path / "mismatched.yaml"
    config_path.write_text(yaml.safe_dump(external, sort_keys=True), encoding="utf-8")

    result = validate_benchmark_artifacts(output, config_path=config_path)

    assert result.status == "incomplete"
    assert any("disagrees with metadata.config" in message for message in result.errors)
    assert any("expected train-step records" in message for message in result.incomplete)


def test_external_config_locks_vllm_process_and_serialization_mode(tmp_path: Path) -> None:
    output, metadata = _build_complete_artifacts(tmp_path)
    metadata["config"].update(
        {
            "rollout_backend": "vllm",
            "vllm_enable_v1_multiprocessing": False,
            "vllm_allow_insecure_serialization": False,
        }
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    external = dict(metadata["config"])
    external["vllm_enable_v1_multiprocessing"] = True
    external["vllm_allow_insecure_serialization"] = True
    config_path = tmp_path / "wrong-vllm-process.yaml"
    config_path.write_text(yaml.safe_dump(external, sort_keys=True), encoding="utf-8")

    result = validate_benchmark_artifacts(output, config_path=config_path)

    assert result.status == "invalid"
    joined = "\n".join(result.errors)
    assert "vllm_enable_v1_multiprocessing" in joined
    assert "vllm_allow_insecure_serialization" in joined


def test_machine_readable_cli_failure_is_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, _ = _build_complete_artifacts(tmp_path)
    (output / "metadata.json").unlink()

    assert main([str(output), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete"
    assert payload["passed"] is False
