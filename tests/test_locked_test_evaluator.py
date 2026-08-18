from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

import rl_no_backward.locked_source_sealing as sealing_module
import rl_no_backward.locked_test_evaluator as evaluator_module
from rl_no_backward.evaluation_manifest import build_evaluation_split_manifest
from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.locked_source_sealing import (
    DevelopmentSourceEntry,
    PriorQuestionRecord,
    seal_locked_source_indices,
)
from rl_no_backward.locked_test_evaluator import (
    build_locked_evaluation_plan,
    evaluate_locked_policy,
    load_locked_examples,
    load_locked_source_index_receipt,
    run_locked_test_evaluation,
    write_locked_source_index_receipt,
)
from rl_no_backward.standard_lora import lora_state_digest
from rl_no_backward.vllm_lora_rollout import LoRANextTokenProbe, LoRAReloadReceipt


def _official_examples(count: int = 4) -> tuple[GSM8KExample, ...]:
    return tuple(
        GSM8KExample(
            question=f"Official question {index}?",
            answer=f"reasoning\n#### {index}",
            split="test",
            source_index=index,
        )
        for index in range(count)
    )


def _manifest_and_receipt(tmp_path: Path):
    examples = _official_examples()
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"test_example_ids": [examples[0].example_id]}), encoding="utf-8")
    manifest = build_evaluation_split_manifest(
        [example.example_id for example in examples],
        [prior],
        dev_size=1,
        locked_test_size=2,
        seed=7,
        namespace="locked-evaluator-test",
    )
    by_id = {example.example_id: example for example in examples}
    receipt, audit = seal_locked_source_indices(
        manifest,
        [example.question for example in examples],
        [PriorQuestionRecord(examples[0].example_id, examples[0].question)],
        [
            DevelopmentSourceEntry(
                by_id[example_id].source_index,
                example_id,
            )
            for example_id in manifest.dev_example_ids
        ],
        dataset_revision="dataset-commit",
        access_sources={
            "prior_exposure_question_samples": [
                {
                    "fields_used": ["example_id", "question"],
                    "projection_parser": (
                        "selective JSON lexer; nonselected values skipped without decoding"
                    ),
                    "reference_answer_values_used": False,
                    "completion_values_used": False,
                }
            ]
        },
    )
    return examples, manifest, receipt, audit


def test_locked_source_receipt_is_answer_free_bound_and_immutable(tmp_path) -> None:
    examples, manifest, receipt, _ = _manifest_and_receipt(tmp_path)
    path = write_locked_source_index_receipt(tmp_path / "locked-indices.json", receipt)
    serialized = path.read_text(encoding="utf-8")
    assert "answer" not in serialized
    assert all(example.question not in serialized for example in examples)
    assert (
        load_locked_source_index_receipt(path, manifest, dataset_revision="dataset-commit")
        == receipt
    )
    write_locked_source_index_receipt(path, receipt)

    tampered = json.loads(serialized)
    tampered["entries"][0]["source_index"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted|digest"):
        load_locked_source_index_receipt(path, manifest, dataset_revision="dataset-commit")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_locked_source_index_receipt(path, receipt)


def test_locked_loader_selects_only_committed_indices_without_full_iteration(tmp_path) -> None:
    examples, manifest, receipt, _ = _manifest_and_receipt(tmp_path)

    class SelectOnlyDataset:
        def __init__(self) -> None:
            self.selected_indices = None

        def __len__(self):
            return len(examples)

        def __iter__(self):
            raise AssertionError("official test rows must not be iterated")

        def select(self, indices):
            self.selected_indices = list(indices)
            return [
                {"question": examples[index].question, "answer": examples[index].answer}
                for index in indices
            ]

    dataset = SelectOnlyDataset()
    loaded = load_locked_examples(
        manifest,
        receipt,
        dataset_revision="dataset-commit",
        dataset_loader=lambda **_: dataset,
    )
    assert dataset.selected_indices == [entry.source_index for entry in receipt.entries]
    assert tuple(example.example_id for example in loaded) == manifest.locked_test_example_ids
    assert all(example.source_index in dataset.selected_indices for example in loaded)


def _write_frozen_benchmark(tmp_path: Path, manifest, *, seeds=(0, 1, 2)) -> Path:
    benchmark = tmp_path / "benchmark"
    (benchmark / "checkpoints").mkdir(parents=True)
    (benchmark / "selection").mkdir()
    (benchmark / "raw").mkdir()
    (benchmark / "learning_gate").mkdir()
    config = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "1" * 40,
        "dataset_revision": "dataset-commit",
        "dtype": "bfloat16",
        "device": "cuda",
        "attention_implementation": "flash_attention_2",
        "methods": ["base", "bp_grpo", "fo_npg"],
        "seeds": list(seeds),
        "run_test_evaluation": False,
        "test_size": 0,
        "dev_size": 1,
        "evaluation_manifest": "manifest.json",
        "dev_source_index_receipt": "dev.json",
        "touched_test_exclusions": "quarantine.json",
        "rollout_backend": "vllm_lora",
        "vllm_flash_attn_version": 2,
        "vllm_batch_invariant": False,
        "vllm_enable_v1_multiprocessing": False,
        "vllm_allow_insecure_serialization": False,
        "vllm_enforce_eager": False,
        "vllm_kv_cache_memory_bytes": 1024,
        "max_prompt_tokens": 32,
        "max_new_tokens": 16,
        "eval_batch_size": 2,
        "expected_lora_parameter_count": 1,
        "lora": {
            "rank": 1,
            "alpha": 1,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
            "layer_indices": [0],
            "bias": "none",
        },
    }
    metadata = {
        "schema_version": 1,
        "config": config,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": "dataset-commit",
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "resolved_model_snapshot": f"/snapshots/{config['model_revision']}",
        "evaluation_backend": "same_vllm_0.22_standard_peft_lora_engine",
        "vllm_attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "hf_attention_implementation": "flash_attention_2",
        "adapter_parameter_count": 1,
        "lora_parameterization": config["lora"],
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "evaluation_manifest_path": "manifest.json",
        "evaluation_manifest_dev_ids_sha256": manifest.dev_ids_sha256,
        "evaluation_manifest_locked_test_ids_sha256": manifest.locked_test_ids_sha256,
        "val_example_ids": list(manifest.dev_example_ids),
        "excluded_test_example_ids": list(manifest.excluded_test_example_ids),
        "dev_source_index_receipt_path": "dev.json",
        "dev_source_index_receipt_sha256": "d" * 64,
        "development_row_loading": "Dataset.select(committed_dev_source_indices)",
        "test_example_ids": [],
        "locked_test_rows_materialized": False,
        "locked_test_accessed": False,
        "locked_test_evaluated": False,
    }
    (benchmark / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (benchmark / "artifact_validation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "status": "complete",
                "output_dir": str(benchmark.resolve()),
                "expected_runs": 1 + 2 * len(seeds),
                "validated_runs": 1 + 2 * len(seeds),
                "errors": [],
                "incomplete": [],
            }
        ),
        encoding="utf-8",
    )
    policy_pairs = [
        (method, seed)
        for seed_index, seed in enumerate(seeds)
        for method in (("base", "bp_grpo", "fo_npg") if seed_index == 0 else ("bp_grpo", "fo_npg"))
    ]
    for value, (method, seed) in enumerate(policy_pairs, start=1):
        selected_step = 0 if method == "base" else value
        state = {"layer.lora_A.default.weight": torch.tensor([float(value)])}
        checkpoint = benchmark / "checkpoints" / f"gsm8k_{method}_seed{seed}.pt"
        torch.save(
            {
                "method": method,
                "seed": seed,
                "selected_step": selected_step,
                "selection_val_accuracy": value / 10,
                "lora_state": state,
            },
            checkpoint,
        )
        selection = {
            "schema": "rl-no-backward-validation-selection-v1",
            "method": method,
            "seed": seed,
            "selected_step": selected_step,
            "selection_split": "development",
            "selection_metric": "exact_match",
            "tie_breaker": "latest_checkpoint",
            "selection_val_accuracy": value / 10,
            "selected_lora_state_digest": lora_state_digest(state),
            "checkpoint_path": str(checkpoint.resolve()),
        }
        (benchmark / "selection" / f"gsm8k_{method}_seed{seed}.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
        raw_records = (
            [
                {
                    "kind": "evaluation",
                    "split": "validation",
                    "method": method,
                    "seed": seed,
                    "step": 0,
                    "val_accuracy": value / 10,
                }
            ]
            if method == "base"
            else [
                {
                    "kind": "evaluation",
                    "split": "validation",
                    "method": method,
                    "seed": seed,
                    "step": 0,
                    "val_accuracy": max(0.0, value / 10 - 0.1),
                },
                {
                    "kind": "evaluation",
                    "split": "validation",
                    "method": method,
                    "seed": seed,
                    "step": selected_step,
                    "val_accuracy": value / 10,
                },
            ]
        )
        (benchmark / "raw" / f"gsm8k_{method}_seed{seed}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in raw_records), encoding="utf-8"
        )
        learning_gate = {
            "schema": "rl-no-backward-matched-learning-gate-v2",
            "passed": True,
            "hard_structural_checks_passed": True,
            "failed_hard_structural_checks": [],
            "hard_structural_checks": {},
        }
        if method == "base":
            learning_gate["role"] = "non-updating baseline"
        else:
            learning_gate["method"] = method
            learning_gate["seed"] = seed
            learning_gate["best_dev_accuracy"] = value / 10
        (benchmark / "learning_gate" / f"gsm8k_{method}_seed{seed}.json").write_text(
            json.dumps(learning_gate), encoding="utf-8"
        )
    return benchmark


def _patch_synthetic_plan_guards(monkeypatch) -> None:
    monkeypatch.setattr(evaluator_module, "EXPECTED_OFFICIAL_TEST_COUNT", 4)
    monkeypatch.setattr(evaluator_module, "EXPECTED_EXCLUDED_TEST_COUNT", 1)
    monkeypatch.setattr(evaluator_module, "EXPECTED_DEV_COUNT", 1)
    monkeypatch.setattr(evaluator_module, "EXPECTED_LOCKED_TEST_COUNT", 2)
    monkeypatch.setattr(evaluator_module, "EXPECTED_DATASET_REVISION", "dataset-commit")
    monkeypatch.setattr(sealing_module, "EXPECTED_OFFICIAL_TEST_COUNT", 4)
    monkeypatch.setattr(sealing_module, "EXPECTED_EXCLUDED_COUNT", 1)
    monkeypatch.setattr(sealing_module, "EXPECTED_DEV_COUNT", 1)
    monkeypatch.setattr(sealing_module, "EXPECTED_LOCKED_COUNT", 2)
    monkeypatch.setattr(evaluator_module, "validate_locked_source_audit", lambda *_: None)
    monkeypatch.setattr(
        evaluator_module,
        "_validate_bound_split_inputs",
        lambda *_: {
            "development_source_receipt_relpath": "dev.json",
            "quarantine_relpath": "quarantine.json",
        },
    )
    monkeypatch.setattr(
        evaluator_module,
        "_validate_prior_question_evidence",
        lambda *_: (
            {
                "metadata_file_sha256": "a" * 64,
                "evidence_relpath": "evidence.json",
            },
        ),
    )


def test_plan_freezes_all_selected_checkpoints_without_loading_dataset(
    monkeypatch, tmp_path
) -> None:
    _, manifest, receipt, audit = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    receipt_path.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    benchmark = _write_frozen_benchmark(tmp_path, manifest)

    def fake_git(*args, **_):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        raise AssertionError(args)

    monkeypatch.setattr(evaluator_module, "_git_output", fake_git)
    _patch_synthetic_plan_guards(monkeypatch)
    plan = build_locked_evaluation_plan(
        benchmark,
        manifest_path,
        receipt_path,
        enforce_committed_inputs=False,
    )
    assert plan["dataset"]["locked_test_count"] == 2
    assert [row["method"] for row in plan["checkpoints"]] == [
        "base",
        "bp_grpo",
        "fo_npg",
        "bp_grpo",
        "fo_npg",
        "bp_grpo",
        "fo_npg",
    ]
    assert len(plan["checkpoint_set_sha256"]) == 64
    assert len(plan["plan_sha256"]) == 64
    assert plan["row_loading"] == (
        "Dataset.select(committed_locked_source_indices), then authorized opaque-ID reorder"
    )
    assert plan["base_policy_semantics"] == {
        "evaluated_once": True,
        "seed": 0,
        "role": "shared non-updating reference for every trained seed",
    }


def test_three_seed_plan_evaluates_one_shared_base_and_each_trained_policy(
    monkeypatch, tmp_path
) -> None:
    _, manifest, receipt, audit = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    receipt_path.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    benchmark = _write_frozen_benchmark(tmp_path, manifest, seeds=(0, 1, 2))

    def fake_git(*args, **_):
        values = {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("rev-parse", "--show-toplevel"): str(tmp_path),
        }
        return values[args]

    monkeypatch.setattr(evaluator_module, "_git_output", fake_git)
    _patch_synthetic_plan_guards(monkeypatch)
    plan = build_locked_evaluation_plan(
        benchmark,
        manifest_path,
        receipt_path,
        enforce_committed_inputs=False,
    )
    assert [(row["method"], row["seed"]) for row in plan["checkpoints"]] == [
        ("base", 0),
        ("bp_grpo", 0),
        ("fo_npg", 0),
        ("bp_grpo", 1),
        ("fo_npg", 1),
        ("bp_grpo", 2),
        ("fo_npg", 2),
    ]
    assert plan["base_policy_semantics"]["seed"] == 0
    assert len(plan["checkpoints"]) == 7
    assert all("raw_file_sha256" in row for row in plan["checkpoints"])
    assert all("learning_gate_file_sha256" in row for row in plan["checkpoints"])


def test_locked_plan_rejects_nonfinal_counts_before_artifact_reads(tmp_path) -> None:
    _, manifest, receipt, audit = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    receipt_path.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="1319/384/256/679"):
        build_locked_evaluation_plan(tmp_path / "missing-benchmark", manifest_path, receipt_path)


def test_training_metadata_rejects_any_seed_set_except_exact_final_three(tmp_path) -> None:
    _, manifest, _, _ = _manifest_and_receipt(tmp_path)
    benchmark = _write_frozen_benchmark(tmp_path, manifest, seeds=(0,))
    metadata = json.loads((benchmark / "metadata.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exact seeds"):
        evaluator_module._validate_training_metadata(metadata, manifest)


@pytest.mark.parametrize(
    ("expected_runs", "validated_runs"),
    ((6, 6), (7, 6), (6, 7), (8, 8)),
)
def test_plan_requires_exactly_seven_validated_runs(
    monkeypatch, tmp_path, expected_runs, validated_runs
) -> None:
    _, manifest, receipt, audit = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    receipt_path.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    benchmark = _write_frozen_benchmark(tmp_path, manifest)
    validation_path = benchmark / "artifact_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(expected_runs=expected_runs, validated_runs=validated_runs)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    _patch_synthetic_plan_guards(monkeypatch)
    with pytest.raises(ValueError, match="artifact validation"):
        build_locked_evaluation_plan(
            benchmark,
            manifest_path,
            receipt_path,
            enforce_committed_inputs=False,
        )


def test_selection_is_recomputed_from_raw_latest_max_and_gate_is_bound(tmp_path) -> None:
    _, manifest, _, _ = _manifest_and_receipt(tmp_path)
    benchmark = _write_frozen_benchmark(tmp_path, manifest)
    label = "gsm8k_bp_grpo_seed0"
    raw_path = benchmark / "raw" / f"{label}.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    selected = json.loads((benchmark / "selection" / f"{label}.json").read_text(encoding="utf-8"))
    records.append(
        {
            "kind": "evaluation",
            "split": "validation",
            "method": "bp_grpo",
            "seed": 0,
            "step": selected["selected_step"] + 10,
            "val_accuracy": selected["selection_val_accuracy"],
        }
    )
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="latest-step max"):
        evaluator_module._selection_and_checkpoint_receipt(
            benchmark, method="bp_grpo", seed=0, expected_parameter_count=1
        )

    # Restore the raw log and independently prove that a failed structural
    # learning gate cannot enter a locked plan.
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in records[:-1]), encoding="utf-8")
    gate_path = benchmark / "learning_gate" / f"{label}.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["hard_structural_checks_passed"] = False
    gate["failed_hard_structural_checks"] = ["policy_digest_changed"]
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="structurally passing"):
        evaluator_module._selection_and_checkpoint_receipt(
            benchmark, method="bp_grpo", seed=0, expected_parameter_count=1
        )


def test_benchmark_and_config_paths_reject_symlinks_and_parent_escapes(tmp_path) -> None:
    _, manifest, _, _ = _manifest_and_receipt(tmp_path)
    benchmark = _write_frozen_benchmark(tmp_path, manifest)
    raw_path = benchmark / "raw" / "gsm8k_bp_grpo_seed0.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    raw_path.unlink()
    os.symlink(outside, raw_path)
    with pytest.raises(ValueError, match="symlink"):
        evaluator_module._selection_and_checkpoint_receipt(
            benchmark, method="bp_grpo", seed=0, expected_parameter_count=1
        )
    with pytest.raises(ValueError, match="must not be absolute or contain"):
        evaluator_module._resolve_config_input_path(
            "../outside.json", tmp_path, name="adversarial config path"
        )
    linked_config = tmp_path / "linked-config.json"
    os.symlink(outside, linked_config)
    with pytest.raises(ValueError, match="symlink"):
        evaluator_module._resolve_config_input_path(
            linked_config.name, tmp_path, name="adversarial config symlink"
        )


def test_selection_and_source_audit_paths_cannot_escape_or_traverse_symlinks(
    monkeypatch, tmp_path
) -> None:
    _, manifest, receipt, audit = _manifest_and_receipt(tmp_path)
    benchmark = _write_frozen_benchmark(tmp_path, manifest)
    selection_path = benchmark / "selection" / "gsm8k_bp_grpo_seed0.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["checkpoint_path"] = str(tmp_path / "outside" / "gsm8k_bp_grpo_seed0.pt")
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="selection contract"):
        evaluator_module._selection_and_checkpoint_receipt(
            benchmark, method="bp_grpo", seed=0, expected_parameter_count=1
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    outside_audit = tmp_path / "outside-audit.json"
    outside_audit.write_text(json.dumps(audit), encoding="utf-8")
    os.symlink(outside_audit, receipt_path.with_suffix(".audit.json"))
    _patch_synthetic_plan_guards(monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        build_locked_evaluation_plan(
            benchmark,
            manifest_path,
            receipt_path,
            enforce_committed_inputs=False,
        )


def test_real_split_bindings_link_dev_quarantine_manifest_and_seal() -> None:
    worktree = Path.cwd()
    manifest_path = worktree / "configs/gsm8k_standard_lora_eval_manifest.json"
    receipt_path = worktree / "configs/gsm8k_standard_lora_locked_source_indices.json"
    audit_path = receipt_path.with_suffix(".audit.json")
    manifest = evaluator_module.load_evaluation_split_manifest(manifest_path)
    receipt = load_locked_source_index_receipt(
        receipt_path,
        manifest,
        dataset_revision="740312add88f781978c0658806c59bc2815b9866",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((worktree / "configs/gsm8k_matched_lora_final.yaml").read_text())
    metadata = {
        "dev_source_index_receipt_sha256": audit["access_sources"]["development_source_receipt"][
            "receipt_sha256"
        ],
        "dev_source_index_receipt_path": config["dev_source_index_receipt"],
        "evaluation_manifest_path": config["evaluation_manifest"],
    }
    bound = evaluator_module._validate_bound_split_inputs(
        config,
        metadata,
        manifest,
        receipt,
        audit,
        manifest_path,
        worktree,
    )
    assert bound["development_ids_sha256"] == manifest.dev_ids_sha256
    assert bound["excluded_test_ids_sha256"] == manifest.excluded_test_ids_sha256
    bad_metadata = {**metadata, "dev_source_index_receipt_sha256": "0" * 64}
    with pytest.raises(ValueError, match="same dev/manifest"):
        evaluator_module._validate_bound_split_inputs(
            config,
            bad_metadata,
            manifest,
            receipt,
            audit,
            manifest_path,
            worktree,
        )


def test_unauthorized_or_previously_consumed_run_never_calls_dataset_loader(tmp_path) -> None:
    calls = 0

    def forbidden_loader(**_):
        nonlocal calls
        calls += 1
        raise AssertionError("locked dataset loader must not be called")

    output = tmp_path / "output"
    with pytest.raises(PermissionError, match="authorize"):
        run_locked_test_evaluation(
            tmp_path / "benchmark",
            tmp_path / "manifest.json",
            tmp_path / "indices.json",
            output,
            expected_plan_sha256="a" * 64,
            authorize_locked_test_once=False,
            dataset_loader=forbidden_loader,
        )
    assert calls == 0
    assert not output.exists()

    output.mkdir()
    with pytest.raises(FileExistsError, match="reuse"):
        run_locked_test_evaluation(
            tmp_path / "benchmark",
            tmp_path / "manifest.json",
            tmp_path / "indices.json",
            output,
            expected_plan_sha256="a" * 64,
            authorize_locked_test_once=True,
            dataset_loader=forbidden_loader,
        )
    assert calls == 0

    output.rmdir()
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    ledger = benchmark / "locked_test_consumed.json"
    ledger.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already consumed"):
        run_locked_test_evaluation(
            benchmark,
            tmp_path / "manifest.json",
            tmp_path / "indices.json",
            output,
            expected_plan_sha256="a" * 64,
            authorize_locked_test_once=True,
            dataset_loader=forbidden_loader,
        )
    assert calls == 0


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **_):
        return messages[-1]["content"]

    def __call__(self, prompt, **_):
        return {"input_ids": [len(prompt)]}

    def decode(self, token_ids, **_):
        return {1: "The final answer is 3", 2: "The final answer is 99"}[token_ids[0]]


class _FakeGenerator:
    policy_version = "selected-policy"

    def __init__(self):
        self.cursor = 0

    def generate_greedy(self, prompt_ids, **_):
        count = len(prompt_ids)
        values = (1, 2)[self.cursor : self.cursor + count]
        self.cursor += count
        return SimpleNamespace(
            response_token_ids=tuple((value,) for value in values),
            finish_reasons=tuple("stop" for _ in values),
            policy_version=self.policy_version,
        )


def test_locked_policy_evaluation_records_exact_predictions_in_manifest_order() -> None:
    examples = (
        GSM8KExample("First?", "work\n#### 3", "test", 10),
        GSM8KExample("Second?", "work\n#### 4", "test", 11),
    )
    metrics, samples = evaluate_locked_policy(
        _FakeTokenizer(),
        _FakeGenerator(),
        examples,
        max_prompt_tokens=32,
        max_new_tokens=16,
        eval_batch_size=1,
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["correct_count"] == 1
    assert metrics["finish_reason_counts"] == {"stop": 2}
    assert [sample["example_id"] for sample in samples] == [
        example.example_id for example in examples
    ]
    assert [sample["predicted_answer"] for sample in samples] == ["3", "99"]


def test_authorized_run_consumes_ledger_once_and_writes_every_policy(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    output = tmp_path / "locked-output"
    ledger = benchmark / "locked_test_consumed.json"
    config = {
        "model_name": "model",
        "model_revision": "1" * 40,
        "dataset_revision": "dataset-commit",
        "dtype": "bfloat16",
        "device": "cuda",
        "attention_implementation": "flash_attention_2",
        "max_prompt_tokens": 32,
        "max_new_tokens": 16,
        "eval_batch_size": 64,
        "vllm_kv_cache_memory_bytes": 1024,
        "vllm_enforce_eager": False,
        "vllm_flash_attn_version": 2,
        "lora": {"rank": 1},
    }
    (benchmark / "metadata.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    ids = tuple(f"locked-{index}" for index in range(679))
    policy_pairs = [
        ("base", 0, "a"),
        ("bp_grpo", 0, "b"),
        ("fo_npg", 0, "c"),
        ("bp_grpo", 1, "d"),
        ("fo_npg", 1, "e"),
        ("bp_grpo", 2, "f"),
        ("fo_npg", 2, "0"),
    ]
    checkpoints = [
        {
            "method": method,
            "seed": seed,
            "selected_step": index,
            "selection_val_accuracy": 0.5,
            "selected_lora_state_digest": character * 64,
            "checkpoint_file_sha256": str(index) * 64,
            "checkpoint_relpath": f"checkpoints/{method}-seed{seed}.pt",
        }
        for index, (method, seed, character) in enumerate(policy_pairs, start=1)
    ]
    plan = {
        "plan_sha256": "d" * 64,
        "checkpoint_set_sha256": "e" * 64,
        "evaluation_manifest_sha256": "f" * 64,
        "locked_test_ids_sha256": "1" * 64,
        "source_index_receipt_sha256": "2" * 64,
        "benchmark_metadata_sha256": evaluator_module._file_digest(benchmark / "metadata.json"),
        "seeds": [0, 1, 2],
        "checkpoints": checkpoints,
    }
    monkeypatch.setattr(evaluator_module, "build_locked_evaluation_plan", lambda *_: plan)
    monkeypatch.setattr(evaluator_module, "_require_output_outside_worktree", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.digest = ""

    model = FakeModel()
    monkeypatch.setattr(
        evaluator_module,
        "_load_evaluation_model",
        lambda *_args, **_kwargs: (model, SimpleNamespace(), "/snapshots/pinned"),
    )
    monkeypatch.setattr(
        evaluator_module, "create_standard_lora_vllm_engine", lambda *_a, **_k: object()
    )
    states = [{"digest": checkpoint["selected_lora_state_digest"]} for checkpoint in checkpoints]
    monkeypatch.setattr(
        evaluator_module,
        "_checkpoint_state",
        lambda _benchmark, checkpoint: next(
            state for state in states if state["digest"] == checkpoint["selected_lora_state_digest"]
        ),
    )

    def fake_load_state(target, state):
        target.digest = state["digest"]

    monkeypatch.setattr(evaluator_module, "load_lora_state_dict", fake_load_state)
    monkeypatch.setattr(evaluator_module, "assert_lora_frozen", lambda _: None)
    monkeypatch.setattr(
        evaluator_module,
        "lora_state_digest",
        lambda value: value.digest if hasattr(value, "digest") else value["digest"],
    )

    class FakeReloadableGenerator:
        def __init__(self, *_args, **_kwargs):
            self.policy_version = None

        def sync(self, target, export_root, *, version, policy_version):
            self.policy_version = policy_version
            return LoRAReloadReceipt(
                version=version,
                policy_version=policy_version,
                state_digest=target.digest,
                adapter_model_sha256="3" * 64,
                parameter_count=1,
                adapter_path=str(export_root / f"adapter-{version}"),
                active_lora_ids=(1,),
                prefix_cache_reset=True,
            )

        def probe_next_token(self, _prompt):
            return LoRANextTokenProbe(
                policy_version=self.policy_version,
                state_digest=model.digest,
                token_id=1,
                selected_token_logprob=-0.5,
            )

    monkeypatch.setattr(evaluator_module, "ReloadableLoRAGenerator", FakeReloadableGenerator)
    manifest = SimpleNamespace(
        manifest_sha256=plan["evaluation_manifest_sha256"],
        locked_test_ids_sha256=plan["locked_test_ids_sha256"],
        locked_test_example_ids=ids,
    )
    source_receipt = SimpleNamespace(receipt_sha256=plan["source_index_receipt_sha256"])
    monkeypatch.setattr(evaluator_module, "load_evaluation_split_manifest", lambda _: manifest)
    monkeypatch.setattr(
        evaluator_module, "load_locked_source_index_receipt", lambda *_a, **_k: source_receipt
    )

    def fake_load_examples(*_args, **_kwargs):
        assert ledger.is_file(), "the immutable ledger must exist before locked rows load"
        return tuple(SimpleNamespace(example_id=example_id) for example_id in ids)

    monkeypatch.setattr(evaluator_module, "load_locked_examples", fake_load_examples)
    monkeypatch.setattr(evaluator_module, "_format_prompts", lambda *_: ("prompt",))
    monkeypatch.setattr(evaluator_module, "_prompt_token_ids", lambda *_a, **_k: ((1,),))

    def fake_evaluate(*_args, **_kwargs):
        samples = [
            {
                "example_id": example_id,
                "source_index": index,
                "completion": "answer",
                "predicted_answer": "1",
                "correct": index % 2 == 0,
                "response_tokens": 1,
                "finish_reason": "stop",
            }
            for index, example_id in enumerate(ids)
        ]
        return {
            "accuracy": 340 / 679,
            "correct_count": 340,
            "example_count": 679,
            "mean_response_tokens": 1.0,
            "finish_reason_counts": {"stop": 679},
            "truncation_fraction": 0.0,
            "eos_terminated_fraction": 1.0,
        }, samples

    monkeypatch.setattr(evaluator_module, "evaluate_locked_policy", fake_evaluate)
    monkeypatch.setattr(evaluator_module, "_runtime_metadata", lambda: {"runtime": "test"})
    result_path = run_locked_test_evaluation(
        benchmark,
        tmp_path / "manifest.json",
        tmp_path / "indices.json",
        output,
        expected_plan_sha256=plan["plan_sha256"],
        authorize_locked_test_once=True,
    )
    assert result_path == output / "results.json"
    assert ledger.is_file()
    assert (output / "COMPLETE.json").is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert [(row["method"], row["seed"]) for row in result["results"]] == [
        (method, seed) for method, seed, _ in policy_pairs
    ]
    assert len(result["results"]) == 7
    assert all(row["example_count"] == 679 for row in result["results"])
    assert len(list((output / "samples").glob("*.jsonl"))) == 7

    with pytest.raises(FileExistsError, match="already consumed"):
        run_locked_test_evaluation(
            benchmark,
            tmp_path / "manifest.json",
            tmp_path / "indices.json",
            tmp_path / "different-output",
            expected_plan_sha256=plan["plan_sha256"],
            authorize_locked_test_once=True,
        )


def test_receipt_rejects_manifest_order_tampering(tmp_path) -> None:
    _, _, receipt, _ = _manifest_and_receipt(tmp_path)
    with pytest.raises(ValueError, match="locked_source_indices must be sorted"):
        replace(receipt, entries=tuple(reversed(receipt.entries)))
