from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import rl_no_backward.matched_lora_runner as runner_module
from rl_no_backward.evaluation_manifest import (
    EvaluationSplitManifest,
    build_evaluation_split_manifest,
    write_evaluation_split_manifest,
)
from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.matched_lora_runner import (
    MatchedLoRAExperimentConfig,
    _init_wandb,
    _load_data,
    _old_policy_rescore_accounting,
    _prepare_empty_output_directory,
    _require_output_outside_worktree,
    _rollout_finish_telemetry,
    _validate_isolated_trial_outputs,
    build_prompt_schedule,
    load_matched_config,
    main,
)


def _examples(count: int):
    return tuple(
        GSM8KExample(
            question=f"Question {index}?",
            answer=f"work\n#### {index}",
            split="train",
            source_index=index,
        )
        for index in range(count)
    )


def test_locked_overfit_plan_and_schedule_are_deterministic(capsys) -> None:
    config = load_matched_config("configs/gsm8k_matched_lora_overfit_gate.yaml")
    assert config.responses_per_step == 64
    assert config.response_budget == 1_600
    assert config.run_test_evaluation is False
    examples = _examples(8)
    first = build_prompt_schedule(examples, config, seed=0)
    second = build_prompt_schedule(examples, config, seed=0)
    assert first == second
    assert len(first) == 25
    assert all(row["example_ids"] == first[0]["example_ids"] for row in first)
    assert len({row["rollout_seed"] for row in first}) == 25

    main(
        [
            "--config",
            "configs/gsm8k_matched_lora_overfit_gate.yaml",
            "--print-plan",
        ]
    )
    output = capsys.readouterr().out
    assert '"responses_per_step": 64' in output
    assert '"locked_test_evaluation": false' in output


def test_config_rejects_locked_test_and_unmatched_method_sets() -> None:
    with pytest.raises(ValueError, match="locked-test"):
        MatchedLoRAExperimentConfig(run_test_evaluation=True, test_size=1)
    with pytest.raises(ValueError, match="headline gate"):
        MatchedLoRAExperimentConfig(methods=["bp_grpo", "fo_npg"])


def test_shuffled_schedule_is_method_independent() -> None:
    config = MatchedLoRAExperimentConfig(
        train_size=16,
        batch_size=8,
        steps=6,
        schedule_mode="shuffled_cycles",
    )
    schedule = build_prompt_schedule(_examples(16), config, seed=4)
    assert all(len(row["example_ids"]) == 8 for row in schedule)
    assert schedule == build_prompt_schedule(_examples(16), config, seed=4)
    assert schedule != build_prompt_schedule(_examples(16), config, seed=5)
    assert schedule[0]["rollout_seed"] == 24_001


def test_old_hf_rescore_and_finish_telemetry_are_exact() -> None:
    assert _old_policy_rescore_accounting(64, 1_024, 16) == {
        "forward_calls": 4,
        "teacher_forced_examples": 64,
        "scored_tokens": 1_024,
    }
    counts, truncation, eos = _rollout_finish_telemetry(
        (("stop", "length"), ("eos", "stop"))
    )
    assert counts == {"eos": 1, "length": 1, "stop": 2}
    assert truncation == pytest.approx(0.25)
    assert eos == pytest.approx(0.75)


def test_load_data_consumes_and_validates_full_manifest(monkeypatch, tmp_path) -> None:
    train = _examples(8)
    official = tuple(
        GSM8KExample(
            question=f"Official question {index}?",
            answer=f"work\n#### {index}",
            split="test",
            source_index=index,
        )
        for index in range(388)
    )
    first = tmp_path / "prior-a.json"
    second = tmp_path / "prior-b.json"
    first.write_text(
        json.dumps({"test_example_ids": [example.example_id for example in official[:128]]}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"test_example_ids": [example.example_id for example in official[128:384]]}),
        encoding="utf-8",
    )
    manifest = build_evaluation_split_manifest(
        [example.example_id for example in official],
        [first, second],
        dev_size=2,
        locked_test_size=2,
        seed=7,
        namespace="test-matched-loader",
    )
    manifest_path = write_evaluation_split_manifest(tmp_path / "manifest.json", manifest)
    exclusions_path = tmp_path / "exclusions.json"
    exclusions_path.write_text(
        json.dumps({"test_example_ids": list(manifest.excluded_test_example_ids)}),
        encoding="utf-8",
    )
    by_id = {example.example_id: index for index, example in enumerate(official)}
    receipt_payload = {
        "schema": "rl-no-backward-gsm8k-dev-source-index-v1",
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": MatchedLoRAExperimentConfig().dataset_revision,
        "evaluation_manifest_sha256": manifest.manifest_sha256,
        "official_test_count": len(official),
        "dev_count": manifest.dev_count,
        "entries": [
            {"source_index": by_id[example_id], "example_id": example_id}
            for example_id in manifest.dev_example_ids
        ],
    }
    receipt_payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt_path = tmp_path / "dev-source-indices.json"
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    class SelectOnlyDataset:
        def __init__(self, rows):
            self.rows = rows
            self.selected_indices = None

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            raise AssertionError("the complete official-test dataset must not be materialized")

        def select(self, indices):
            self.selected_indices = list(indices)
            return [self.rows[index] for index in indices]

    raw_rows = [
        {"question": example.question, "answer": example.answer} for example in official
    ]
    dataset = SelectOnlyDataset(raw_rows)
    monkeypatch.setattr(
        runner_module,
        "load_gsm8k_split",
        lambda split, **_: train,
    )
    monkeypatch.setattr(runner_module, "_load_pinned_gsm8k_test_dataset", lambda _: dataset)
    config = MatchedLoRAExperimentConfig(
        dev_size=2,
        evaluation_manifest=str(manifest_path),
        touched_test_exclusions=str(exclusions_path),
        dev_source_index_receipt=str(receipt_path),
    )
    _, dev, loaded = _load_data(config)
    assert isinstance(loaded, EvaluationSplitManifest)
    assert loaded.manifest_sha256 == manifest.manifest_sha256
    assert tuple(example.example_id for example in dev) == manifest.dev_example_ids
    assert not ({example.example_id for example in dev} & set(manifest.locked_test_example_ids))
    assert dataset.selected_indices == [entry["source_index"] for entry in receipt_payload["entries"]]

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["entries"][0]["source_index"] += 1
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt digest"):
        _load_data(config)


def test_wandb_root_is_not_double_nested(monkeypatch, tmp_path) -> None:
    captured = {}
    fake_wandb = ModuleType("wandb")

    def fake_init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    fake_wandb.init = fake_init
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = MatchedLoRAExperimentConfig()
    _init_wandb(config, tmp_path, method="bp_grpo", seed=0)
    assert captured["dir"] == str(tmp_path.resolve())
    assert os.environ["WANDB_DIR"] == str(tmp_path.resolve())
    assert not tmp_path.joinpath("wandb", "wandb").exists()


def test_isolated_output_validator_requires_distinct_children_and_exact_counters(tmp_path) -> None:
    config = MatchedLoRAExperimentConfig(steps=1, eval_interval=1)
    (tmp_path / "trial_metadata").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "learning_gate").mkdir()
    receipt = {
        "adapter_path_transient": True,
        "durable_hash_fields": ["state_digest", "adapter_model_sha256"],
    }
    for method, process_id in (("base", 91_001), ("bp_grpo", 91_002), ("fo_npg", 91_003)):
        metadata = {
            "method": method,
            "seed": 0,
            "process_id": process_id,
            "process_isolation": "fresh_python_process_per_method_seed",
            "frozen_base_parameter_digest_before": "frozen",
            "frozen_base_parameter_digest_after": "frozen",
            "vllm_same_id_reload_gate": {"passed": True},
            "vllm_lora_reload_receipts": [receipt],
            "fo_reverse_mode_modules_called": False if method == "fo_npg" else None,
            "fo_backprop_module_imported": False if method == "fo_npg" else None,
        }
        (tmp_path / "trial_metadata" / f"{method}_seed0.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        records = []
        if method != "base":
            counters = {
                "environment_samples": 64,
                "generated_tokens": 512,
                "scored_tokens": 1_024,
                "forward_calls": 12,
                "backward_calls": 8 if method == "bp_grpo" else 0,
                "teacher_forced_examples": 128,
            }
            records.append(
                {
                    "kind": "train_step",
                    "step": 1,
                    **counters,
                    **{f"cumulative_{key}": value for key, value in counters.items()},
                    "behavior_policy_version": f"{method}/rollout/1",
                    "behavior_policy_digest": "center",
                    "lora_digest_before": "center",
                    "old_policy_rescore_forward_calls": 4,
                    "old_policy_rescore_teacher_forced_examples": 64,
                    "rollout_finish_reason_counts": {"stop": 64},
                    "rollout_token_digest": "a" * 64,
                    "behavior_logprob_digest": "b" * 64,
                }
            )
        (tmp_path / "raw" / f"gsm8k_{method}_seed0.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (tmp_path / "learning_gate" / f"gsm8k_{method}_seed0.json").write_text(
            json.dumps({"passed": True}),
            encoding="utf-8",
        )

    metadata, _ = _validate_isolated_trial_outputs(tmp_path, config)
    assert set(metadata) == {"base_seed0", "bp_grpo_seed0", "fo_npg_seed0"}

    fo_path = tmp_path / "trial_metadata" / "fo_npg_seed0.json"
    fo_metadata = json.loads(fo_path.read_text(encoding="utf-8"))
    fo_metadata["process_id"] = 91_002
    fo_path.write_text(json.dumps(fo_metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distinct fresh child"):
        _validate_isolated_trial_outputs(tmp_path, config)


def test_output_directory_must_be_empty(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _prepare_empty_output_directory(empty) == empty
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    occupied.joinpath("partial.jsonl").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        _prepare_empty_output_directory(occupied)


def test_output_directory_must_be_outside_worktree(tmp_path) -> None:
    worktree = tmp_path / "checkout"
    worktree.mkdir()
    with pytest.raises(ValueError, match="outside the Git worktree"):
        _require_output_outside_worktree(worktree / "artifacts" / "gate", worktree)
    _require_output_outside_worktree(tmp_path / "external-artifacts", worktree)


def test_runner_import_does_not_import_reverse_mode_module() -> None:
    imports = "import sys; import rl_no_backward.matched_lora_runner"
    assertion = "assert 'rl_no_backward.matched_lora_backprop' not in sys.modules"
    code = f"{imports}; {assertion}"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
