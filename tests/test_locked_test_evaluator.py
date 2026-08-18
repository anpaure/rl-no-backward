from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import rl_no_backward.locked_test_evaluator as evaluator_module
from rl_no_backward.evaluation_manifest import build_evaluation_split_manifest
from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.locked_test_evaluator import (
    LockedSourceIndexReceipt,
    build_locked_evaluation_plan,
    build_locked_source_index_receipt,
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
    receipt = build_locked_source_index_receipt(
        manifest,
        [example.example_id for example in examples],
        dataset_revision="dataset-commit",
    )
    return examples, manifest, receipt


def test_locked_source_receipt_is_answer_free_bound_and_immutable(tmp_path) -> None:
    _, manifest, receipt = _manifest_and_receipt(tmp_path)
    path = write_locked_source_index_receipt(tmp_path / "locked-indices.json", receipt)
    serialized = path.read_text(encoding="utf-8")
    assert "question" not in serialized
    assert "answer" not in serialized
    assert (
        load_locked_source_index_receipt(path, manifest, dataset_revision="dataset-commit")
        == receipt
    )
    write_locked_source_index_receipt(path, receipt)

    tampered = json.loads(serialized)
    tampered["entries"][0]["source_index"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_locked_source_index_receipt(path, manifest, dataset_revision="dataset-commit")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_locked_source_index_receipt(path, receipt)


def test_locked_loader_selects_only_committed_indices_without_full_iteration(tmp_path) -> None:
    examples, manifest, receipt = _manifest_and_receipt(tmp_path)

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


def _write_frozen_benchmark(tmp_path: Path, manifest, *, seeds=(0,)) -> Path:
    benchmark = tmp_path / "benchmark"
    (benchmark / "checkpoints").mkdir(parents=True)
    (benchmark / "selection").mkdir()
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
        "evaluation_manifest_locked_test_ids_sha256": manifest.locked_test_ids_sha256,
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
        state = {"layer.lora_A.default.weight": torch.tensor([float(value)])}
        checkpoint = benchmark / "checkpoints" / f"gsm8k_{method}_seed{seed}.pt"
        torch.save(
            {
                "method": method,
                "seed": seed,
                "selected_step": value,
                "selection_val_accuracy": value / 10,
                "lora_state": state,
            },
            checkpoint,
        )
        selection = {
            "schema": "rl-no-backward-validation-selection-v1",
            "method": method,
            "seed": seed,
            "selected_step": value,
            "selection_split": "development",
            "selection_metric": "exact_match",
            "tie_breaker": "latest_checkpoint",
            "selection_val_accuracy": value / 10,
            "selected_lora_state_digest": lora_state_digest(state),
            "checkpoint_path": f"/original/{checkpoint.name}",
        }
        (benchmark / "selection" / f"gsm8k_{method}_seed{seed}.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
    return benchmark


def test_plan_freezes_all_selected_checkpoints_without_loading_dataset(
    monkeypatch, tmp_path
) -> None:
    _, manifest, receipt = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
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
    plan = build_locked_evaluation_plan(
        benchmark,
        manifest_path,
        receipt_path,
        expected_locked_count=2,
        enforce_committed_inputs=False,
    )
    assert plan["dataset"]["locked_test_count"] == 2
    assert [row["method"] for row in plan["checkpoints"]] == [
        "base",
        "bp_grpo",
        "fo_npg",
    ]
    assert len(plan["checkpoint_set_sha256"]) == 64
    assert len(plan["plan_sha256"]) == 64
    assert plan["row_loading"] == "Dataset.select(committed_locked_source_indices)"
    assert plan["base_policy_semantics"] == {
        "evaluated_once": True,
        "seed": 0,
        "role": "shared non-updating reference for every trained seed",
    }


def test_three_seed_plan_evaluates_one_shared_base_and_each_trained_policy(
    monkeypatch, tmp_path
) -> None:
    _, manifest, receipt = _manifest_and_receipt(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    receipt_path = write_locked_source_index_receipt(tmp_path / "indices.json", receipt)
    benchmark = _write_frozen_benchmark(tmp_path, manifest, seeds=(0, 1, 2))

    def fake_git(*args, **_):
        values = {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("rev-parse", "--show-toplevel"): str(tmp_path),
        }
        return values[args]

    monkeypatch.setattr(evaluator_module, "_git_output", fake_git)
    plan = build_locked_evaluation_plan(
        benchmark,
        manifest_path,
        receipt_path,
        expected_locked_count=2,
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
    checkpoints = [
        {
            "method": method,
            "seed": 0,
            "selected_step": index,
            "selection_val_accuracy": 0.5,
            "selected_lora_state_digest": character * 64,
            "checkpoint_file_sha256": str(index) * 64,
            "checkpoint_relpath": f"checkpoints/{method}.pt",
        }
        for index, (method, character) in enumerate(
            (("base", "a"), ("bp_grpo", "b"), ("fo_npg", "c")), start=1
        )
    ]
    plan = {
        "plan_sha256": "d" * 64,
        "checkpoint_set_sha256": "e" * 64,
        "evaluation_manifest_sha256": "f" * 64,
        "locked_test_ids_sha256": "1" * 64,
        "source_index_receipt_sha256": "2" * 64,
        "benchmark_metadata_sha256": evaluator_module._file_digest(benchmark / "metadata.json"),
        "seeds": [0],
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
    assert len(result["results"]) == 3
    assert all(row["example_count"] == 679 for row in result["results"])
    assert len(list((output / "samples").glob("*.jsonl"))) == 3

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
    _, manifest, receipt = _manifest_and_receipt(tmp_path)
    reversed_receipt = LockedSourceIndexReceipt(
        dataset_id=receipt.dataset_id,
        dataset_config=receipt.dataset_config,
        dataset_revision=receipt.dataset_revision,
        evaluation_manifest_sha256=receipt.evaluation_manifest_sha256,
        official_test_count=receipt.official_test_count,
        locked_test_count=receipt.locked_test_count,
        entries=tuple(reversed(receipt.entries)),
        receipt_sha256=evaluator_module._json_digest(
            {
                **receipt.payload_without_digest(),
                "entries": [entry.as_dict() for entry in reversed(receipt.entries)],
            }
        ),
    )
    with pytest.raises(ValueError, match="manifest locked order"):
        reversed_receipt.validate_manifest(manifest, dataset_revision="dataset-commit")
