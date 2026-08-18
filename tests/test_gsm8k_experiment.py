from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import rl_no_backward.gsm8k_experiment as experiment_module
from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.gsm8k_experiment import (
    GSM8KExperimentConfig,
    _excluded_test_ids,
    _generate_vllm_sequence_rollouts,
    _model_runtime_metadata,
    _peak_gpu_memory_metrics,
    _rollout_truncation_fraction,
    evaluate_gsm8k,
    run_gsm8k_trial,
    shaped_gsm8k_reward,
)
from rl_no_backward.model import ModelBundle, parameter_vector
from rl_no_backward.sequence_optimizers import ForwardSequenceStepResult
from rl_no_backward.vllm_rollout import VLLMGreedyGeneration, VLLMGroupedGeneration


def _example(answer: str = "10") -> GSM8KExample:
    return GSM8KExample(
        question="What is five plus five?",
        answer=f"Five plus five is ten.\n#### {answer}",
        split="test",
        source_index=0,
    )


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, **_kwargs):
        rows = [[1, 3 + index] for index, _ in enumerate(texts)]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones(len(rows), 2, dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs):
        values = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, torch.Tensor)
            else list(token_ids)
        )
        if 5 in values:
            return "Final answer: 10"
        if 6 in values:
            return "Final answer: 9"
        return "No numeric answer"

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"


class _NoGenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Parameter(torch.zeros(1))

    def generate(self, **_kwargs):
        raise AssertionError("HF generation must not run for the vLLM backend")


def test_shaped_reward_keeps_exact_match_dominant() -> None:
    example = _example()

    assert shaped_gsm8k_reward("Final answer: 10", example, 0.1) == 1.0
    near = shaped_gsm8k_reward("Final answer: 9", example, 0.1)
    far = shaped_gsm8k_reward("Final answer: 1000", example, 0.1)
    assert 0.0 < far < near < 0.1
    assert shaped_gsm8k_reward("No numeric answer", example, 0.1) == 0.0
    assert shaped_gsm8k_reward("Final answer: 9", example, 0.0) == 0.0


def test_gsm8k_config_validates_benchmark_invariants() -> None:
    config = GSM8KExperimentConfig.from_mapping(
        {"methods": ["base", "focus_npg"], "seeds": [7], "steps": 2}
    )
    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.adapter_rank**2 * config.adapter_layers == 256
    assert config.attention_implementation == "eager"
    assert not config.compile_model_forward
    assert config.compile_model_forward_mode == "default"
    assert config.vllm_allow_insecure_serialization is False
    assert config.vllm_flash_attn_version == 2
    assert config.vllm_logprob_p99_abs_tolerance == pytest.approx(0.2)

    with pytest.raises(ValueError, match="unknown methods"):
        GSM8KExperimentConfig(methods=["not_an_optimizer"]).validate()
    with pytest.raises(ValueError, match="numeric_shaping_weight"):
        GSM8KExperimentConfig(numeric_shaping_weight=1.0).validate()
    with pytest.raises(ValueError, match="group_size"):
        GSM8KExperimentConfig(group_size=1).validate()
    with pytest.raises(ValueError, match="model_revision"):
        GSM8KExperimentConfig(model_revision="").validate()
    with pytest.raises(ValueError, match="attention_implementation"):
        GSM8KExperimentConfig(attention_implementation="unknown").validate()
    with pytest.raises(TypeError, match="compile_model_forward"):
        GSM8KExperimentConfig(compile_model_forward=1).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compile_model_forward_mode"):
        GSM8KExperimentConfig(compile_model_forward_mode="unsafe-custom").validate()
    with pytest.raises(TypeError, match="use_frozen_prefix_scoring"):
        GSM8KExperimentConfig(use_frozen_prefix_scoring="yes").validate()  # type: ignore[arg-type]


def test_gsm8k_config_accepts_opt_in_hf_runtime_controls() -> None:
    config = GSM8KExperimentConfig.from_mapping(
        {
            "attention_implementation": "sdpa",
            "compile_model_forward": True,
            "compile_model_forward_mode": "reduce-overhead",
        }
    )

    assert config.attention_implementation == "sdpa"
    assert config.compile_model_forward is True
    assert config.compile_model_forward_mode == "reduce-overhead"


def test_gsm8k_config_validates_vllm_backend_controls() -> None:
    config = GSM8KExperimentConfig(
        rollout_backend="vllm",
        device="cuda",
        vllm_kv_cache_memory_bytes=1234,
        vllm_enforce_eager=True,
        vllm_flash_attn_version=2,
        vllm_allow_insecure_serialization=True,
    )
    config.validate()

    with pytest.raises(ValueError, match="rollout_backend"):
        GSM8KExperimentConfig(rollout_backend="other").validate()
    with pytest.raises(ValueError, match="CUDA"):
        GSM8KExperimentConfig(rollout_backend="vllm", device="cpu").validate()
    with pytest.raises(ValueError, match="kv_cache"):
        GSM8KExperimentConfig(vllm_kv_cache_memory_bytes=0).validate()
    with pytest.raises(ValueError, match="explicit.*trusted local callable IPC"):
        GSM8KExperimentConfig(rollout_backend="vllm", device="cuda").validate()
    with pytest.raises(TypeError, match="vllm_allow_insecure_serialization"):
        GSM8KExperimentConfig(
            vllm_allow_insecure_serialization=1  # type: ignore[arg-type]
        ).validate()
    with pytest.raises(ValueError, match="vllm_flash_attn_version"):
        GSM8KExperimentConfig(vllm_flash_attn_version=1).validate()
    with pytest.raises(ValueError, match="mean <= p99 <= maximum"):
        GSM8KExperimentConfig(
            vllm_logprob_mean_abs_tolerance=0.2,
            vllm_logprob_p99_abs_tolerance=0.1,
            vllm_logprob_max_abs_tolerance=0.5,
        ).validate()


def test_vllm_rollout_keeps_behavior_logprobs_and_reports_hf_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = ModelBundle(
        model=_NoGenerateModel(),
        tokenizer=_TinyTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )
    behavior_log_probs = torch.tensor([[[-0.4, -0.7], [-0.8, -0.7]]])

    class Policy:
        policy_version = "step-1"
        state_digest = "digest"

        def generate(self, prompt_token_ids, **_kwargs):
            assert prompt_token_ids == ((1, 3),)
            return VLLMGroupedGeneration(
                prompt_token_ids=prompt_token_ids,
                response_input_ids=torch.tensor([[[5, 2], [6, 2]]]),
                response_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                old_token_log_probs=behavior_log_probs.clone(),
                finish_reasons=(("eos", "eos"),),
                policy_version=self.policy_version,
            )

    monkeypatch.setattr(
        experiment_module,
        "teacher_forced_token_log_probs",
        lambda _bundle, rollout, **_kwargs: rollout.old_token_log_probs + 0.001,
    )
    config = GSM8KExperimentConfig(
        rollout_backend="vllm",
        device="cuda",
        batch_size=1,
        group_size=2,
        scoring_micro_batch_size=2,
        vllm_logprob_mean_abs_tolerance=0.01,
        vllm_logprob_p99_abs_tolerance=0.02,
        vllm_logprob_max_abs_tolerance=0.05,
    )

    rollout, diagnostics = _generate_vllm_sequence_rollouts(
        bundle,
        ["prompt"],
        [_example()],
        config,
        Policy(),  # type: ignore[arg-type]
        seed=123,
    )

    assert torch.equal(rollout.old_token_log_probs, behavior_log_probs)
    assert rollout.completions == (("Final answer: 10", "Final answer: 9"),)
    assert rollout.rewards.tolist() == [[1.0, pytest.approx(0.09, abs=0.02)]]
    assert diagnostics["behavior_hf_logprob_mean_abs_delta"] == pytest.approx(0.001, abs=1e-6)
    assert diagnostics["behavior_hf_logprob_max_abs_delta"] == pytest.approx(0.001, abs=1e-6)


def test_vllm_rollout_hard_fails_backend_equivalence_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = ModelBundle(
        model=_NoGenerateModel(),
        tokenizer=_TinyTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )

    class Policy:
        def generate(self, prompt_token_ids, **_kwargs):
            return VLLMGroupedGeneration(
                prompt_token_ids=prompt_token_ids,
                response_input_ids=torch.tensor([[[5, 2], [6, 2]]]),
                response_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                old_token_log_probs=torch.full((1, 2, 2), -1.0),
                finish_reasons=(("eos", "eos"),),
                policy_version="bad-step",
            )

        policy_version = "bad-step"
        state_digest = "bad-digest"

    monkeypatch.setattr(
        experiment_module,
        "teacher_forced_token_log_probs",
        lambda _bundle, rollout, **_kwargs: rollout.old_token_log_probs + 0.2,
    )
    config = GSM8KExperimentConfig(
        rollout_backend="vllm",
        device="cuda",
        batch_size=1,
        group_size=2,
        vllm_logprob_mean_abs_tolerance=0.01,
        vllm_logprob_p99_abs_tolerance=0.02,
        vllm_logprob_max_abs_tolerance=0.05,
    )

    with pytest.raises(
        RuntimeError,
        match=r"equivalence gate failed: .*p99_abs=.*HF attention=.*flash_attn_version=2",
    ):
        _generate_vllm_sequence_rollouts(
            bundle,
            ["prompt"],
            [_example()],
            config,
            Policy(),  # type: ignore[arg-type]
            seed=123,
        )


def test_vllm_rollout_rejects_stale_behavior_policy_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = ModelBundle(
        model=_NoGenerateModel(),
        tokenizer=_TinyTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )

    class Policy:
        policy_version = "step-2"
        state_digest = "digest-step-2"

        def generate(self, prompt_token_ids, **_kwargs):
            return VLLMGroupedGeneration(
                prompt_token_ids=prompt_token_ids,
                response_input_ids=torch.tensor([[[5, 2], [6, 2]]]),
                response_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                old_token_log_probs=torch.full((1, 2, 2), -1.0),
                finish_reasons=(("stop", "stop"),),
                policy_version="stale-step-1",
            )

    with pytest.raises(RuntimeError, match="wrong behavior-policy version"):
        _generate_vllm_sequence_rollouts(
            bundle,
            ["prompt"],
            [_example()],
            GSM8KExperimentConfig(
                rollout_backend="vllm",
                device="cuda",
                batch_size=1,
                group_size=2,
            ),
            Policy(),  # type: ignore[arg-type]
            seed=123,
        )


def test_vllm_greedy_evaluation_never_calls_hf_generate() -> None:
    bundle = ModelBundle(
        model=_NoGenerateModel(),
        tokenizer=_TinyTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )

    class Policy:
        policy_version = "eval-step"
        state_digest = "eval-digest"

        def generate_greedy(self, prompt_token_ids, **_kwargs):
            return VLLMGreedyGeneration(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=((5, 2),),
                finish_reasons=("eos",),
                policy_version=self.policy_version,
            )

    metrics, rows = evaluate_gsm8k(
        bundle,
        [_example()],
        GSM8KExperimentConfig(
            rollout_backend="vllm",
            device="cuda",
            eval_batch_size=1,
        ),
        rollout_policy=Policy(),  # type: ignore[arg-type]
    )

    assert metrics["val_accuracy"] == 1.0
    assert metrics["val_truncation_fraction"] == 0.0
    assert rows[0]["completion"] == "Final answer: 10"
    assert rows[0]["response_tokens"] == 2


def test_model_runtime_metadata_records_requested_and_effective_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self._rl_no_backward_forward_compiled = True
            self._rl_no_backward_forward_compile_mode = "reduce-overhead"

    model = RuntimeModel()
    bundle = ModelBundle(
        model=model,
        tokenizer=SimpleNamespace(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )
    config = GSM8KExperimentConfig(
        attention_implementation="flash_attention_2",
        compile_model_forward=True,
        compile_model_forward_mode="reduce-overhead",
    )
    monkeypatch.setattr(experiment_module, "installed_flash_attn_version", lambda: "2.8.3")

    metadata = _model_runtime_metadata(bundle, config)

    assert metadata == {
        "requested_attention_implementation": "flash_attention_2",
        "resolved_attention_implementation": "sdpa",
        "flash_attn_version": "2.8.3",
        "compile_model_forward": True,
        "compile_model_forward_mode": "reduce-overhead",
        "model_forward_compiled": True,
    }


def test_gsm8k_config_validates_opt_in_fast_paths() -> None:
    config = GSM8KExperimentConfig.from_mapping(
        {
            "methods": ["bp_grpo", "fo_pg"],
            "backprop": {"use_streaming_backward": True},
            "forward": {
                "use_fused_probes": True,
                "fused_probe_directions_per_forward": 4,
                "fused_probe_examples_per_forward": 8,
            },
        }
    )
    assert config.backprop["use_streaming_backward"] is True
    assert config.forward["use_fused_probes"] is True

    with pytest.raises(TypeError, match="use_streaming_backward"):
        GSM8KExperimentConfig(backprop={"use_streaming_backward": "yes"}).validate()
    with pytest.raises(ValueError, match="fused_probe_directions_per_forward"):
        GSM8KExperimentConfig(forward={"fused_probe_directions_per_forward": 0}).validate()


def test_excluded_test_ids_are_loaded_from_pilot_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"test_example_ids": ["seen-a", "seen-b", "seen-a"]}),
        encoding="utf-8",
    )

    assert _excluded_test_ids(str(metadata)) == {"seen-a", "seen-b"}

    metadata.write_text(json.dumps({"test_example_ids": [1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="string list"):
        _excluded_test_ids(str(metadata))


def test_rollout_truncation_uses_terminal_eos_token() -> None:
    rollout = SimpleNamespace(
        response_lengths=torch.tensor([[3, 3]]),
        response_input_ids=torch.tensor([[[7, 8, 2], [7, 8, 9]]]),
    )
    tokenizer = SimpleNamespace(eos_token_id=2)

    assert _rollout_truncation_fraction(rollout, tokenizer) == pytest.approx(0.5)


def test_trial_records_synchronised_phase_and_memory_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Parameter(torch.zeros(1))

    tokenizer = SimpleNamespace(
        eos_token_id=2,
        pad_token_id=0,
        apply_chat_template=lambda *_args, **_kwargs: "prompt",
    )
    model = TinyModel()
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )
    rollout = SimpleNamespace(
        completions=(("Final answer: 10", "Final answer: 9"),),
        environment_samples=2,
        valid_response_tokens=4,
        rewards=torch.tensor([[1.0, 0.0]]),
        response_lengths=torch.tensor([[2, 2]]),
        response_input_ids=torch.tensor([[[3, 2], [4, 2]]]),
    )
    result = ForwardSequenceStepResult(
        accepted=True,
        reward_mean=0.5,
        zero_advantage_fraction=0.0,
        empirical_kl=0.001,
        surrogate_improvement=0.01,
        step_norm=0.02,
        projected_gradient_norm=0.1,
        fisher_condition=1.0,
        line_search_trials=1,
        policy_evaluations=3,
        forward_calls=3,
        backward_calls=0,
        environment_samples=2,
        teacher_forced_examples=6,
        scored_tokens=12,
        derivative_variance=0.2,
        full_prefix_calls=3,
        suffix_calls=3,
    )
    monkeypatch.setattr(experiment_module, "generate_sequence_rollouts", lambda *_a, **_k: rollout)
    monkeypatch.setattr(experiment_module, "_exact_rollout_metrics", lambda *_a: (0.5, 0.0))
    monkeypatch.setattr(experiment_module, "forward_sequence_step", lambda *_a, **_k: result)
    monkeypatch.setattr(
        experiment_module,
        "evaluate_gsm8k",
        lambda *_a, **_k: ({"val_accuracy": 0.25}, []),
    )
    config = GSM8KExperimentConfig(
        methods=["fo_pg"],
        seeds=[0],
        steps=1,
        batch_size=1,
        group_size=2,
        scoring_micro_batch_size=2,
        eval_interval=1,
        wandb_mode="disabled",
        forward={"directions": 1},
    )

    raw_path = run_gsm8k_trial(
        bundle,
        [_example()],
        [_example()],
        [],
        config,
        "fo_pg",
        0,
        parameter_vector(bundle).clone(),
        {"val_accuracy": 0.2},
        [],
        tmp_path,
        initial_evaluation_seconds=0.123,
    )

    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    initial, train, evaluation = records
    assert initial["evaluation_seconds"] == pytest.approx(0.123)
    assert train["rollout_and_old_score_seconds"] >= 0
    assert train["optimizer_seconds"] >= 0
    assert train["full_prefix_calls"] == train["forward_calls"]
    assert train["suffix_calls"] == train["forward_calls"]
    assert evaluation["evaluation_seconds"] >= 0
    for record in (train, evaluation):
        assert record["peak_gpu_memory_bytes"] == record["peak_gpu_memory_allocated_bytes"]
        assert record["peak_gpu_memory_reserved_bytes"] == 0
    assert _peak_gpu_memory_metrics(torch.device("cpu")) == {
        "peak_gpu_memory_bytes": 0,
        "peak_gpu_memory_allocated_bytes": 0,
        "peak_gpu_memory_reserved_bytes": 0,
    }


def test_vllm_trial_records_behavior_and_next_policy_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Parameter(torch.zeros(1))

    bundle = ModelBundle(
        model=TinyModel(),
        tokenizer=SimpleNamespace(
            eos_token_id=2,
            pad_token_id=0,
            apply_chat_template=lambda *_args, **_kwargs: "prompt",
        ),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="tiny",
    )
    rollout = SimpleNamespace(
        completions=(("Final answer: 10", "Final answer: 9"),),
        environment_samples=2,
        valid_response_tokens=4,
        rewards=torch.tensor([[1.0, 0.0]]),
        response_lengths=torch.tensor([[2, 2]]),
        response_input_ids=torch.tensor([[[3, 2], [4, 2]]]),
    )
    result = ForwardSequenceStepResult(
        accepted=True,
        reward_mean=0.5,
        zero_advantage_fraction=0.0,
        empirical_kl=0.001,
        surrogate_improvement=0.01,
        step_norm=0.02,
        projected_gradient_norm=0.1,
        fisher_condition=1.0,
        line_search_trials=1,
        policy_evaluations=3,
        forward_calls=3,
        backward_calls=0,
        environment_samples=2,
        teacher_forced_examples=6,
        scored_tokens=12,
        derivative_variance=0.2,
        full_prefix_calls=3,
        suffix_calls=3,
    )

    class Policy:
        policy_version = "benchmark-initial"
        state_digest = "digest::benchmark-initial"

    def fake_sync(_bundle, policy, *, version):
        policy.policy_version = version
        policy.state_digest = f"digest::{version}"

    zero_diagnostics = {
        "behavior_hf_logprob_mean_abs_delta": 0.0,
        "behavior_hf_logprob_p99_abs_delta": 0.0,
        "behavior_hf_logprob_max_abs_delta": 0.0,
        "behavior_hf_logprob_mean_signed_delta": 0.0,
        "behavior_hf_max_importance_ratio_deviation": 0.0,
    }
    monkeypatch.setattr(experiment_module, "_sync_vllm_policy", fake_sync)
    monkeypatch.setattr(
        experiment_module,
        "_generate_vllm_sequence_rollouts",
        lambda *_a, **_k: (rollout, zero_diagnostics),
    )
    monkeypatch.setattr(experiment_module, "_exact_rollout_metrics", lambda *_a: (0.5, 0.0))
    monkeypatch.setattr(experiment_module, "forward_sequence_step", lambda *_a, **_k: result)
    monkeypatch.setattr(
        experiment_module,
        "evaluate_gsm8k",
        lambda *_a, **_k: ({"val_accuracy": 0.25}, []),
    )
    config = GSM8KExperimentConfig(
        rollout_backend="vllm",
        device="cuda",
        methods=["fo_pg"],
        seeds=[0],
        steps=1,
        batch_size=1,
        group_size=2,
        scoring_micro_batch_size=2,
        eval_interval=1,
        wandb_mode="disabled",
        forward={"directions": 1},
    )

    raw_path = run_gsm8k_trial(
        bundle,
        [_example()],
        [_example()],
        [],
        config,
        "fo_pg",
        0,
        parameter_vector(bundle).clone(),
        {"val_accuracy": 0.2},
        [],
        tmp_path,
        rollout_policy=Policy(),  # type: ignore[arg-type]
    )

    initial, train, evaluation = [json.loads(line) for line in raw_path.read_text().splitlines()]
    reset_version = "method=fo_pg/seed=0/reset"
    next_version = "method=fo_pg/seed=0/step=1"
    assert initial["rollout_policy_version"] == reset_version
    assert train["behavior_policy_version"] == reset_version
    assert train["behavior_policy_state_digest"] == f"digest::{reset_version}"
    assert train["next_policy_version"] == next_version
    assert train["next_policy_state_digest"] == f"digest::{next_version}"
    assert "rollout_policy_version" not in train
    assert evaluation["rollout_policy_version"] == next_version
