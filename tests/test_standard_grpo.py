from __future__ import annotations

import json

import pytest

from rl_no_backward.standard_grpo import (
    StandardGRPOConfig,
    _Telemetry,
    build_trl_grpo_arguments,
    gsm8k_exact_reward,
    load_standard_grpo_config,
    main,
    trl_log_to_record,
)


def test_locked_budget_is_exactly_4800_responses() -> None:
    config = StandardGRPOConfig()
    assert config.responses_per_update == 16
    assert config.prompts_per_update == 2
    assert config.response_budget == 4_800
    assert config.backward_call_budget == 2_400


def test_rollout_and_accumulation_budget_must_match() -> None:
    with pytest.raises(ValueError, match="each optimizer step uses one matched rollout"):
        StandardGRPOConfig.from_mapping({"gradient_accumulation_steps": 4})


def test_exact_reward_accepts_trl_conversational_completions() -> None:
    rewards = gsm8k_exact_reward(
        [
            [{"role": "assistant", "content": "Work. Final answer: 12"}],
            [{"role": "assistant", "content": "Final answer: 7"}],
        ],
        ["12", "8"],
    )
    assert rewards == [1.0, 0.0]


def test_trl_log_conversion_uses_exact_environment_accounting(monkeypatch) -> None:
    config = StandardGRPOConfig()
    telemetry = _Telemetry(started_at=10.0)
    monkeypatch.setattr("rl_no_backward.standard_grpo.time.perf_counter", lambda: 12.5)
    record = trl_log_to_record(
        {
            "reward": 0.375,
            "completions/mean_length": 100.5,
            "frac_reward_zero_std": 0.5,
            "step_time": 1.25,
        },
        step=3,
        seed=0,
        config=config,
        telemetry=telemetry,
        peak_gpu_memory_bytes=123,
    )
    assert record is not None
    assert record["environment_samples"] == 48
    assert record["generated_tokens"] == round(100.5 * 16 * 3)
    assert record["backward_calls"] == 24
    assert record["rollout_exact_reward"] == 0.375
    assert record["importance_sampling_level"] == "token"
    assert (
        trl_log_to_record(
            {"reward": 0.5},
            step=3,
            seed=0,
            config=config,
            telemetry=telemetry,
        )
        is None
    )


def test_config_round_trip_and_print_plan(tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "steps: 300\n"
        "prompts_per_update: 2\n"
        "group_size: 8\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 8\n"
        "lora:\n"
        "  rank: 8\n"
        "  alpha: 16\n"
        "  target_modules: [q_proj, v_proj]\n"
        f"  layer_indices: {list(range(28))}\n",
        encoding="utf-8",
    )
    config = load_standard_grpo_config(config_path)
    assert config.lora.rank == 8
    main(["--config", str(config_path), "--print-plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["total_responses"] == 4_800
    assert plan["expected_trainable_lora_parameters"] == 1_089_536


def test_installed_trl_constructor_smoke(tmp_path) -> None:
    pytest.importorskip("trl")
    config = StandardGRPOConfig(steps=5, eval_interval=5)
    arguments = build_trl_grpo_arguments(config, tmp_path, seed=0)
    assert arguments.max_steps == 5
    assert arguments.warmup_steps == 0
    assert arguments.generation_batch_size == 16
    assert arguments.loss_type == "grpo"
