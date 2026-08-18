from __future__ import annotations

import pytest

from rl_no_backward.gsm8k import GSM8KExample
from rl_no_backward.gsm8k_experiment import (
    GSM8KExperimentConfig,
    shaped_gsm8k_reward,
)


def _example(answer: str = "10") -> GSM8KExample:
    return GSM8KExample(
        question="What is five plus five?",
        answer=f"Five plus five is ten.\n#### {answer}",
        split="test",
        source_index=0,
    )


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

    with pytest.raises(ValueError, match="unknown methods"):
        GSM8KExperimentConfig(methods=["not_an_optimizer"]).validate()
    with pytest.raises(ValueError, match="numeric_shaping_weight"):
        GSM8KExperimentConfig(numeric_shaping_weight=1.0).validate()
    with pytest.raises(ValueError, match="group_size"):
        GSM8KExperimentConfig(group_size=1).validate()
