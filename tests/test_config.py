from __future__ import annotations

import pytest

from rl_no_backward.config import RolloutConfig, TaskConfig


def test_task_config_defaults_build_balanced_splits() -> None:
    config = TaskConfig()
    splits = config.build_splits()

    assert (len(splits.train), len(splits.val), len(splits.test)) == (80, 10, 10)
    assert config.build_splits() == splits


def test_task_config_validates_per_target_partition() -> None:
    assert TaskConfig(train_per_target=6, val_per_target=2, test_per_target=2)

    with pytest.raises(ValueError, match="must equal 10"):
        TaskConfig(train_per_target=8, val_per_target=2, test_per_target=1)
    with pytest.raises(TypeError, match="split_seed"):
        TaskConfig(split_seed=True)


def test_rollout_config_requires_a_leave_one_out_group() -> None:
    assert RolloutConfig(group_size=4, max_new_tokens=2, temperature=0.7)

    with pytest.raises(ValueError, match="at least 2"):
        RolloutConfig(group_size=1)
    with pytest.raises(ValueError, match="greater than zero"):
        RolloutConfig(temperature=0.0)
