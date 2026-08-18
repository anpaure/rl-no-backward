from __future__ import annotations

import pytest
import torch

pytest.importorskip("trl", reason="TRL 1.10 oracle runs in the pinned H100 environment")

from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    matched_token_grpo_surrogate,
    trl_group_standardized_advantages,
)
from rl_no_backward.trl_grpo_oracle import trl_110_fixed_rollout_loss


def test_matched_loss_gradient_and_update_equal_upstream_trl_110() -> None:
    config = MatchedGRPOObjectiveConfig(
        clip_epsilon=0.2,
        advantage_epsilon=1.0e-4,
        inference_correction_mode="sequence_mask",
        inference_ratio_min=None,
        inference_ratio_max=3.0,
    )
    rewards = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5]], dtype=torch.float64)
    advantages = trl_group_standardized_advantages(rewards, config.advantage_epsilon).double()
    old = torch.tensor(
        [
            [[-1.0, -1.3, -0.8, -0.5], [-0.7, -1.1, -1.4, -0.4], [-1.2, -0.9, -0.6, -1.5]],
            [[-0.8, -1.2, -0.7, -0.9], [-1.4, -0.6, -1.0, -0.5], [-0.9, -0.7, -1.3, -0.8]],
        ],
        dtype=torch.float64,
    )
    sampler = old - torch.tensor(
        [
            [[0.01, -0.02, 0.01, 0.00], [0.03, 0.02, -0.01, 0.00], [0.02, 0.01, 0.00, -0.01]],
            [[0.01, 0.01, -0.02, 0.00], [0.04, -0.02, 0.01, 0.00], [0.02, -0.01, 0.03, 0.00]],
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
        ],
        dtype=torch.bool,
    )
    feature = torch.linspace(-0.7, 0.9, old.numel(), dtype=torch.float64).reshape_as(old)
    native_parameter = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    trl_parameter = native_parameter.detach().clone().requires_grad_(True)
    native_new = old + native_parameter * feature
    trl_new = old + trl_parameter * feature

    native_loss = -matched_token_grpo_surrogate(
        native_new,
        old,
        sampler,
        advantages,
        mask,
        config,
    )
    trl_loss = trl_110_fixed_rollout_loss(
        trl_new,
        old,
        sampler,
        advantages,
        mask,
        config,
    )
    native_gradient = torch.autograd.grad(native_loss, native_parameter)[0]
    trl_gradient = torch.autograd.grad(trl_loss, trl_parameter)[0]
    torch.testing.assert_close(native_loss, trl_loss, rtol=1.0e-10, atol=1.0e-12)
    torch.testing.assert_close(native_gradient, trl_gradient, rtol=1.0e-10, atol=1.0e-12)

    learning_rate = 3.0e-4
    native_update = native_parameter.detach() - learning_rate * native_gradient
    trl_update = trl_parameter.detach() - learning_rate * trl_gradient
    torch.testing.assert_close(native_update, trl_update, rtol=1.0e-10, atol=1.0e-12)
