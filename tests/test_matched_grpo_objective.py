from __future__ import annotations

import torch

from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    detached_inference_correction,
    matched_token_grpo_surrogate,
    sampled_hf_policy_kl,
    trl_group_standardized_advantages,
)


def _fixture():
    old_hf = torch.tensor(
        [[[-1.2, -0.7, -2.1], [-0.9, -1.4, -0.2]]], dtype=torch.float64
    )
    q_vllm = old_hf + torch.tensor(
        [[[0.12, -0.04, 0.03], [-0.08, 0.02, 0.01]]], dtype=torch.float64
    )
    mask = torch.tensor([[[True, True, False], [True, True, True]]])
    advantages = torch.tensor([[0.75, -0.5]], dtype=torch.float64)
    features = torch.tensor(
        [[[0.2, -0.1, 0.7], [-0.3, 0.4, 0.1]]], dtype=torch.float64
    )
    return old_hf, q_vllm, mask, advantages, features


def test_group_advantages_match_trl_sample_std() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    actual = trl_group_standardized_advantages(rewards)
    expected = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, correction=1, keepdim=True) + 1.0e-4
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[1], torch.zeros(4))


def test_sampler_mismatch_never_enters_ppo_ratio_at_center() -> None:
    old_hf, q_vllm, mask, advantages, _ = _fixture()
    config = MatchedGRPOObjectiveConfig(
        inference_correction_mode="token_truncate",
        inference_ratio_min=0.1,
        inference_ratio_max=3.0,
    )
    # By construction pi_new_HF == pi_old_HF, hence every PPO ratio is one
    # despite deliberately different vLLM log probabilities.
    ppo_ratio = (old_hf - old_hf).exp()
    assert torch.equal(ppo_ratio, torch.ones_like(ppo_ratio))
    correction = detached_inference_correction(old_hf, q_vllm, mask, config)
    assert not torch.equal(correction[mask], torch.ones_like(correction[mask]))

    actual = matched_token_grpo_surrogate(
        old_hf,
        old_hf,
        q_vllm,
        advantages,
        mask,
        config,
    )
    # Explicit TRL-equivalent fixed-rollout formula: correction is separate
    # from the unit policy ratio and each completion is length-normalized.
    manual_tokens = advantages.unsqueeze(-1) * correction
    expected = (
        manual_tokens.masked_fill(~mask, 0.0).sum(-1)
        / mask.sum(-1).clamp_min(1)
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_autograd_gradient_matches_finite_difference_of_shared_objective() -> None:
    old_hf, q_vllm, mask, advantages, features = _fixture()
    config = MatchedGRPOObjectiveConfig(
        inference_correction_mode="token_truncate",
        inference_ratio_min=0.1,
        inference_ratio_max=3.0,
    )
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return matched_token_grpo_surrogate(
            old_hf + value * features,
            old_hf,
            q_vllm,
            advantages,
            mask,
            config,
        )

    objective(theta).backward()
    assert theta.grad is not None
    radius = 1.0e-5
    with torch.no_grad():
        finite_difference = (
            objective(torch.tensor(radius, dtype=torch.float64))
            - objective(torch.tensor(-radius, dtype=torch.float64))
        ) / (2.0 * radius)
    torch.testing.assert_close(theta.grad, finite_difference, rtol=2.0e-4, atol=2.0e-5)


def test_sequence_mask_correction_is_broadcast_per_completion() -> None:
    old_hf, q_vllm, mask, _, _ = _fixture()
    config = MatchedGRPOObjectiveConfig(
        inference_correction_mode="sequence_mask",
        inference_ratio_max=3.0,
    )
    correction = detached_inference_correction(old_hf, q_vllm, mask, config)
    assert correction.shape == (1, 2, 1)
    expected = ((old_hf - q_vllm) * mask).sum(-1, keepdim=True).exp().float()
    torch.testing.assert_close(correction, expected)


def test_sampled_kl_uses_same_detached_completion_correction() -> None:
    old = torch.zeros(1, 2, 2)
    new = torch.tensor([[[0.2, -0.1], [0.3, 0.4]]])
    mask = torch.ones_like(old, dtype=torch.bool)
    weights = torch.tensor([[[0.0], [2.0]]])
    actual = sampled_hf_policy_kl(new, old, mask, sampling_weights=weights)
    token_kl = torch.expm1(new) - new
    expected = torch.stack(
        [torch.zeros((), dtype=token_kl.dtype), 2.0 * token_kl[0, 1].mean()]
    ).mean()
    torch.testing.assert_close(actual, expected)
