from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    matched_token_grpo_surrogate,
    trl_group_standardized_advantages,
)
from rl_no_backward.matched_lora_backprop import (
    MatchedBackpropConfig,
    make_matched_lora_optimizer,
    make_matched_lr_scheduler,
    matched_backprop_grpo_step,
    matched_grpo_streaming_backward,
)
from rl_no_backward.model import ModelBundle
from rl_no_backward.sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.linspace(-0.3, 0.3, 7), requires_grad=False)
        self.lora = nn.Parameter(torch.tensor([0.02, -0.01, 0.03]))
        self.register_buffer(
            "features", torch.randn(7, 3, generator=torch.Generator().manual_seed(9))
        )

    def forward(self, *, input_ids: Tensor, attention_mask: Tensor, use_cache: bool):
        del attention_mask, use_cache
        context = 1.0 + input_ids.float().unsqueeze(-1) * 0.02
        return SimpleNamespace(logits=self.base + context * (self.features @ self.lora))


def _fixture() -> tuple[ModelBundle, SequenceRolloutBatch, Tensor]:
    model = _ToyPolicy()
    bundle = ModelBundle(
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["lora"],
        device=torch.device("cpu"),
        model_name="toy",
    )
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor(
        [[[True, True, False], [True, True, True]], [[True, True, False], [True, True, True]]]
    )
    rollout = SequenceRolloutBatch(
        prompts=("a", "b"),
        completions=(("x", "y"), ("x", "y")),
        prompt_input_ids=torch.tensor([[0, 2], [2, 3]]),
        prompt_attention_mask=torch.tensor([[False, True], [True, True]]),
        response_input_ids=torch.tensor([[[4, 1, 0], [5, 6, 1]], [[6, 1, 0], [4, 5, 1]]]),
        response_mask=mask,
        old_token_log_probs=torch.zeros(2, 2, 3),
        rewards=rewards,
        advantages=trl_group_standardized_advantages(rewards),
        sampling_temperature=1.0,
        pad_token_id=0,
        eos_token_ids=(1,),
    )
    with torch.inference_mode():
        old = teacher_forced_token_log_probs(bundle, rollout)
    rollout = replace(rollout, old_token_log_probs=old)
    sampler = old - 0.003 * mask
    return bundle, rollout, sampler


def test_streaming_gradient_matches_explicit_trl_equivalent_objective() -> None:
    full_bundle, full_rollout, full_sampler = _fixture()
    stream_bundle, stream_rollout, stream_sampler = _fixture()
    config = MatchedGRPOObjectiveConfig(
        inference_correction_mode="token_truncate",
        inference_ratio_min=0.1,
        inference_ratio_max=3.0,
    )

    full_new = teacher_forced_token_log_probs(full_bundle, full_rollout)
    full_loss = -matched_token_grpo_surrogate(
        full_new,
        full_rollout.old_token_log_probs,
        full_sampler,
        full_rollout.advantages,
        full_rollout.response_mask,
        config,
    )
    full_loss.backward()
    expected_gradient = full_bundle.trainable_parameters[0].grad.detach().clone()

    matched_grpo_streaming_backward(
        stream_bundle,
        stream_rollout,
        stream_sampler,
        config,
        prompt_groups_per_micro_batch=1,
    )
    actual_gradient = stream_bundle.trainable_parameters[0].grad
    assert actual_gradient is not None
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1.0e-5, atol=1.0e-6)
    assert stream_bundle.model.base.grad is None


def test_warmup_starts_at_zero_like_transformers_scheduler() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-5)
    scheduler = make_matched_lr_scheduler(
        optimizer,
        MatchedBackpropConfig(warmup_ratio=0.1),
        total_steps=30,
    )
    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5 / 3)


def test_backprop_step_reports_same_rollout_objective_and_negative_loss() -> None:
    bundle, rollout, sampler = _fixture()
    objective_config = MatchedGRPOObjectiveConfig(
        inference_correction_mode="token_truncate",
        inference_ratio_min=0.1,
        inference_ratio_max=3.0,
    )
    optimizer_config = MatchedBackpropConfig(
        learning_rate=0.02,
        fused_adamw=False,
    )
    optimizer = make_matched_lora_optimizer(bundle, optimizer_config)
    expected_before = float(
        matched_token_grpo_surrogate(
            rollout.old_token_log_probs,
            rollout.old_token_log_probs,
            sampler,
            rollout.advantages,
            rollout.response_mask,
            objective_config,
        ).item()
    )

    result = matched_backprop_grpo_step(
        bundle,
        rollout,
        sampler,
        optimizer,
        objective_config,
        optimizer_config,
    )
    with torch.inference_mode():
        new_logps = teacher_forced_token_log_probs(bundle, rollout)
        expected_after = float(
            matched_token_grpo_surrogate(
                new_logps,
                rollout.old_token_log_probs,
                sampler,
                rollout.advantages,
                rollout.response_mask,
                objective_config,
            ).item()
        )

    assert result.grpo_objective_before == pytest.approx(expected_before)
    assert result.grpo_objective_after == pytest.approx(expected_after)
    assert result.grpo_loss_before == pytest.approx(-expected_before)
    assert result.grpo_loss_after == pytest.approx(-expected_after)
    assert result.surrogate_improvement == pytest.approx(expected_after - expected_before)
