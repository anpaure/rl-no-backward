from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    trl_group_standardized_advantages,
)
from rl_no_backward.matched_lora_forward_only import MatchedForwardConfig
from rl_no_backward.matched_lora_gradient_gate import fixed_rollout_projected_gradient_gate
from rl_no_backward.model import ModelBundle, parameter_vector
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


def test_center_token_score_coordinates_match_exact_gradient_with_length_and_weights() -> None:
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
    rollout = SequenceRolloutBatch(
        prompts=("a", "b"),
        completions=(("x", "y"), ("x", "y")),
        prompt_input_ids=torch.tensor([[0, 2], [2, 3]]),
        prompt_attention_mask=torch.tensor([[False, True], [True, True]]),
        response_input_ids=torch.tensor([[[4, 1, 0], [5, 6, 1]], [[6, 5, 1], [4, 0, 0]]]),
        response_mask=torch.tensor(
            [
                [[True, True, False], [True, True, True]],
                [[True, True, True], [True, False, False]],
            ]
        ),
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
    sampler_offsets = torch.tensor(
        [
            [[0.01, 0.04, 0.00], [0.02, 0.05, 0.08]],
            [[0.03, 0.06, 0.09], [0.07, 0.00, 0.00]],
        ]
    )
    sampler = old - sampler_offsets
    before = parameter_vector(bundle).clone()
    report = fixed_rollout_projected_gradient_gate(
        bundle,
        rollout,
        sampler,
        MatchedGRPOObjectiveConfig(
            inference_correction_mode="token_truncate",
            inference_ratio_min=0.1,
            inference_ratio_max=3.0,
        ),
        MatchedForwardConfig(
            directions=2,
            finite_difference_mu=1.0e-3,
            scoring_micro_batch_size=4,
        ),
        torch.Generator().manual_seed(4),
    )
    assert report.parameter_integrity
    assert report.cosine_similarity > 0.999
    assert report.relative_l2_error < 0.01
    torch.testing.assert_close(parameter_vector(bundle), before, rtol=0, atol=0)
