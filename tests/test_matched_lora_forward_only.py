from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import Tensor, nn

import rl_no_backward.matched_lora_forward_only as forward_module
from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    trl_group_standardized_advantages,
)
from rl_no_backward.matched_lora_forward_only import (
    MatchedForwardConfig,
    matched_forward_npg_step,
)
from rl_no_backward.model import ModelBundle, parameter_vector
from rl_no_backward.sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_logits = nn.Parameter(torch.linspace(-0.3, 0.3, 7), requires_grad=False)
        self.lora = nn.Parameter(torch.zeros(3))
        self.register_buffer(
            "features",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, -0.3, 0.1],
                    [-0.4, 0.2, 0.3],
                    [0.1, 0.5, -0.2],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def forward(self, *, input_ids: Tensor, attention_mask: Tensor, use_cache: bool):
        del attention_mask, use_cache
        scale = 1.0 + 0.03 * input_ids.float().unsqueeze(-1)
        return SimpleNamespace(logits=self.base_logits + scale * (self.features @ self.lora))


def _fixture() -> tuple[ModelBundle, SequenceRolloutBatch]:
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
        old_hf = teacher_forced_token_log_probs(bundle, rollout)
    return bundle, replace(rollout, old_token_log_probs=old_hf)


def test_matched_forward_npg_moves_policy_without_reverse_mode() -> None:
    bundle, rollout = _fixture()
    sampler_logps = rollout.old_token_log_probs - 0.002 * rollout.response_mask
    before = parameter_vector(bundle).clone()
    result = matched_forward_npg_step(
        bundle,
        rollout,
        sampler_logps,
        torch.Generator().manual_seed(7),
        MatchedGRPOObjectiveConfig(
            inference_correction_mode="token_truncate",
            inference_ratio_min=0.1,
            inference_ratio_max=3.0,
        ),
        MatchedForwardConfig(
            directions=2,
            finite_difference_mu=0.02,
            kl_budget=0.03,
            max_step_norm=0.2,
            scoring_micro_batch_size=2,
        ),
    )
    assert result.accepted
    assert result.backward_calls == 0
    assert result.step_norm > 0
    assert not torch.equal(parameter_vector(bundle), before)
    assert not bundle.trainable_parameters[0].requires_grad
    assert bundle.trainable_parameters[0].grad is None


def test_forward_module_source_has_no_reverse_mode_api_calls() -> None:
    tree = ast.parse(inspect.getsource(forward_module))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "grad"}:
            forbidden.append(node.func.attr)
        if isinstance(node.func, ast.Name) and node.func.id in {"grad", "backward"}:
            forbidden.append(node.func.id)
    assert forbidden == []
