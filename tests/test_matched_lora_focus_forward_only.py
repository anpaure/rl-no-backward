from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_no_backward.matched_lora_focus_forward_only as focus_step_module
from rl_no_backward.matched_focus import MatchedFocusState, canonical_lora_partition
from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    trl_group_standardized_advantages,
)
from rl_no_backward.matched_lora_focus_forward_only import (
    deterministic_prompt_half_coordinates,
    matched_focus_npg_step,
)
from rl_no_backward.matched_lora_forward_only import MatchedForwardConfig
from rl_no_backward.model import ModelBundle, parameter_vector
from rl_no_backward.sequence_policy import (
    ProjectedScoreStatistics,
    SequenceRolloutBatch,
    teacher_forced_token_log_probs,
)
from rl_no_backward.standard_lora import named_lora_parameters


class _ToyLoRAPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(
                torch.tensor([[0.4, -0.2, 0.1, 0.3], [-0.1, 0.5, 0.2, -0.4]])
            )
            self.lora_B["default"].weight.zero_()


class _ToyLoRAPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_logits = nn.Parameter(
            torch.linspace(-0.3, 0.3, 7),
            requires_grad=False,
        )
        self.proj = _ToyLoRAPair()
        self.register_buffer(
            "features",
            torch.tensor(
                [
                    [0.2, -0.1, 0.3, 0.4],
                    [-0.4, 0.2, 0.1, 0.3],
                    [0.1, 0.5, -0.2, 0.2],
                    [0.3, -0.3, 0.4, -0.1],
                    [0.7, 0.1, -0.2, 0.2],
                    [-0.2, 0.6, 0.3, -0.4],
                    [0.4, -0.5, 0.2, 0.1],
                ]
            ),
        )

    def forward(self, *, input_ids: Tensor, attention_mask: Tensor, use_cache: bool):
        del attention_mask, use_cache
        a_weight = self.proj.lora_A["default"].weight
        b_weight = self.proj.lora_B["default"].weight
        token_features = self.features[input_ids]
        adapted = token_features @ (b_weight @ a_weight).T
        logits = self.base_logits + adapted @ self.features.T
        return SimpleNamespace(logits=logits)


def _fixture() -> tuple[ModelBundle, SequenceRolloutBatch]:
    model = _ToyLoRAPolicy()
    adapter_names = [name for name, _ in named_lora_parameters(model)]
    bundle = ModelBundle(
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=adapter_names,
        device=torch.device("cpu"),
        model_name="toy-lora",
    )
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    response_mask = torch.tensor(
        [
            [[True, True, False], [True, True, True]],
            [[True, True, False], [True, True, True]],
            [[True, True, False], [True, True, True]],
            [[True, True, False], [True, True, True]],
        ]
    )
    rollout = SequenceRolloutBatch(
        prompts=("a", "b", "c", "d"),
        completions=(("x", "y"),) * 4,
        prompt_input_ids=torch.tensor([[0, 2], [2, 3], [1, 4], [3, 5]]),
        prompt_attention_mask=torch.ones(4, 2, dtype=torch.bool),
        response_input_ids=torch.tensor(
            [
                [[4, 1, 0], [5, 6, 1]],
                [[6, 1, 0], [4, 5, 1]],
                [[5, 1, 0], [6, 4, 1]],
                [[4, 1, 0], [5, 3, 1]],
            ]
        ),
        response_mask=response_mask,
        old_token_log_probs=torch.zeros(4, 2, 3),
        rewards=rewards,
        advantages=trl_group_standardized_advantages(rewards),
        sampling_temperature=1.0,
        pad_token_id=0,
        eos_token_ids=(1,),
    )
    with torch.inference_mode():
        old_hf = teacher_forced_token_log_probs(bundle, rollout)
    return bundle, replace(rollout, old_token_log_probs=old_hf)


def _forward_config(**overrides) -> MatchedForwardConfig:
    values = {
        "directions": 8,
        "finite_difference_mu": 0.05,
        "fisher_damping": 0.2,
        "kl_budget": 1.0,
        "max_step_norm": 0.02,
        "line_search_steps": 2,
        "line_search_decay": 0.5,
        "minimum_surrogate_improvement": -10.0,
        "scoring_micro_batch_size": 8,
    }
    values.update(overrides)
    return MatchedForwardConfig(**values)


def test_prompt_halves_use_even_odd_prompts_and_existing_completion_scores() -> None:
    _, rollout = _fixture()
    completion_scores = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    statistics = ProjectedScoreStatistics(
        directional_token_scores=torch.empty(4, 2, 3, 3),
        completion_scores=completion_scores,
        gradient=torch.empty(3),
        fisher=torch.empty(3, 3),
    )
    first, second = deterministic_prompt_half_coordinates(statistics, rollout)
    contributions = rollout.advantages.unsqueeze(-1) * completion_scores
    torch.testing.assert_close(first, contributions[0::2].mean(dim=(0, 1)))
    torch.testing.assert_close(second, contributions[1::2].mean(dim=(0, 1)))


def test_focus_step_bootstraps_b_then_uses_a4_b4_without_extra_model_evaluations(
    monkeypatch,
) -> None:
    bundle, rollout = _fixture()
    state = MatchedFocusState(canonical_lora_partition(bundle.model))
    original_teacher = focus_step_module.teacher_forced_token_log_probs
    model_evaluations = 0

    def counted_teacher(*args, **kwargs):
        nonlocal model_evaluations
        model_evaluations += 1
        assert not torch.is_grad_enabled()
        return original_teacher(*args, **kwargs)

    monkeypatch.setattr(
        focus_step_module,
        "teacher_forced_token_log_probs",
        counted_teacher,
    )
    before = parameter_vector(bundle).clone()
    first = matched_focus_npg_step(
        bundle,
        rollout,
        rollout.old_token_log_probs,
        torch.Generator().manual_seed(71),
        MatchedGRPOObjectiveConfig(),
        _forward_config(),
        state,
    )
    assert first.accepted
    assert first.focus_bootstrap_b_only is True
    assert first.focus_a_update_count == 0
    assert first.focus_b_update_count == 1
    assert first.focus_cross_sketch_count == 1
    assert first.focus_state_update_policy_evaluations == 0
    assert first.policy_evaluations == model_evaluations
    assert first.backward_calls == 0
    assert not torch.equal(parameter_vector(bundle), before)
    assert all(not parameter.requires_grad for parameter in bundle.trainable_parameters)
    assert all(parameter.grad is None for parameter in bundle.trainable_parameters)

    with torch.inference_mode():
        new_old = original_teacher(bundle, rollout)
    current_rollout = replace(rollout, old_token_log_probs=new_old)
    prior_evaluations = model_evaluations
    second = matched_focus_npg_step(
        bundle,
        current_rollout,
        current_rollout.old_token_log_probs,
        torch.Generator().manual_seed(72),
        MatchedGRPOObjectiveConfig(),
        _forward_config(),
        state,
    )
    assert second.focus_bootstrap_b_only is False
    assert second.focus_a_update_count == 1
    assert second.focus_b_update_count == 2
    assert second.focus_cross_sketch_count == 2
    assert second.policy_evaluations == model_evaluations - prior_evaluations
    assert second.focus_state_numel <= second.focus_state_numel_cap


def test_focus_covariance_state_persists_when_line_search_rejects() -> None:
    bundle, rollout = _fixture()
    state = MatchedFocusState(canonical_lora_partition(bundle.model))
    center = parameter_vector(bundle).clone()
    config = _forward_config(
        line_search_steps=1,
        minimum_surrogate_improvement=1.0e6,
    )
    first = matched_focus_npg_step(
        bundle,
        rollout,
        rollout.old_token_log_probs,
        torch.Generator().manual_seed(81),
        MatchedGRPOObjectiveConfig(),
        config,
        state,
    )
    assert not first.accepted
    assert first.focus_b_update_count == 1
    assert first.grpo_objective_after == first.grpo_objective_before
    assert first.grpo_loss_after == first.grpo_loss_before
    assert first.surrogate_improvement == 0.0
    assert first.line_search_candidate_grpo_objective is not None
    assert first.line_search_candidate_grpo_loss == pytest.approx(
        -first.line_search_candidate_grpo_objective
    )
    torch.testing.assert_close(parameter_vector(bundle), center, rtol=0, atol=0)
    first_state_numel = state.persistent_numel

    second = matched_focus_npg_step(
        bundle,
        rollout,
        rollout.old_token_log_probs,
        torch.Generator().manual_seed(82),
        MatchedGRPOObjectiveConfig(),
        config,
        state,
    )
    assert not second.accepted
    assert second.focus_b_update_count == 2
    assert second.focus_a_update_count == 1
    assert second.focus_bootstrap_b_only is False
    assert second.focus_cross_sketch_count == 2
    assert state.persistent_numel >= first_state_numel
    torch.testing.assert_close(parameter_vector(bundle), center, rtol=0, atol=0)


def test_focus_step_rejects_non_q8_and_unequal_prompt_halves() -> None:
    bundle, rollout = _fixture()
    state = MatchedFocusState(canonical_lora_partition(bundle.model))
    with pytest.raises(ValueError, match="exactly eight"):
        matched_focus_npg_step(
            bundle,
            rollout,
            rollout.old_token_log_probs,
            torch.Generator().manual_seed(91),
            MatchedGRPOObjectiveConfig(),
            MatchedForwardConfig(directions=4),
            state,
        )

    odd_rollout = replace(
        rollout,
        prompts=rollout.prompts[:3],
        completions=rollout.completions[:3],
        prompt_input_ids=rollout.prompt_input_ids[:3],
        prompt_attention_mask=rollout.prompt_attention_mask[:3],
        response_input_ids=rollout.response_input_ids[:3],
        response_mask=rollout.response_mask[:3],
        old_token_log_probs=rollout.old_token_log_probs[:3],
        rewards=rollout.rewards[:3],
        advantages=rollout.advantages[:3],
    )
    with pytest.raises(ValueError, match="even prompt batch"):
        matched_focus_npg_step(
            bundle,
            odd_rollout,
            odd_rollout.old_token_log_probs,
            torch.Generator().manual_seed(92),
            MatchedGRPOObjectiveConfig(),
            _forward_config(),
            state,
        )


def test_focus_production_step_has_no_reverse_mode_or_backprop_import() -> None:
    tree = ast.parse(inspect.getsource(focus_step_module))
    forbidden_calls: list[str] = []
    forbidden_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "backprop" in node.module:
            forbidden_imports.append(node.module)
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "grad"}:
            forbidden_calls.append(node.func.attr)
        if isinstance(node.func, ast.Name) and node.func.id in {"backward", "grad"}:
            forbidden_calls.append(node.func.id)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "autograd"
        ):
            forbidden_calls.append("autograd")
    assert forbidden_calls == []
    assert forbidden_imports == []
