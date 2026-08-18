from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_no_backward.sequence_forward_only as forward_module
from rl_no_backward.model import ModelBundle, parameter_vector
from rl_no_backward.sequence_optimizers import (
    BackpropSequenceConfig,
    ForwardSequenceConfig,
    SequenceActiveSubspace,
    directional_sequence_score_statistics,
    forward_sequence_step,
    make_sequence_grpo_optimizer,
    sampled_sequence_kl,
    sequence_grpo_step,
)
from rl_no_backward.sequence_policy import (
    SequenceRolloutBatch,
    group_leave_one_out_advantages,
    teacher_forced_token_log_probs,
)


class ToySequencePolicy(nn.Module):
    """Tiny differentiable causal policy with no generation API."""

    def __init__(self) -> None:
        super().__init__()
        self.base_logits = nn.Parameter(
            torch.tensor([0.0, -0.2, 0.1, -0.1, 0.2, -0.05, 0.05]),
            requires_grad=False,
        )
        self.adapter = nn.Parameter(torch.zeros(3))
        self.register_buffer(
            "token_features",
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
        self.forward_call_count = 0
        self.fail_on_call: int | None = None

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.forward_call_count += 1
        if self.fail_on_call == self.forward_call_count:
            raise RuntimeError("intentional toy forward failure")
        adapter_logits = self.token_features @ self.adapter
        context_scale = 1.0 + 0.04 * input_ids.float().unsqueeze(-1)
        logits = self.base_logits + context_scale * adapter_logits
        return SimpleNamespace(logits=logits)


class UnusedTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def make_bundle_and_rollout() -> tuple[ModelBundle, ToySequencePolicy, SequenceRolloutBatch]:
    model = ToySequencePolicy()
    bundle = ModelBundle(
        model=model,
        tokenizer=UnusedTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="toy-sequence-policy",
    )
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    response_mask = torch.tensor(
        [[[True, True, False], [True, True, True]], [[True, True, False], [True, True, True]]]
    )
    rollout = SequenceRolloutBatch(
        prompts=("p0", "p1"),
        completions=(("a", "b"), ("c", "d")),
        prompt_input_ids=torch.tensor([[0, 2], [2, 3]]),
        prompt_attention_mask=torch.tensor([[False, True], [True, True]]),
        response_input_ids=torch.tensor([[[4, 1, 0], [5, 6, 1]], [[6, 1, 0], [4, 5, 1]]]),
        response_mask=response_mask,
        old_token_log_probs=torch.zeros(2, 2, 3),
        rewards=rewards,
        advantages=group_leave_one_out_advantages(rewards),
        sampling_temperature=1.0,
        pad_token_id=0,
        eos_token_ids=(1,),
    )
    with torch.no_grad():
        old_token_log_probs = teacher_forced_token_log_probs(bundle, rollout)
    rollout = replace(rollout, old_token_log_probs=old_token_log_probs.detach())
    model.forward_call_count = 0
    return bundle, model, rollout


def test_backprop_sequence_grpo_updates_only_adapters_and_tracks_token_work() -> None:
    bundle, model, rollout = make_bundle_and_rollout()
    config = BackpropSequenceConfig(
        learning_rate=0.03,
        epochs_per_rollout=2,
        max_grad_norm=2.0,
        scoring_micro_batch_size=2,
    )
    optimizer = make_sequence_grpo_optimizer(bundle, config)
    before = parameter_vector(bundle).clone()
    base_before = model.base_logits.detach().clone()

    result = sequence_grpo_step(bundle, rollout, optimizer, config)

    assert result.accepted
    assert not torch.equal(parameter_vector(bundle), before)
    assert torch.equal(model.base_logits, base_before)
    assert model.base_logits.grad is None
    assert result.reward_mean == pytest.approx(0.5)
    assert result.backward_calls == 2
    assert result.policy_evaluations == 3
    assert result.forward_calls == 6
    assert result.full_prefix_calls == result.forward_calls
    assert result.suffix_calls == result.forward_calls
    assert model.forward_call_count == result.forward_calls
    assert result.teacher_forced_examples == 12
    assert result.scored_tokens == 30
    assert result.environment_samples == 4
    assert result.step_norm > 0
    assert result.projected_gradient_norm > 0
    assert result.empirical_kl >= 0
    assert result.surrogate_improvement > 0
    assert torch.equal(rollout.old_token_log_probs, rollout.old_token_log_probs.detach())


def test_streaming_backprop_matches_full_graph_update_and_counts_chunk_backwards() -> None:
    reference_bundle, reference_model, reference_rollout = make_bundle_and_rollout()
    streaming_bundle, streaming_model, streaming_rollout = make_bundle_and_rollout()
    reference_config = BackpropSequenceConfig(
        learning_rate=0.03,
        epochs_per_rollout=2,
        max_grad_norm=2.0,
        scoring_micro_batch_size=2,
    )
    streaming_config = replace(reference_config, use_streaming_backward=True)

    reference_result = sequence_grpo_step(
        reference_bundle,
        reference_rollout,
        make_sequence_grpo_optimizer(reference_bundle, reference_config),
        reference_config,
    )
    streaming_result = sequence_grpo_step(
        streaming_bundle,
        streaming_rollout,
        make_sequence_grpo_optimizer(streaming_bundle, streaming_config),
        streaming_config,
    )

    torch.testing.assert_close(
        parameter_vector(streaming_bundle),
        parameter_vector(reference_bundle),
        atol=2e-6,
        rtol=2e-5,
    )
    assert reference_result.forward_calls == streaming_result.forward_calls == 6
    assert streaming_result.full_prefix_calls == streaming_result.forward_calls
    assert streaming_result.suffix_calls == streaming_result.forward_calls
    assert reference_model.forward_call_count == reference_result.forward_calls
    assert streaming_model.forward_call_count == streaming_result.forward_calls
    assert reference_result.backward_calls == 2
    assert streaming_result.backward_calls == 4
    assert streaming_result.policy_evaluations == reference_result.policy_evaluations == 3
    assert streaming_result.teacher_forced_examples == reference_result.teacher_forced_examples
    assert streaming_result.scored_tokens == reference_result.scored_tokens
    assert streaming_result.empirical_kl == pytest.approx(
        reference_result.empirical_kl, abs=2e-6
    )


@pytest.mark.parametrize("method", ["fo_pg", "fo_npg", "focus_npg"])
def test_forward_sequence_methods_accept_trust_region_steps_without_backward(method: str) -> None:
    bundle, model, rollout = make_bundle_and_rollout()
    config = ForwardSequenceConfig(
        method=method,
        directions=2,
        finite_difference_mu=0.02,
        fisher_damping=0.1,
        kl_budget=0.03,
        max_step_norm=0.2,
        line_search_steps=5,
        scoring_micro_batch_size=2,
    )
    state = SequenceActiveSubspace(rank=1, history_size=4) if method == "focus_npg" else None
    generator = torch.Generator().manual_seed(23)
    before = parameter_vector(bundle).clone()
    model.adapter.grad = torch.ones_like(model.adapter)

    result = forward_sequence_step(bundle, rollout, generator, config, state)

    assert result.accepted
    assert not torch.equal(parameter_vector(bundle), before)
    assert not bundle.trainable_parameters[0].requires_grad
    assert bundle.trainable_parameters[0].grad is None
    assert result.backward_calls == 0
    assert result.policy_evaluations == 2 * config.directions + result.line_search_trials
    assert result.forward_calls == 2 * result.policy_evaluations
    assert result.full_prefix_calls == result.forward_calls
    assert result.suffix_calls == result.forward_calls
    assert model.forward_call_count == result.forward_calls
    assert result.teacher_forced_examples == 4 * result.policy_evaluations
    assert result.scored_tokens == 10 * result.policy_evaluations
    assert 0 < result.step_norm <= config.max_step_norm * 1.001
    assert 0 <= result.empirical_kl <= config.kl_budget * 1.05
    assert result.surrogate_improvement >= config.minimum_surrogate_improvement
    assert result.projected_gradient_norm > 0
    assert torch.isfinite(torch.tensor(result.fisher_condition))
    assert torch.isfinite(torch.tensor(result.derivative_variance))
    if state is not None:
        assert len(state.history) == 1
        assert state.basis is not None
        assert state.basis.shape == (3, 1)


def test_directional_probes_and_rejected_line_search_restore_exact_center() -> None:
    bundle, model, rollout = make_bundle_and_rollout()
    center = parameter_vector(bundle).float().clone()
    basis = torch.eye(center.numel())[:, :2]

    statistics, evaluations = directional_sequence_score_statistics(
        bundle,
        rollout,
        center,
        basis,
        0.03,
        scoring_micro_batch_size=2,
    )
    assert evaluations == 4
    assert statistics.gradient.shape == (2,)
    assert statistics.fisher.shape == (2, 2)
    assert torch.equal(parameter_vector(bundle), center)

    model.forward_call_count = 0
    config = ForwardSequenceConfig(
        method="fo_pg",
        directions=2,
        finite_difference_mu=0.03,
        kl_budget=0.02,
        max_step_norm=0.2,
        line_search_steps=3,
        minimum_surrogate_improvement=100.0,
        scoring_micro_batch_size=2,
    )
    result = forward_sequence_step(bundle, rollout, torch.Generator().manual_seed(4), config)
    assert not result.accepted
    assert result.line_search_trials == 3
    assert result.policy_evaluations == 2 * config.directions + config.line_search_steps
    assert result.step_norm == 0
    assert torch.equal(parameter_vector(bundle), center)


def test_probe_exception_restores_parameters() -> None:
    bundle, model, rollout = make_bundle_and_rollout()
    center = parameter_vector(bundle).float().clone()
    basis = torch.eye(center.numel())[:, :2]
    model.fail_on_call = 2

    with pytest.raises(RuntimeError, match="intentional"):
        directional_sequence_score_statistics(bundle, rollout, center, basis, 0.02)

    assert torch.equal(parameter_vector(bundle), center)


def test_sampled_kl_is_masked_nonnegative_and_zero_at_behavior_policy() -> None:
    _, _, rollout = make_bundle_and_rollout()
    assert sampled_sequence_kl(rollout.old_token_log_probs, rollout).item() == pytest.approx(0.0)

    changed = rollout.old_token_log_probs.clone()
    changed[rollout.response_mask] += 0.2
    changed[~rollout.response_mask] = 100.0
    assert sampled_sequence_kl(changed, rollout).item() > 0


def test_forward_only_module_has_no_reverse_mode_calls_and_scores_under_inference() -> None:
    source = Path(forward_module.__file__).read_text()
    tree = ast.parse(source)
    forbidden_call_names = {"backward", "grad", "jvp", "vjp", "jacrev", "jacfwd"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts: list[str] = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        parts.reverse()
        if "autograd" in parts or (parts and parts[-1] in forbidden_call_names):
            violations.append(".".join(parts))
    assert violations == []

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in (
        "sampled_sequence_kl",
        "directional_sequence_score_statistics",
        "_directional_sequence_score_statistics_with_counts",
        "_solve_projected_coordinates",
        "_initial_step_scale",
        "_sequence_line_search",
        "forward_sequence_step",
    ):
        decorators = {ast.unparse(decorator) for decorator in functions[name].decorator_list}
        assert "torch.inference_mode()" in decorators
