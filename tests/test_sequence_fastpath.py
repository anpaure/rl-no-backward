from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from rl_no_backward.model import (
    ModelBundle,
    ResidualCoreAdapter,
    parameter_vector,
    set_parameter_vector,
)
from rl_no_backward.sequence_backprop_fastpath import streaming_grpo_backward
from rl_no_backward.sequence_fastpath import (
    BatchedProbeResidualCoreAdapter,
    FusedProbeConfig,
    enable_batched_probe_adapters,
    fused_directional_token_log_probs,
    selected_token_log_probs,
)
from rl_no_backward.sequence_optimizers import ForwardSequenceConfig, forward_sequence_step
from rl_no_backward.sequence_policy import (
    SequenceRolloutBatch,
    clipped_grpo_surrogate,
    teacher_forced_token_log_probs,
)


class TinyProbeLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(11, 4)
        basis = torch.linalg.qr(torch.randn(4, 2), mode="reduced").Q
        self.adapter = BatchedProbeResidualCoreAdapter(basis, basis, scale=0.4)
        self.head = nn.Linear(4, 11, bias=False)
        self.forward_call_count = 0
        self.inference_modes: list[bool] = []
        self.requires_grad_(False)
        self.adapter.core.requires_grad_(True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.forward_call_count += 1
        self.inference_modes.append(torch.is_inference_mode_enabled())
        hidden = self.adapter(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def make_bundle_and_rollout() -> tuple[ModelBundle, SequenceRolloutBatch]:
    model = TinyProbeLM()
    bundle = ModelBundle(
        model=model,
        tokenizer=None,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter.core"],
        device=torch.device("cpu"),
        model_name="tiny-probe",
    )
    response_ids = torch.tensor([[[5, 6], [7, 8]], [[4, 3], [2, 1]]])
    response_mask = torch.tensor(
        [[[True, True], [True, False]], [[True, True], [True, True]]]
    )
    rollout = SequenceRolloutBatch(
        prompts=("a", "b"),
        completions=(("x", "y"), ("z", "w")),
        prompt_input_ids=torch.tensor([[1, 2], [3, 4]]),
        prompt_attention_mask=torch.ones(2, 2, dtype=torch.bool),
        response_input_ids=response_ids,
        response_mask=response_mask,
        old_token_log_probs=torch.zeros(2, 2, 2),
        rewards=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        advantages=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        sampling_temperature=0.8,
        pad_token_id=0,
        eos_token_ids=(10,),
    )
    return bundle, rollout


def test_selected_token_log_probs_matches_reference() -> None:
    logits = torch.tensor(
        [[[1000.0, 999.0, -1000.0], [-4.0, 2.0, 1.0]]], dtype=torch.float16
    )
    targets = torch.tensor([[0, 2]])
    actual = selected_token_log_probs(logits, targets, temperature=0.7)
    expected = (logits.float() / 0.7).log_softmax(dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    # Selected-logit minus log-sum-exp and materialized log-softmax differ by
    # one FP32 subtraction at this deliberately extreme scale.
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_selected_token_log_probs_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        selected_token_log_probs(torch.ones(1, 2), torch.zeros(1, dtype=torch.long), temperature)


def test_batched_probe_adapter_matches_independent_policies() -> None:
    bundle, rollout = make_bundle_and_rollout()
    center = torch.tensor([0.08, -0.03, 0.05, -0.02])
    set_parameter_vector(bundle, center)
    basis = torch.linalg.qr(torch.tensor([[1.0, 0.2], [0.1, 1.0], [0.3, 0.4], [0.5, -0.2]]))[0]
    radius = 0.17

    positive = []
    negative = []
    with torch.inference_mode():
        for index in range(basis.shape[1]):
            set_parameter_vector(bundle, center + radius * basis[:, index])
            positive.append(teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2))
            set_parameter_vector(bundle, center - radius * basis[:, index])
            negative.append(teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2))
        set_parameter_vector(bundle, center)

    result = fused_directional_token_log_probs(
        bundle,
        rollout,
        center,
        basis,
        radius,
        FusedProbeConfig(directions_per_forward=1, examples_per_forward=2),
    )

    torch.testing.assert_close(result.positive_token_log_probs, torch.stack(positive, dim=-1))
    torch.testing.assert_close(result.negative_token_log_probs, torch.stack(negative, dim=-1))
    assert result.model_calls == 4
    assert result.full_prefix_calls == result.model_calls
    assert result.suffix_calls == result.model_calls
    assert torch.equal(parameter_vector(bundle), center)
    assert bundle.model.adapter._probe_cores is None


def test_fused_probe_accepts_an_unresolved_bundle_device_alias() -> None:
    bundle, rollout = make_bundle_and_rollout()
    bundle.device = torch.device("cpu:0")
    center = parameter_vector(bundle)
    basis = torch.eye(center.numel())[:, :1]

    result = fused_directional_token_log_probs(
        bundle,
        rollout,
        center,
        basis,
        0.1,
        FusedProbeConfig(directions_per_forward=1, examples_per_forward=4),
    )

    assert result.positive_token_log_probs.shape[-1] == 1


def test_probe_context_restores_state_after_failure() -> None:
    bundle, _ = make_bundle_and_rollout()
    adapter = bundle.model.adapter
    cores = torch.zeros(3, 2, 2)
    with pytest.raises(RuntimeError, match="boom"), adapter.use_probe_cores(cores):
        raise RuntimeError("boom")
    assert adapter._probe_cores is None


def test_enabling_batched_probe_adapters_preserves_ordinary_policy() -> None:
    bundle, rollout = make_bundle_and_rollout()
    old_adapter = bundle.model.adapter
    ordinary = ResidualCoreAdapter(
        old_adapter.p_basis,
        old_adapter.q_basis,
        scale=old_adapter.scale,
    )
    bundle.model.adapter = ordinary
    center = torch.tensor([0.08, -0.03, 0.05, -0.02])
    set_parameter_vector(bundle, center)

    with torch.inference_mode():
        before = teacher_forced_token_log_probs(bundle, rollout)
    assert enable_batched_probe_adapters(bundle) == 1
    with torch.inference_mode():
        after = teacher_forced_token_log_probs(bundle, rollout)

    assert isinstance(bundle.model.adapter, BatchedProbeResidualCoreAdapter)
    assert torch.equal(parameter_vector(bundle), center)
    torch.testing.assert_close(after, before)
    assert enable_batched_probe_adapters(bundle) == 0


def test_fused_probe_config_reports_peak_model_batch() -> None:
    config = FusedProbeConfig(directions_per_forward=3, examples_per_forward=8)
    assert config.maximum_model_batch == 48


def test_forward_optimizer_uses_fused_probes_and_exact_call_counters() -> None:
    bundle, rollout = make_bundle_and_rollout()
    config = ForwardSequenceConfig(
        method="fo_pg",
        directions=2,
        finite_difference_mu=0.05,
        kl_budget=0.01,
        max_step_norm=0.1,
        line_search_steps=2,
        minimum_surrogate_improvement=1_000_000.0,
        scoring_micro_batch_size=2,
        use_fused_probes=True,
        fused_probe_directions_per_forward=2,
        fused_probe_examples_per_forward=4,
    )

    result = forward_sequence_step(
        bundle,
        rollout,
        torch.Generator().manual_seed(31),
        config,
    )

    assert not result.accepted
    assert result.line_search_trials == config.line_search_steps
    assert result.policy_evaluations == 2 * config.directions + result.line_search_trials
    # All four signed policies and all four examples fit in one probe call;
    # each line-search policy still uses two ordinary scoring micro-batches.
    assert result.forward_calls == 1 + 2 * result.line_search_trials
    assert result.full_prefix_calls == result.forward_calls
    assert result.suffix_calls == result.forward_calls
    assert bundle.model.forward_call_count == result.forward_calls
    assert result.backward_calls == 0
    assert result.teacher_forced_examples == rollout.environment_samples * (
        2 * config.directions + result.line_search_trials
    )
    assert result.scored_tokens == rollout.valid_response_tokens * (
        2 * config.directions + result.line_search_trials
    )
    assert bundle.model.inference_modes and all(bundle.model.inference_modes)
    assert bundle.model.adapter._probe_cores is None


def test_streaming_backward_matches_full_graph_gradient() -> None:
    bundle, rollout = make_bundle_and_rollout()
    center = torch.tensor([0.04, -0.02, 0.03, 0.01])
    set_parameter_vector(bundle, center)

    bundle.model.adapter.core.grad = None
    token_log_probs = teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2)
    reference_surrogate = clipped_grpo_surrogate(token_log_probs, rollout, clip_epsilon=0.2)
    (-reference_surrogate).backward()
    reference_gradient = bundle.model.adapter.core.grad.detach().clone()

    bundle.model.adapter.core.grad = None
    result = streaming_grpo_backward(
        bundle,
        rollout,
        clip_epsilon=0.2,
        micro_batch_size=1,
    )

    torch.testing.assert_close(
        bundle.model.adapter.core.grad,
        reference_gradient,
        atol=2e-6,
        rtol=2e-5,
    )
    assert result.surrogate == pytest.approx(float(reference_surrogate.item()), abs=2e-6)
    assert result.model_calls == rollout.environment_samples
    assert result.full_prefix_calls == result.model_calls
    assert result.suffix_calls == result.model_calls
