from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import Tensor, nn

from rl_no_backward.model import ModelBundle
from rl_no_backward.sequence_policy import (
    CompletionSample,
    central_difference_score_statistics,
    clipped_grpo_surrogate,
    completion_log_probs,
    generate_sequence_rollouts,
    group_leave_one_out_advantages,
    response_token_mask,
    teacher_forced_token_log_probs,
)


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    _prompts: ClassVar[dict[str, list[int]]] = {"short": [2], "long": [2, 7]}
    _decoded: ClassVar[dict[int, str]] = {3: "A", 4: "B", 5: "C", 6: "D", 7: "E"}

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int | None = None,
    ) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        assert padding
        rows = [
            self._prompts[text][:max_length] if truncation else self._prompts[text]
            for text in texts
        ]
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self._decoded.get(token_id, "") for token_id in token_ids)


class ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_logits = nn.Parameter(
            torch.tensor([-0.4, -0.2, 0.0, 0.1, 0.3, -0.1, 0.2, -0.3]),
            requires_grad=False,
        )
        self.adapter = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "adapter_feature", torch.tensor([-0.6, 0.2, 0.8, -0.3, 0.5, -0.8, 0.4, 0.1])
        )
        self.generate_input_ids: Tensor | None = None
        self.generate_attention_mask: Tensor | None = None

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        context = 0.01 * input_ids.float().unsqueeze(-1) * self.adapter_feature
        logits = self.base_logits + self.adapter * self.adapter_feature + context
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        generation_config: object,
    ) -> SimpleNamespace:
        assert generation_config.return_dict_in_generate
        assert generation_config.num_return_sequences == 2
        assert generation_config.max_new_tokens == 3
        assert generation_config.do_sample
        assert generation_config.top_k == 0
        assert generation_config.top_p == 1.0
        self.generate_input_ids = input_ids.detach().clone()
        self.generate_attention_mask = attention_mask.detach().clone()
        responses = torch.tensor(
            [
                [3, 1, 0],  # prompt 0, completion 0: "A" then EOS and padding
                [4, 5, 6],  # prompt 0, completion 1: "BCD", length limited
                [1, 0, 0],  # prompt 1, completion 0: immediate EOS
                [6, 3, 1],  # prompt 1, completion 1: "DA" then EOS
            ],
            device=input_ids.device,
        )
        repeated = input_ids.repeat_interleave(generation_config.num_return_sequences, dim=0)
        return SimpleNamespace(sequences=torch.cat([repeated, responses], dim=1))


def make_bundle() -> tuple[ModelBundle, ToyCausalLM]:
    model = ToyCausalLM()
    bundle = ModelBundle(
        model=model,
        tokenizer=ToyTokenizer(),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="toy-causal-lm",
    )
    return bundle, model


def make_rollout() -> tuple[ModelBundle, ToyCausalLM, object]:
    bundle, model = make_bundle()
    expected_answers = ("A", "DA")

    def exact_reward(sample: CompletionSample) -> bool:
        return sample.completion == expected_answers[sample.prompt_index]

    rollout = generate_sequence_rollouts(
        bundle,
        ["short", "long"],
        exact_reward,
        group_size=2,
        max_new_tokens=3,
        temperature=0.7,
        seed=17,
    )
    return bundle, model, rollout


def test_generation_records_grouped_tokens_masks_log_probs_and_exact_rewards() -> None:
    bundle, model, rollout = make_rollout()

    assert torch.equal(model.generate_input_ids, torch.tensor([[0, 2], [2, 7]]))
    assert torch.equal(model.generate_attention_mask, torch.tensor([[False, True], [True, True]]))
    assert rollout.completions == (("A", "BCD"), ("", "DA"))
    assert torch.equal(
        rollout.response_input_ids,
        torch.tensor([[[3, 1, 0], [4, 5, 6]], [[1, 0, 0], [6, 3, 1]]]),
    )
    assert torch.equal(
        rollout.response_mask,
        torch.tensor(
            [[[True, True, False], [True, True, True]], [[True, False, False], [True, True, True]]]
        ),
    )
    assert torch.equal(rollout.rewards, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(rollout.advantages, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
    assert rollout.environment_samples == 4
    assert rollout.valid_response_tokens == 9
    assert not rollout.old_token_log_probs.requires_grad
    assert torch.equal(
        rollout.old_token_log_probs.masked_select(~rollout.response_mask), torch.zeros(3)
    )

    rescored = teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2)
    assert torch.allclose(rescored, rollout.old_token_log_probs)
    assert torch.equal(
        rollout.old_completion_log_probs, completion_log_probs(rescored, rollout.response_mask)
    )
    assert rollout.flat_input_ids.shape == (4, 5)
    assert rollout.flat_attention_mask.dtype == torch.bool


def test_teacher_forced_rescoring_supports_backprop_and_inference_only_perturbations() -> None:
    bundle, model, rollout = make_rollout()

    token_log_probs = teacher_forced_token_log_probs(bundle, rollout)
    completion_log_probs(token_log_probs, rollout.response_mask).sum().backward()
    assert model.adapter.grad is not None
    assert torch.isfinite(model.adapter.grad)
    assert model.adapter.grad.abs() > 0
    assert model.base_logits.grad is None

    with torch.no_grad():
        model.adapter.add_(0.5)
    with torch.inference_mode():
        perturbed = teacher_forced_token_log_probs(bundle, rollout)
    assert not torch.allclose(
        perturbed.masked_select(rollout.response_mask),
        rollout.old_token_log_probs.masked_select(rollout.response_mask),
    )
    assert torch.equal(perturbed.masked_select(~rollout.response_mask), torch.zeros(3))


def test_clipped_grpo_surrogate_is_token_masked_and_completion_normalized() -> None:
    _, _, rollout = make_rollout()
    new_log_probs = rollout.old_token_log_probs.clone()

    assert clipped_grpo_surrogate(new_log_probs, rollout, 0.2) == pytest.approx(0.0)

    # An enormous change at padding positions must not affect the objective.
    new_log_probs[~rollout.response_mask] = 100.0
    assert clipped_grpo_surrogate(new_log_probs, rollout, 0.2) == pytest.approx(0.0)

    # Increasing all valid-token probabilities clips positive-advantage samples
    # at 1.2, while negative-advantage samples retain the larger 2.0 ratio.
    new_log_probs[rollout.response_mask] += torch.log(torch.tensor(2.0))
    assert clipped_grpo_surrogate(new_log_probs, rollout, 0.2) == pytest.approx(-0.4)


def test_eos_mask_includes_first_eos_even_when_eos_is_also_padding() -> None:
    responses = torch.tensor([[[5, 1, 1, 1], [5, 6, 7, 7]]])

    mask = response_token_mask(responses, eos_token_ids=(1, 7))

    assert torch.equal(
        mask,
        torch.tensor([[[True, True, False, False], [True, True, True, False]]]),
    )
    assert response_token_mask(responses, eos_token_ids=()).all()


def test_group_loo_and_projected_score_statistics_use_fixed_sequences() -> None:
    _, _, rollout = make_rollout()
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(
        group_leave_one_out_advantages(rewards),
        torch.tensor([[2 / 3, -2 / 3, -2 / 3, 2 / 3]]),
    )
    with pytest.raises(ValueError, match="finite"):
        group_leave_one_out_advantages(torch.tensor([[0.0, float("nan")]]))

    directions = 2
    radius = 0.25
    desired_scores = torch.arange(
        rollout.response_input_ids.numel() * directions, dtype=torch.float32
    ).reshape(*rollout.response_input_ids.shape, directions)
    center = rollout.old_token_log_probs.unsqueeze(-1).expand_as(desired_scores)
    positive = center + radius * desired_scores
    negative = center - radius * desired_scores

    statistics = central_difference_score_statistics(positive, negative, rollout, radius)
    expected_token_scores = desired_scores.masked_fill(~rollout.response_mask.unsqueeze(-1), 0.0)
    expected_completion_scores = expected_token_scores.sum(dim=2)
    lengths = rollout.response_lengths.to(expected_token_scores.dtype)
    expected_fisher = (
        torch.einsum("bgtd,bgte->bgde", expected_token_scores, expected_token_scores)
        / lengths[..., None, None]
    ).mean(dim=(0, 1))
    assert torch.allclose(statistics.directional_token_scores, expected_token_scores)
    assert torch.allclose(statistics.completion_scores, expected_completion_scores)
    assert torch.allclose(
        statistics.gradient,
        (rollout.advantages.unsqueeze(-1) * expected_completion_scores).mean(dim=(0, 1)),
    )
    assert torch.allclose(statistics.fisher, expected_fisher)
