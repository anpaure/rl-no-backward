from __future__ import annotations

import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_no_backward.matched_lora_diagnostics as diagnostics_module
import rl_no_backward.matched_lora_forward_only as forward_only_module
from rl_no_backward.matched_grpo_objective import (
    MatchedGRPOObjectiveConfig,
    trl_group_standardized_advantages,
)
from rl_no_backward.matched_lora_diagnostics import (
    GradientAgreementThresholds,
    fixed_rollout_finite_difference_diagnostic,
    load_fixed_rollout_cache,
    save_fixed_rollout_cache,
)
from rl_no_backward.model import ModelBundle, parameter_vector
from rl_no_backward.sequence_policy import SequenceRolloutBatch, teacher_forced_token_log_probs


class _ToyLoRAPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.linspace(-0.3, 0.3, 7), requires_grad=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 1, bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.tensor([[0.02, -0.01, 0.03]]))
        self.register_buffer(
            "features",
            torch.randn(7, 3, generator=torch.Generator().manual_seed(9)),
        )

    def forward(self, *, input_ids: Tensor, attention_mask: Tensor, use_cache: bool):
        del attention_mask, use_cache
        adapter = self.lora_A["default"].weight.reshape(-1)
        context = 1.0 + input_ids.float().unsqueeze(-1) * 0.02
        return SimpleNamespace(logits=self.base + context * (self.features @ adapter))


def _fixture() -> tuple[ModelBundle, SequenceRolloutBatch, Tensor]:
    model = _ToyLoRAPolicy()
    bundle = ModelBundle(
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["lora_A.default.weight"],
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
    rollout = replace(rollout, old_token_log_probs=old.detach())
    sampler_offsets = torch.tensor(
        [
            [[0.01, 0.04, 0.00], [0.02, 0.05, 0.08]],
            [[0.03, 0.06, 0.09], [0.07, 0.00, 0.00]],
        ]
    )
    return bundle, rollout, (old - sampler_offsets).detach()


def test_multi_mu_diagnostic_matches_exact_gradient_and_restores_everything() -> None:
    bundle, rollout, sampler = _fixture()
    parameter = bundle.trainable_parameters[0]
    parameter.grad = torch.full_like(parameter, 0.125)
    original_gradient = parameter.grad.clone()
    original_vector = parameter_vector(bundle).clone()
    original_training = bundle.model.training
    report = fixed_rollout_finite_difference_diagnostic(
        bundle,
        rollout,
        sampler,
        MatchedGRPOObjectiveConfig(
            inference_correction_mode="token_truncate",
            inference_ratio_min=0.1,
            inference_ratio_max=3.0,
        ),
        directions=2,
        basis_seed=70_004,
        mu_values=(1.0e-4, 5.0e-4, 1.0e-3),
        scoring_micro_batch_size=4,
        prompt_groups_per_micro_batch=1,
        thresholds=GradientAgreementThresholds(
            minimum_cosine_similarity=0.999,
            maximum_relative_l2_error=0.02,
            minimum_passing_mu_count=3,
            required_mu=5.0e-4,
        ),
    )

    assert report["passed"]
    assert report["basis_seed"] == 70_004
    assert report["passing_mu_count"] == 3
    assert report["required_mu_passed"]
    assert report["exact_gradient_full_l2_norm"] > 0
    assert report["exact_gradient_missing_parameter_count"] == 0
    assert report["finite_difference_policy_evaluations"] == 12
    assert all(comparison["passed"] for comparison in report["comparisons"])
    assert all(
        comparison["coordinate_estimator"] == "center_policy_token_score_statistics"
        for comparison in report["comparisons"]
    )
    assert all(
        comparison["maximum_absolute_error"] is not None for comparison in report["comparisons"]
    )
    assert report["integrity"]["adapter_state_restored"]
    assert report["integrity"]["frozen_base_unchanged"]
    assert report["integrity"]["gradient_buffers_restored"]
    torch.testing.assert_close(parameter_vector(bundle), original_vector, rtol=0, atol=0)
    torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)
    assert bundle.model.training is original_training


def test_fixed_rollout_cache_round_trip_and_tensor_receipts(tmp_path) -> None:
    _, rollout, sampler = _fixture()
    path = save_fixed_rollout_cache(
        tmp_path / "rollout.pt",
        rollout,
        sampler,
        {"seed": 0, "step": 1, "adapter_state_digest": "abc"},
    )
    loaded, loaded_sampler, metadata = load_fixed_rollout_cache(path, device="cpu")
    assert metadata == {"seed": 0, "step": 1, "adapter_state_digest": "abc"}
    assert loaded.prompts == rollout.prompts
    assert loaded.completions == rollout.completions
    torch.testing.assert_close(loaded.response_input_ids, rollout.response_input_ids)
    torch.testing.assert_close(loaded.old_token_log_probs, rollout.old_token_log_probs)
    torch.testing.assert_close(loaded_sampler, sampler)

    tampered = torch.load(path, map_location="cpu", weights_only=True)
    tampered["response_input_ids"][0, 0, 0] += 1
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="tensors do not match"):
        load_fixed_rollout_cache(tampered_path, device="cpu")


def test_diagnostic_rejects_bad_radius_and_impossible_gate() -> None:
    bundle, rollout, sampler = _fixture()
    with pytest.raises(ValueError, match="positive and finite"):
        fixed_rollout_finite_difference_diagnostic(
            bundle,
            rollout,
            sampler,
            MatchedGRPOObjectiveConfig(),
            directions=2,
            basis_seed=7,
            mu_values=(0.0,),
        )
    with pytest.raises(ValueError, match="exceeds the number of radii"):
        fixed_rollout_finite_difference_diagnostic(
            bundle,
            rollout,
            sampler,
            MatchedGRPOObjectiveConfig(),
            directions=2,
            basis_seed=7,
            mu_values=(1.0e-3,),
            thresholds=GradientAgreementThresholds(minimum_passing_mu_count=2),
        )
    with pytest.raises(ValueError, match="required_mu must be present"):
        fixed_rollout_finite_difference_diagnostic(
            bundle,
            rollout,
            sampler,
            MatchedGRPOObjectiveConfig(),
            directions=2,
            basis_seed=7,
            mu_values=(0.5, 2.0),
            thresholds=GradientAgreementThresholds(required_mu=1.0),
        )


def test_cli_writes_and_prints_json(monkeypatch, tmp_path, capsys) -> None:
    expected = {"schema": "test", "passed": True, "comparisons": []}
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(diagnostics_module, "run_real_model_diagnostic", fake_run)
    destination = tmp_path / "report.json"
    diagnostics_module.main(
        [
            "--config",
            "config.yaml",
            "--output",
            str(destination),
            "--rollout-cache",
            str(tmp_path / "rollout.pt"),
            "--mu",
            "0.5",
            "1",
            "2",
        ]
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == expected
    assert captured["mu_values"] == [0.5, 1.0, 2.0]
    assert captured["rollout_cache"] == tmp_path / "rollout.pt"


def test_forward_only_module_never_imports_reverse_mode_diagnostic() -> None:
    source = inspect.getsource(forward_only_module)
    assert "matched_lora_diagnostics" not in source
    assert "matched_lora_backprop" not in source


def test_unindexed_cuda_bundle_device_accepts_cuda_zero_parameters() -> None:
    assert diagnostics_module._device_matches_bundle(
        torch.device("cuda:0"),
        torch.device("cuda"),
    )
    assert not diagnostics_module._device_matches_bundle(
        torch.device("cuda:1"),
        torch.device("cuda:0"),
    )
