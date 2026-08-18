from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_no_backward.rollout_provenance import build_rollout_provenance


def _rollout(**overrides):
    values = {
        "prompt_input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
        "prompt_attention_mask": torch.tensor(
            [[False, True, True], [True, True, True]], dtype=torch.bool
        ),
        "response_input_ids": torch.tensor([[[31, 2], [32, 2]], [[41, 42], [43, 2]]]),
        "response_mask": torch.tensor(
            [[[True, True], [True, True]], [[True, True], [True, True]]],
            dtype=torch.bool,
        ),
        "old_token_log_probs": torch.tensor(
            [[[-0.1, -0.2], [-0.3, -0.4]], [[-0.5, -0.6], [-0.7, -0.8]]]
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rollout_provenance_is_reproducible_and_canonical_across_integer_dtypes() -> None:
    source = _rollout()
    first = build_rollout_provenance(source, seed=20_123)
    repeated = build_rollout_provenance(source, seed=20_123)
    int32_ids = build_rollout_provenance(
        _rollout(
            prompt_input_ids=source.prompt_input_ids.to(torch.int32),
            response_input_ids=source.response_input_ids.to(torch.int32),
        ),
        seed=20_123,
    )

    assert first == repeated == int32_ids
    fields = first.as_record_fields()
    assert fields["rollout_seed"] == 20_123
    assert fields["rollout_digest_algorithm"] == "sha256"
    assert fields["rollout_provenance_version"] == "rl-no-backward-rollout-v1"
    assert len(fields["rollout_digest"]) == 64


def test_rollout_provenance_changes_with_tokens_masks_logprobs_and_seed() -> None:
    source = _rollout()
    baseline = build_rollout_provenance(source, seed=20_001)

    changed_ids = source.response_input_ids.clone()
    changed_ids[0, 0, 0] += 1
    token_change = build_rollout_provenance(
        _rollout(response_input_ids=changed_ids), seed=20_001
    )
    changed_mask = source.response_mask.clone()
    changed_mask[0, 0, 1] = False
    mask_change = build_rollout_provenance(_rollout(response_mask=changed_mask), seed=20_001)
    changed_logprobs = source.old_token_log_probs.clone()
    changed_logprobs[0, 0, 0] -= 0.01
    logprob_change = build_rollout_provenance(
        _rollout(old_token_log_probs=changed_logprobs), seed=20_001
    )
    seed_change = build_rollout_provenance(source, seed=20_002)

    assert token_change.token_digest != baseline.token_digest
    assert mask_change.token_digest != baseline.token_digest
    assert token_change.behavior_logprob_digest == baseline.behavior_logprob_digest
    assert logprob_change.token_digest == baseline.token_digest
    assert logprob_change.behavior_logprob_digest != baseline.behavior_logprob_digest
    assert seed_change.token_digest != baseline.token_digest
    assert seed_change.behavior_logprob_digest != baseline.behavior_logprob_digest
    assert len(
        {
            baseline.digest,
            token_change.digest,
            mask_change.digest,
            logprob_change.digest,
            seed_change.digest,
        }
    ) == 5


def test_rollout_provenance_rejects_nonfinite_behavior_logprobs() -> None:
    with pytest.raises(ValueError, match="finite"):
        build_rollout_provenance(
            _rollout(old_token_log_probs=torch.tensor([[[float("nan")]]])),
            seed=1,
        )
