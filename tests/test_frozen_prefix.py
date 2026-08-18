from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from rl_no_backward.frozen_prefix import (
    maybe_build_frozen_prefix_cache,
    qwen_teacher_forcing_logits_to_keep,
    replay_frozen_suffix_logits,
    resolve_frozen_prefix_structure,
)
from rl_no_backward.model import (
    AdapterWrappedLayer,
    ModelBundle,
    ResidualCoreAdapter,
    parameter_vector,
)
from rl_no_backward.sequence_optimizers import (
    BackpropSequenceConfig,
    ForwardSequenceConfig,
    enable_batched_probe_adapters,
    forward_sequence_step,
    make_sequence_grpo_optimizer,
    sequence_grpo_step,
)
from rl_no_backward.sequence_policy import (
    SequenceRolloutBatch,
    attach_frozen_prefix_cache,
    clipped_grpo_surrogate,
    group_leave_one_out_advantages,
    teacher_forced_token_log_probs,
)


class ToyQwenBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.forward_calls = 0
        self.inference_modes: list[bool] = []

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        use_cache: bool,
    ) -> Tensor:
        del position_ids, use_cache
        self.forward_calls += 1
        self.inference_modes.append(torch.is_inference_mode_enabled())
        position_signal = position_embeddings[0].to(hidden_states.dtype)
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        delta = torch.tanh(self.projection(hidden_states) + 0.01 * position_signal)
        return hidden_states + mask * delta


class ToyQwenBackbone(nn.Module):
    def __init__(self, config: SimpleNamespace, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(17, hidden_size)
        self.layers = nn.ModuleList(
            ToyQwenBlock(hidden_size) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        use_cache: bool,
    ) -> SimpleNamespace:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.shape[0], -1)
        position_embeddings = (
            position_ids.to(hidden_states.dtype).unsqueeze(-1),
            (position_ids + 1).to(hidden_states.dtype).unsqueeze(-1),
        )
        for layer in self.layers:
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))


class ToyQwenForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(19)
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=4,
            layer_types=["full_attention"] * 4,
        )
        self.model = ToyQwenBackbone(self.config, hidden_size=6)
        self.lm_head = nn.Linear(6, 17, bias=False)
        self.logits_to_keep_history: list[Tensor | int] = []

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
        logits_to_keep: int | Tensor = 0,
    ) -> SimpleNamespace:
        self.logits_to_keep_history.append(logits_to_keep)
        hidden_states = self.model(
            input_ids,
            attention_mask,
            use_cache=use_cache,
        ).last_hidden_state
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        return SimpleNamespace(logits=self.lm_head(hidden_states[:, indices, :]))


def make_bundle() -> ModelBundle:
    model = ToyQwenForCausalLM().eval()
    model.requires_grad_(False)
    layers = model.model.layers
    basis = torch.linalg.qr(torch.randn(6, 2), mode="reduced").Q
    for index in (2, 3):
        layers[index] = AdapterWrappedLayer(
            layers[index],
            ResidualCoreAdapter(basis, basis, scale=0.3),
        )
    adapter_names = [
        name for name, _parameter in model.named_parameters() if name.endswith("adapter.core")
    ]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in adapter_names)
    return ModelBundle(
        model=model,
        tokenizer=None,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=adapter_names,
        device=torch.device("cpu"),
        model_name="toy-qwen",
    )


def make_rollout(bundle: ModelBundle, *, cached: bool) -> SequenceRolloutBatch:
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rollout = SequenceRolloutBatch(
        prompts=("a", "b"),
        completions=(("x", "y"), ("z", "w")),
        prompt_input_ids=torch.tensor([[1, 2], [3, 4]]),
        prompt_attention_mask=torch.tensor([[True, True], [False, True]]),
        response_input_ids=torch.tensor([[[5, 6, 2], [7, 8, 2]], [[9, 2, 0], [10, 11, 2]]]),
        response_mask=torch.tensor(
            [
                [[True, True, True], [True, True, True]],
                [[True, True, False], [True, True, True]],
            ]
        ),
        old_token_log_probs=torch.zeros(2, 2, 3),
        rewards=rewards,
        advantages=group_leave_one_out_advantages(rewards),
        sampling_temperature=0.8,
        pad_token_id=0,
        eos_token_ids=(2,),
    )
    with torch.no_grad():
        old = teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2)
    rollout = replace(rollout, old_token_log_probs=old.detach())
    return attach_frozen_prefix_cache(bundle, rollout) if cached else rollout


def _inner_blocks(bundle: ModelBundle) -> list[ToyQwenBlock]:
    blocks: list[ToyQwenBlock] = []
    for layer in bundle.model.model.layers:
        block = layer.layer if isinstance(layer, AdapterWrappedLayer) else layer
        assert isinstance(block, ToyQwenBlock)
        blocks.append(block)
    return blocks


def test_cached_suffix_logits_and_log_probs_match_full_qwen_path() -> None:
    bundle = make_bundle()
    rollout = make_rollout(bundle, cached=False)
    names_before = tuple(name for name, _parameter in bundle.model.named_parameters())
    indices = qwen_teacher_forcing_logits_to_keep(bundle, 2, 3)
    assert indices is not None
    full_logits = bundle.model(
        input_ids=rollout.flat_input_ids,
        attention_mask=rollout.flat_attention_mask,
        use_cache=False,
        logits_to_keep=indices,
    ).logits

    build = maybe_build_frozen_prefix_cache(
        bundle,
        rollout.flat_input_ids,
        rollout.flat_attention_mask,
    )
    assert build.fallback_reason is None
    assert build.cache is not None
    cached_logits = replay_frozen_suffix_logits(
        bundle,
        build.cache,
        logits_to_keep=indices,
    )
    torch.testing.assert_close(cached_logits, full_logits, atol=0, rtol=0)

    cached_rollout = replace(rollout, frozen_prefix_cache=build.cache)
    full_log_probs = teacher_forced_token_log_probs(bundle, rollout, micro_batch_size=2)
    cached_log_probs = teacher_forced_token_log_probs(bundle, cached_rollout, micro_batch_size=2)
    torch.testing.assert_close(cached_log_probs, full_log_probs, atol=0, rtol=0)
    assert tuple(name for name, _parameter in bundle.model.named_parameters()) == names_before
    assert isinstance(bundle.model.logits_to_keep_history[-1], Tensor)
    assert bundle.model.logits_to_keep_history[-1].tolist() == [1, 2, 3]


def test_prefix_cache_uses_live_model_device_not_an_unresolved_device_alias() -> None:
    bundle = replace(make_bundle(), device=torch.device("cpu:0"))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    result = maybe_build_frozen_prefix_cache(bundle, input_ids, attention_mask)

    assert result.cache is not None, result.fallback_reason
    assert result.cache.hidden_states.device == bundle.trainable_parameters[0].device


def test_real_transformers_qwen_structure_replays_exact_selected_logits() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(29)
    config = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.requires_grad_(False)
    basis = torch.linalg.qr(torch.randn(16, 2), mode="reduced").Q
    for index in (2, 3):
        model.model.layers[index] = AdapterWrappedLayer(
            model.model.layers[index],
            ResidualCoreAdapter(basis, basis, scale=0.2),
        )
    adapter_names = [
        name for name, _parameter in model.named_parameters() if name.endswith("adapter.core")
    ]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in adapter_names)
    bundle = ModelBundle(
        model=model,
        tokenizer=None,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=adapter_names,
        device=torch.device("cpu"),
        model_name="random-qwen",
    )
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]])
    attention_mask = torch.tensor(
        [[False, True, True, True, True, True, True], [True] * 7]
    )
    indices = qwen_teacher_forcing_logits_to_keep(bundle, prompt_width=3, response_width=4)
    assert indices is not None and indices.tolist() == [2, 3, 4, 5]
    with torch.inference_mode():
        full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=indices,
        ).logits
    build = maybe_build_frozen_prefix_cache(bundle, input_ids, attention_mask)
    assert build.cache is not None, build.fallback_reason
    with torch.inference_mode():
        cached = replay_frozen_suffix_logits(
            bundle,
            build.cache,
            logits_to_keep=indices,
        )
    torch.testing.assert_close(cached, full, atol=0, rtol=0)
    with torch.inference_mode():
        repeated_full = model(
            input_ids=input_ids[:1].repeat(2, 1),
            attention_mask=attention_mask[:1].repeat(2, 1),
            use_cache=False,
            logits_to_keep=indices,
        ).logits
        repeated_cached = replay_frozen_suffix_logits(
            bundle,
            build.cache,
            start=0,
            stop=1,
            repeats=2,
            logits_to_keep=indices,
        )
    # Repeating one row changes the CPU GEMM batch shape. Some PyTorch/BLAS
    # builds then differ by a few float32 ulps even though the single-row
    # cached path above is bit-exact.
    torch.testing.assert_close(repeated_cached, repeated_full, atol=2e-8, rtol=1e-5)


def test_cached_suffix_matches_full_gradient_and_streaming_update() -> None:
    gradient_bundle = make_bundle()
    full_rollout = make_rollout(gradient_bundle, cached=False)
    cached_rollout = attach_frozen_prefix_cache(gradient_bundle, full_rollout)

    for parameter in gradient_bundle.trainable_parameters:
        parameter.grad = None
    full_surrogate = clipped_grpo_surrogate(
        teacher_forced_token_log_probs(gradient_bundle, full_rollout, micro_batch_size=2),
        full_rollout,
        0.2,
    )
    (-full_surrogate).backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in gradient_bundle.trainable_parameters]

    for parameter in gradient_bundle.trainable_parameters:
        parameter.grad = None
    cached_surrogate = clipped_grpo_surrogate(
        teacher_forced_token_log_probs(gradient_bundle, cached_rollout, micro_batch_size=2),
        cached_rollout,
        0.2,
    )
    (-cached_surrogate).backward()
    cached_gradients = [
        parameter.grad.detach().clone() for parameter in gradient_bundle.trainable_parameters
    ]
    for cached_gradient, full_gradient in zip(cached_gradients, full_gradients, strict=True):
        torch.testing.assert_close(cached_gradient, full_gradient, atol=0, rtol=0)

    full_bundle = make_bundle()
    cached_bundle = make_bundle()
    full_update_rollout = make_rollout(full_bundle, cached=False)
    cached_update_rollout = make_rollout(cached_bundle, cached=True)
    config = BackpropSequenceConfig(
        learning_rate=0.01,
        epochs_per_rollout=1,
        scoring_micro_batch_size=2,
        use_streaming_backward=True,
    )
    full_result = sequence_grpo_step(
        full_bundle,
        full_update_rollout,
        make_sequence_grpo_optimizer(full_bundle, config),
        config,
    )
    cached_result = sequence_grpo_step(
        cached_bundle,
        cached_update_rollout,
        make_sequence_grpo_optimizer(cached_bundle, config),
        config,
    )
    torch.testing.assert_close(
        parameter_vector(cached_bundle),
        parameter_vector(full_bundle),
        atol=1e-7,
        rtol=1e-6,
    )
    assert full_result.full_prefix_calls == full_result.forward_calls
    assert full_result.suffix_calls == full_result.forward_calls
    assert cached_result.full_prefix_calls == 0
    assert cached_result.suffix_calls == cached_result.forward_calls


def test_cached_fused_forward_only_replays_suffix_under_inference_mode() -> None:
    full_bundle = make_bundle()
    cached_bundle = deepcopy(full_bundle)
    enable_batched_probe_adapters(full_bundle)
    enable_batched_probe_adapters(cached_bundle)
    full_rollout = make_rollout(full_bundle, cached=False)
    cached_rollout = make_rollout(cached_bundle, cached=True)
    for block in _inner_blocks(cached_bundle):
        block.forward_calls = 0
        block.inference_modes.clear()
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
    full_result = forward_sequence_step(
        full_bundle,
        full_rollout,
        torch.Generator().manual_seed(41),
        config,
    )
    cached_result = forward_sequence_step(
        cached_bundle,
        cached_rollout,
        torch.Generator().manual_seed(41),
        config,
    )

    assert cached_result.full_prefix_calls == 0
    assert cached_result.suffix_calls == cached_result.forward_calls
    assert full_result.full_prefix_calls == full_result.forward_calls
    assert full_result.suffix_calls == full_result.forward_calls
    assert cached_result.projected_gradient_norm == pytest.approx(
        full_result.projected_gradient_norm, rel=1e-6, abs=1e-7
    )
    assert cached_result.derivative_variance == pytest.approx(
        full_result.derivative_variance, rel=1e-6, abs=1e-7
    )
    blocks = _inner_blocks(cached_bundle)
    assert [block.forward_calls for block in blocks[:2]] == [0, 0]
    assert all(block.inference_modes and all(block.inference_modes) for block in blocks[2:])


def test_structure_guards_cleanly_reject_unsupported_layouts() -> None:
    wrong_family = make_bundle()
    wrong_family.model.config.model_type = "llama"
    structure, reason = resolve_frozen_prefix_structure(wrong_family)
    assert structure is None
    assert reason is not None and "Qwen" in reason

    noncontiguous = make_bundle()
    noncontiguous.model.model.layers[3] = noncontiguous.model.model.layers[3].layer
    structure, reason = resolve_frozen_prefix_structure(noncontiguous)
    assert structure is None
    assert reason is not None and "contiguous" in reason

    mixed_masks = make_bundle()
    mixed_masks.model.config.layer_types[-1] = "sliding_attention"
    structure, reason = resolve_frozen_prefix_structure(mixed_masks)
    assert structure is None
    assert reason is not None and "layer types" in reason

    compiled = make_bundle()
    compiled.model._rl_no_backward_forward_compiled = True
    structure, reason = resolve_frozen_prefix_structure(compiled)
    assert structure is None
    assert reason is not None and "compiled" in reason

    extra_trainable = make_bundle()
    extra_trainable.model.model.embed_tokens.weight.requires_grad_(True)
    structure, reason = resolve_frozen_prefix_structure(extra_trainable)
    assert structure is None
    assert reason is not None and "only suffix adapter cores" in reason
