from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import rl_no_backward.perf_benchmark as perf
from rl_no_backward.model import AdapterWrappedLayer, ModelBundle


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        return f"<user>{messages[-1]['content']}<assistant>"

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, Tensor]:
        del prompts, truncation, max_length
        assert return_tensors == "pt" and padding
        return {
            "input_ids": torch.tensor([[0, 2], [2, 6]]),
            "attention_mask": torch.tensor([[0, 1], [1, 1]]),
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        symbols = {3: "A", 4: "B", 5: "C", 6: "D"}
        return "".join(symbols.get(token_id, "") for token_id in token_ids)


class ToyGenerationPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer("features", torch.linspace(-0.5, 0.5, 7))
        self.generate_calls = 0
        self.forward_calls = 0
        self.config = SimpleNamespace(_attn_implementation="toy")

    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        generation_config: object,
    ) -> SimpleNamespace:
        del attention_mask
        self.generate_calls += 1
        return_count = generation_config.num_return_sequences
        repeated = input_ids.repeat_interleave(return_count, dim=0)
        if return_count == 2:
            responses = torch.tensor(
                [[3, 1, 0], [4, 5, 1], [3, 1, 0], [4, 5, 1]],
                device=input_ids.device,
            )
        else:
            # Greedy mode repeats prompts before calling generate.
            responses = torch.tensor(
                [[3, 1, 0], [4, 5, 1], [3, 1, 0], [4, 5, 1]],
                device=input_ids.device,
            )
        return SimpleNamespace(sequences=torch.cat([repeated, responses], dim=1))

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.forward_calls += 1
        context = input_ids.float().unsqueeze(-1) * self.features * 0.02
        return SimpleNamespace(logits=context + self.adapter * self.features)


def make_toy_bundle() -> tuple[ModelBundle, ToyGenerationPolicy, ToyTokenizer]:
    model = ToyGenerationPolicy()
    tokenizer = ToyTokenizer()
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        candidate_token_ids=torch.empty(0, dtype=torch.long),
        adapter_names=["adapter"],
        device=torch.device("cpu"),
        model_name="toy",
    )
    return bundle, model, tokenizer


def toy_config(*, do_sample: bool = True) -> perf.PerformanceBenchmarkConfig:
    return perf.PerformanceBenchmarkConfig(
        model_name="toy",
        backends=("eager",),
        dtype="float32",
        device="cpu",
        adapter_rank=1,
        adapter_layers=1,
        prompt_count=2,
        calibration_prompt_count=2,
        group_size=2,
        max_prompt_tokens=8,
        max_new_tokens=3,
        temperature=1.0,
        do_sample=do_sample,
        scoring_micro_batch_size=2,
        warmup_iterations=1,
        benchmark_iterations=2,
    )


def test_config_locks_eager_reference_and_validates_backend_list() -> None:
    config = perf.PerformanceBenchmarkConfig()
    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.adapter_rank == 8
    assert config.adapter_layers == 4
    assert config.backends == ("eager", "sdpa", "flash_attention_2")

    with pytest.raises(ValueError, match="include eager"):
        perf.PerformanceBenchmarkConfig(backends=("sdpa",))
    with pytest.raises(ValueError, match="duplicates"):
        perf.PerformanceBenchmarkConfig(backends=("eager", "eager"))
    with pytest.raises(ValueError, match="group_size"):
        perf.PerformanceBenchmarkConfig(group_size=1)


@pytest.mark.parametrize("do_sample", [True, False])
def test_generation_and_fixed_teacher_scoring_track_tokens_and_calls(do_sample: bool) -> None:
    bundle, model, tokenizer = make_toy_bundle()
    config = toy_config(do_sample=do_sample)
    prompts = perf._chat_prompts(tokenizer, ("one", "two"), 2)
    prompt_batch = perf._tokenize_prompts(tokenizer, prompts, 8)

    generation_metrics, generation = perf.benchmark_rollout_generation(bundle, prompt_batch, config)

    assert model.generate_calls == 3  # one warmup plus two measured iterations
    assert generation.response_input_ids.shape == (4, 3)
    assert generation.valid_tokens == 10
    assert generation_metrics.processed_tokens == 20
    assert generation_metrics.tokens_per_second > 0
    assert generation_metrics.peak_memory_bytes == 0
    assert generation.completions == ("A", "BC", "A", "BC")

    rollout = perf.build_fixed_scoring_rollout(
        prompt_batch,
        generation,
        prompts,
        config,
        pad_token_id=0,
        eos_token_ids=(1,),
    )
    scoring_metrics, log_probs = perf.benchmark_teacher_forced_scoring(bundle, rollout, config)

    # Four candidates / microbatch two = two model forwards per evaluation.
    assert model.forward_calls == 6
    assert scoring_metrics.processed_tokens == 20
    assert scoring_metrics.tokens_per_second > 0
    assert log_probs.shape == (2, 2, 3)
    assert torch.isfinite(log_probs.masked_select(rollout.response_mask)).all()
    assert torch.equal(log_probs.masked_select(~rollout.response_mask), torch.zeros(2))


def test_generation_and_logprob_equivalence_are_mask_aware() -> None:
    response_ids = torch.tensor([[3, 1, 0], [4, 5, 1]])
    response_mask = torch.tensor([[True, True, False], [True, True, True]])
    reference = perf.GenerationArtifact(response_ids, response_mask, ("A", "BC"), "ref")
    same = perf.GenerationArtifact(response_ids.clone(), response_mask.clone(), ("A", "BC"), "same")
    assert perf.compare_generation_outputs(reference, same).exact_match

    changed_ids = response_ids.clone()
    changed_ids[0, 0] = 6
    changed = perf.GenerationArtifact(changed_ids, response_mask.clone(), ("D", "BC"), "changed")
    comparison = perf.compare_generation_outputs(reference, changed)
    assert not comparison.exact_match
    assert comparison.mismatched_token_count == 1
    assert comparison.matching_sequence_fraction == pytest.approx(0.5)

    old = torch.tensor([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]])
    candidate = old.clone()
    candidate[~response_mask] = 999.0  # padding must not affect equivalence
    equivalent = perf.compare_log_probs(old, candidate, response_mask, atol=1e-6, rtol=0.0)
    assert equivalent.allclose
    assert equivalent.compared_tokens == 5
    candidate[0, 0] += 0.1
    assert not perf.compare_log_probs(old, candidate, response_mask, atol=1e-3, rtol=0.0).allclose


class ToyDecoderLayer(nn.Module):
    def forward(self, hidden_states: Tensor) -> tuple[Tensor]:
        return (hidden_states,)


class ToyDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyDecoderLayer(), ToyDecoderLayer()])


def test_adapter_blueprint_installs_identical_frozen_cores() -> None:
    basis = torch.tensor([[1.0], [0.0]])
    core = torch.tensor([[0.25]])
    blueprint = perf.AdapterBlueprint(
        layer_indices=(1,),
        bases=(basis,),
        cores=(core,),
        scale=0.5,
        fingerprint="fixed",
    )
    first = ToyDecoderModel()
    second = ToyDecoderModel()

    first_names = perf.apply_adapter_blueprint(first, blueprint, torch.device("cpu"))
    second_names = perf.apply_adapter_blueprint(second, blueprint, torch.device("cpu"))

    assert first_names == second_names == ["model.layers.1.adapter.core"]
    assert isinstance(first.model.layers[1], AdapterWrappedLayer)
    assert torch.equal(
        first.model.layers[1].adapter.p_basis, second.model.layers[1].adapter.p_basis
    )
    assert torch.equal(first.model.layers[1].adapter.core, core)
    assert all(not parameter.requires_grad for parameter in first.parameters())


def test_unavailable_backend_is_structured_instead_of_raising() -> None:
    def unavailable() -> object:
        raise ImportError("flash-attn is not installed")

    result = perf.benchmark_backend_safely("flash_attention_2", unavailable)  # type: ignore[arg-type]

    assert isinstance(result, perf.BackendBenchmarkResult)
    assert result.status == "unavailable"
    assert result.error_type == "ImportError"
    assert "flash-attn" in result.error_message


def test_report_writer_emits_json_and_parser_is_h100_ready(tmp_path) -> None:
    config = toy_config()
    unavailable = perf.BackendBenchmarkResult(
        backend="flash_attention_2",
        status="unavailable",
        error_type="ImportError",
        error_message="missing",
    )
    report = perf.PerformanceBenchmarkReport(
        created_at_utc="2026-08-18T00:00:00+00:00",
        config=config,
        system={"device": "cpu"},
        adapter_fingerprint="abc",
        calibration_wall_time_seconds=1.25,
        results=(unavailable,),
    )

    output = perf.write_performance_report(report, tmp_path / "nested" / "perf.json")
    payload = json.loads(output.read_text())
    assert payload["adapter_fingerprint"] == "abc"
    assert payload["results"][0]["status"] == "unavailable"

    arguments = perf.build_parser().parse_args(["--output", str(tmp_path / "h100.json")])
    assert arguments.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert arguments.backends == ["eager", "sdpa", "flash_attention_2"]
    assert arguments.device == "cuda"
    assert arguments.dtype == "bfloat16"
