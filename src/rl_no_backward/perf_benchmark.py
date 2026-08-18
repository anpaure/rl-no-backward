"""Attention-backend performance harness for the Qwen sequence-RL policy.

The eager backend calibrates one activation-basis adapter blueprint.  That
exact blueprint (fixed bases and adapter-core values) is then installed into
fresh copies of the same Hugging Face checkpoint loaded with eager, SDPA, and
FlashAttention-2.  Generation is timed independently, while teacher-forced
scoring always uses the same eager-generated candidate sequences so backend
log-probability comparisons are meaningful even when sampled outputs diverge.

Typical H100 invocation after syncing dependencies::

    uv run python -m rl_no_backward.perf_benchmark \
        --output artifacts/attention_backends.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .model import (
    AdapterWrappedLayer,
    ModelBundle,
    ResidualCoreAdapter,
    _decoder_layers,
    calibrate_activation_bases,
)
from .sequence_policy import (
    SequenceRolloutBatch,
    response_token_mask,
    teacher_forced_token_log_probs,
)

AttentionBackend = Literal["eager", "sdpa", "flash_attention_2"]
SUPPORTED_ATTENTION_BACKENDS: tuple[AttentionBackend, ...] = (
    "eager",
    "sdpa",
    "flash_attention_2",
)

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "A shop sold 18 notebooks on Monday and 27 on Tuesday. How many in total?",
    "If 5 identical boxes contain 60 marbles altogether, how many are in each box?",
    "A train travels 72 kilometers per hour for 2.5 hours. How far does it travel?",
    "Mina had 45 dollars, spent 17, and then earned 9. How many dollars does she have?",
    "What is the least positive integer divisible by both 12 and 18?",
    "A rectangle has length 11 and width 7. Find its area and explain briefly.",
    "Solve for x: 3x + 8 = 29.",
    "A fair die is rolled once. What is the probability of rolling an even number?",
)


@dataclass(frozen=True, slots=True)
class PerformanceBenchmarkConfig:
    """Configuration matching the Qwen-1.5B rank-8, last-four-layer adapter."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str | None = None
    backends: tuple[AttentionBackend, ...] = SUPPORTED_ATTENTION_BACKENDS
    dtype: str = "bfloat16"
    device: str = "cuda"
    adapter_rank: int = 8
    adapter_layers: int = 4
    adapter_scale: float = 1.0
    adapter_core_std: float = 0.0
    prompt_count: int = 4
    calibration_prompt_count: int = 8
    group_size: int = 4
    max_prompt_tokens: int = 256
    max_new_tokens: int = 256
    temperature: float = 0.8
    do_sample: bool = True
    scoring_micro_batch_size: int = 4
    warmup_iterations: int = 1
    benchmark_iterations: int = 3
    seed: int = 0
    logprob_atol: float = 5e-3
    logprob_rtol: float = 1e-3

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be a non-empty string or None")
        if not self.backends or "eager" not in self.backends:
            raise ValueError("backends must include eager as the equivalence reference")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must not contain duplicates")
        unsupported = sorted(set(self.backends) - set(SUPPORTED_ATTENTION_BACKENDS))
        if unsupported:
            raise ValueError(f"unsupported attention backends: {unsupported}")
        for name in (
            "adapter_rank",
            "adapter_layers",
            "prompt_count",
            "calibration_prompt_count",
            "group_size",
            "max_prompt_tokens",
            "max_new_tokens",
            "scoring_micro_batch_size",
            "benchmark_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if (
            isinstance(self.warmup_iterations, bool)
            or not isinstance(self.warmup_iterations, int)
            or self.warmup_iterations < 0
        ):
            raise ValueError("warmup_iterations must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, allow_zero in (
            ("adapter_scale", False),
            ("adapter_core_std", True),
            ("temperature", False),
            ("logprob_atol", True),
            ("logprob_rtol", True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            valid = value >= 0 if allow_zero else value > 0
            if not valid or not math.isfinite(float(value)):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier} and finite")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be boolean")
        if not hasattr(torch, self.dtype):
            raise ValueError(f"unknown torch dtype {self.dtype!r}")


@dataclass(frozen=True, slots=True)
class AdapterBlueprint:
    """CPU copy of the exact adapter bases and cores shared by all backends."""

    layer_indices: tuple[int, ...]
    bases: tuple[Tensor, ...]
    cores: tuple[Tensor, ...]
    scale: float
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    token_unit: str
    iterations: int
    processed_tokens: int
    wall_time_seconds: float
    mean_iteration_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int
    peak_memory_increment_bytes: int


@dataclass(frozen=True, slots=True)
class GenerationArtifact:
    response_input_ids: Tensor
    response_mask: Tensor
    completions: tuple[str, ...]
    digest: str

    @property
    def valid_tokens(self) -> int:
        return int(self.response_mask.sum().item())


@dataclass(frozen=True, slots=True)
class OutputEquivalence:
    exact_match: bool
    mask_match: bool
    mismatched_token_count: int
    matching_sequence_fraction: float


@dataclass(frozen=True, slots=True)
class LogprobEquivalence:
    allclose: bool
    compared_tokens: int
    max_absolute_difference: float
    mean_absolute_difference: float


@dataclass(frozen=True, slots=True)
class BackendBenchmarkResult:
    backend: str
    status: Literal["available", "unavailable"]
    resolved_backend: str | None = None
    load_wall_time_seconds: float | None = None
    generation: PhaseMetrics | None = None
    teacher_forced_scoring: PhaseMetrics | None = None
    output_digest: str | None = None
    completion_samples: tuple[str, ...] = ()
    output_equivalence: OutputEquivalence | None = None
    logprob_equivalence: LogprobEquivalence | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class PerformanceBenchmarkReport:
    created_at_utc: str
    config: PerformanceBenchmarkConfig
    system: dict[str, Any]
    adapter_fingerprint: str | None
    calibration_wall_time_seconds: float | None
    results: tuple[BackendBenchmarkResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class _GeneratedDeviceBatch:
    response_input_ids: Tensor
    response_mask: Tensor


@dataclass(frozen=True, slots=True)
class _BackendArtifacts:
    result: BackendBenchmarkResult
    generation: GenerationArtifact
    scoring_log_probs: Tensor


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _system_metadata(device: torch.device) -> dict[str, Any]:
    import transformers

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return metadata


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_allocated(device: torch.device) -> int:
    return int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@contextmanager
def _forked_seed(device: torch.device, seed: int) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        yield


def _measure_operation(
    operation: Callable[[], Any],
    token_counter: Callable[[Any], int],
    *,
    iterations: int,
    device: torch.device,
    token_unit: str,
) -> tuple[PhaseMetrics, Any]:
    """Measure synchronized wall time and CUDA peak allocation for an operation."""

    _sync(device)
    baseline_memory = _memory_allocated(device)
    _reset_peak_memory(device)
    total_seconds = 0.0
    processed_tokens = 0
    final_output: Any = None
    for _ in range(iterations):
        _sync(device)
        started = time.perf_counter()
        final_output = operation()
        _sync(device)
        total_seconds += time.perf_counter() - started
        processed_tokens += int(token_counter(final_output))
    peak_memory = _peak_memory(device)
    metrics = PhaseMetrics(
        token_unit=token_unit,
        iterations=iterations,
        processed_tokens=processed_tokens,
        wall_time_seconds=total_seconds,
        mean_iteration_seconds=total_seconds / iterations,
        tokens_per_second=processed_tokens / total_seconds if total_seconds > 0 else float("inf"),
        peak_memory_bytes=peak_memory,
        peak_memory_increment_bytes=max(0, peak_memory - baseline_memory),
    )
    return metrics, final_output


def _chat_prompts(
    tokenizer: object,
    questions: Sequence[str],
    count: int,
) -> tuple[str, ...]:
    if not questions or any(
        not isinstance(question, str) or not question.strip() for question in questions
    ):
        raise ValueError("questions must be a non-empty sequence of non-empty strings")
    prompts: list[str] = []
    for index in range(count):
        question = questions[index % len(questions)].strip()
        messages = [
            {
                "role": "system",
                "content": "Solve the mathematics problem carefully and state a final answer.",
            },
            {"role": "user", "content": question},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"Solve carefully.\n\nProblem:\n{question}\n\nSolution:\n"
        if not isinstance(rendered, str):
            raise TypeError("tokenizer chat template must return a string")
        prompts.append(rendered)
    return tuple(prompts)


def _tokenize_prompts(
    tokenizer: object,
    prompts: Sequence[str],
    max_prompt_tokens: int,
) -> dict[str, Tensor]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].detach().cpu()
    attention_mask = encoded["attention_mask"].bool().detach().cpu()
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("tokenizer returned incompatible prompt tensors")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _resolve_special_token_ids(tokenizer: object, model: nn.Module) -> tuple[int, tuple[int, ...]]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos_ids: tuple[int, ...] = ()
    elif isinstance(eos, int):
        eos_ids = (eos,)
    else:
        eos_ids = tuple(int(token_id) for token_id in eos)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad is None and eos_ids:
        pad = eos_ids[0]
    if pad is None:
        raise ValueError("generation requires a pad or EOS token id")
    return int(pad), eos_ids


def _generation_config(
    config: PerformanceBenchmarkConfig,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
) -> object:
    from transformers import GenerationConfig

    eos: int | list[int] | None
    if not eos_token_ids:
        eos = None
    elif len(eos_token_ids) == 1:
        eos = eos_token_ids[0]
    else:
        eos = list(eos_token_ids)
    kwargs: dict[str, Any] = {
        "do_sample": config.do_sample,
        # Greedy decoding cannot request multiple return sequences directly;
        # _generate_once repeats each prompt instead in that mode.
        "num_return_sequences": config.group_size if config.do_sample else 1,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if config.do_sample:
        kwargs.update(temperature=config.temperature, top_k=0, top_p=1.0)
    return GenerationConfig(**kwargs)


def _generate_once(
    bundle: ModelBundle,
    prompt_batch: dict[str, Tensor],
    config: PerformanceBenchmarkConfig,
    generation_config: object,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int,
) -> _GeneratedDeviceBatch:
    input_ids = prompt_batch["input_ids"].to(bundle.device)
    attention_mask = prompt_batch["attention_mask"].to(bundle.device)
    generation_input_ids = input_ids
    generation_attention_mask = attention_mask
    if not config.do_sample:
        generation_input_ids = input_ids.repeat_interleave(config.group_size, dim=0)
        generation_attention_mask = attention_mask.repeat_interleave(config.group_size, dim=0)
    with _forked_seed(bundle.device, config.seed), torch.inference_mode():
        output = bundle.model.generate(
            input_ids=generation_input_ids,
            attention_mask=generation_attention_mask,
            generation_config=generation_config,
        )
    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_width = input_ids.shape[1]
    expected = input_ids.shape[0] * config.group_size
    if sequences.ndim != 2 or sequences.shape[0] != expected:
        raise ValueError("generate returned an incompatible sequence tensor")
    expected_prefix = input_ids.repeat_interleave(config.group_size, dim=0)
    if not torch.equal(sequences[:, :prompt_width], expected_prefix):
        raise ValueError("benchmark supports decoder-only generation that preserves the prompt")
    response_ids = sequences[:, prompt_width:]
    mask = response_token_mask(response_ids, eos_token_ids)
    response_ids = response_ids.masked_fill(~mask, pad_token_id)
    return _GeneratedDeviceBatch(response_input_ids=response_ids, response_mask=mask)


def _tensor_digest(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def benchmark_rollout_generation(
    bundle: ModelBundle,
    prompt_batch: dict[str, Tensor],
    config: PerformanceBenchmarkConfig,
) -> tuple[PhaseMetrics, GenerationArtifact]:
    """Benchmark decoder generation without including teacher-forced rescoring."""

    pad_token_id, eos_token_ids = _resolve_special_token_ids(bundle.tokenizer, bundle.model)
    generation_config = _generation_config(config, pad_token_id, eos_token_ids)

    operation = lambda: _generate_once(
        bundle,
        prompt_batch,
        config,
        generation_config,
        eos_token_ids,
        pad_token_id,
    )
    for _ in range(config.warmup_iterations):
        operation()
    metrics, device_batch = _measure_operation(
        operation,
        lambda batch: int(batch.response_mask.sum().item()),
        iterations=config.benchmark_iterations,
        device=bundle.device,
        token_unit="generated_response_tokens_including_eos",
    )
    response_ids = device_batch.response_input_ids.detach().cpu()
    response_mask = device_batch.response_mask.detach().cpu()
    completions = tuple(
        bundle.tokenizer.decode(
            row[mask].tolist(),
            skip_special_tokens=True,
        )
        for row, mask in zip(response_ids, response_mask, strict=True)
    )
    artifact = GenerationArtifact(
        response_input_ids=response_ids,
        response_mask=response_mask,
        completions=completions,
        digest=_tensor_digest(response_ids, response_mask),
    )
    return metrics, artifact


def build_fixed_scoring_rollout(
    prompt_batch: dict[str, Tensor],
    generation: GenerationArtifact,
    prompts: Sequence[str],
    config: PerformanceBenchmarkConfig,
    *,
    pad_token_id: int,
    eos_token_ids: tuple[int, ...],
) -> SequenceRolloutBatch:
    """Build a reward-neutral rollout used only for fixed-candidate rescoring."""

    batch_size = prompt_batch["input_ids"].shape[0]
    expected = batch_size * config.group_size
    if generation.response_input_ids.shape[0] != expected:
        raise ValueError("generation count does not match prompt_count * group_size")
    response_width = generation.response_input_ids.shape[1]
    response_ids = generation.response_input_ids.reshape(
        batch_size, config.group_size, response_width
    )
    response_mask = generation.response_mask.reshape(batch_size, config.group_size, response_width)
    completion_groups = tuple(
        tuple(
            generation.completions[prompt_index * config.group_size + group_index]
            for group_index in range(config.group_size)
        )
        for prompt_index in range(batch_size)
    )
    zeros = torch.zeros(batch_size, config.group_size)
    return SequenceRolloutBatch(
        prompts=tuple(prompts),
        completions=completion_groups,
        prompt_input_ids=prompt_batch["input_ids"],
        prompt_attention_mask=prompt_batch["attention_mask"].bool(),
        response_input_ids=response_ids,
        response_mask=response_mask.bool(),
        old_token_log_probs=torch.zeros_like(response_ids, dtype=torch.float32),
        rewards=zeros,
        advantages=zeros.clone(),
        sampling_temperature=config.temperature,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
    )


def benchmark_teacher_forced_scoring(
    bundle: ModelBundle,
    rollout: SequenceRolloutBatch,
    config: PerformanceBenchmarkConfig,
) -> tuple[PhaseMetrics, Tensor]:
    """Benchmark fixed-candidate response-token log-probability scoring."""

    device_rollout = rollout.to(bundle.device)

    def operation() -> Tensor:
        with torch.inference_mode():
            return teacher_forced_token_log_probs(
                bundle,
                device_rollout,
                micro_batch_size=config.scoring_micro_batch_size,
            )

    for _ in range(config.warmup_iterations):
        operation()
    metrics, log_probs = _measure_operation(
        operation,
        lambda _output: rollout.valid_response_tokens,
        iterations=config.benchmark_iterations,
        device=bundle.device,
        token_unit="teacher_forced_response_tokens_including_eos",
    )
    return metrics, log_probs.detach().float().cpu()


def compare_generation_outputs(
    reference: GenerationArtifact,
    candidate: GenerationArtifact,
) -> OutputEquivalence:
    if reference.response_input_ids.shape != candidate.response_input_ids.shape:
        return OutputEquivalence(False, False, -1, 0.0)
    mask_match = torch.equal(reference.response_mask, candidate.response_mask)
    valid_union = reference.response_mask | candidate.response_mask
    mismatches = reference.response_input_ids.ne(candidate.response_input_ids) & valid_union
    per_sequence_match = (~mismatches).all(dim=-1) & reference.response_mask.eq(
        candidate.response_mask
    ).all(dim=-1)
    mismatch_count = int(mismatches.sum().item())
    exact = mask_match and mismatch_count == 0
    return OutputEquivalence(
        exact_match=exact,
        mask_match=mask_match,
        mismatched_token_count=mismatch_count,
        matching_sequence_fraction=float(per_sequence_match.float().mean().item()),
    )


def compare_log_probs(
    reference: Tensor,
    candidate: Tensor,
    response_mask: Tensor,
    *,
    atol: float,
    rtol: float,
) -> LogprobEquivalence:
    if reference.shape != candidate.shape or reference.shape != response_mask.shape:
        return LogprobEquivalence(False, 0, float("inf"), float("inf"))
    selected_reference = reference.float().masked_select(response_mask)
    selected_candidate = candidate.float().masked_select(response_mask)
    if selected_reference.numel() == 0:
        return LogprobEquivalence(True, 0, 0.0, 0.0)
    differences = (selected_reference - selected_candidate).abs()
    return LogprobEquivalence(
        allclose=bool(torch.allclose(selected_reference, selected_candidate, atol=atol, rtol=rtol)),
        compared_tokens=selected_reference.numel(),
        max_absolute_difference=float(differences.max().item()),
        mean_absolute_difference=float(differences.mean().item()),
    )


def _blueprint_fingerprint(
    indices: Sequence[int], bases: Sequence[Tensor], cores: Sequence[Tensor], scale: float
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"indices": list(indices), "scale": scale}, sort_keys=True).encode())
    for tensor in (*bases, *cores):
        digest.update(_tensor_digest(tensor).encode())
    return digest.hexdigest()


def calibrate_adapter_blueprint(
    model: nn.Module,
    tokenizer: object,
    calibration_prompts: Sequence[str],
    config: PerformanceBenchmarkConfig,
    device: torch.device,
) -> AdapterBlueprint:
    """Calibrate once under eager attention and materialize CPU adapter state."""

    layers = _decoder_layers(model)
    if config.adapter_layers > len(layers):
        raise ValueError("adapter_layers exceeds the decoder layer count")
    indices = tuple(range(len(layers) - config.adapter_layers, len(layers)))
    basis_by_index = calibrate_activation_bases(
        model=model,
        tokenizer=tokenizer,
        prompts=calibration_prompts,
        layer_indices=indices,
        rank=config.adapter_rank,
        device=device,
    )
    bases = tuple(basis_by_index[index].float().cpu().contiguous() for index in indices)
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    cores = tuple(
        (
            torch.randn(config.adapter_rank, config.adapter_rank, generator=generator)
            * config.adapter_core_std
        ).contiguous()
        for _ in indices
    )
    fingerprint = _blueprint_fingerprint(indices, bases, cores, config.adapter_scale)
    return AdapterBlueprint(indices, bases, cores, config.adapter_scale, fingerprint)


def apply_adapter_blueprint(
    model: nn.Module,
    blueprint: AdapterBlueprint,
    device: torch.device,
) -> list[str]:
    """Install identical frozen-basis residual adapters into a fresh base model."""

    layers = _decoder_layers(model)
    if len(blueprint.layer_indices) != len(blueprint.bases) or len(blueprint.bases) != len(
        blueprint.cores
    ):
        raise ValueError("adapter blueprint entries have inconsistent lengths")
    for index, basis, core in zip(
        blueprint.layer_indices, blueprint.bases, blueprint.cores, strict=True
    ):
        if index >= len(layers):
            raise ValueError("adapter blueprint layer index exceeds model depth")
        adapter = ResidualCoreAdapter(
            basis.to(device),
            basis.to(device),
            scale=blueprint.scale,
        ).to(device)
        with torch.no_grad():
            adapter.core.copy_(core.to(device))
        layers[index] = AdapterWrappedLayer(layers[index], adapter)
    adapter_names = [
        name for name, parameter in model.named_parameters() if name.endswith("adapter.core")
    ]
    if len(adapter_names) != len(blueprint.layer_indices):
        raise RuntimeError("installed adapter count does not match blueprint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return adapter_names


def _load_tokenizer(config: PerformanceBenchmarkConfig) -> object:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(
    config: PerformanceBenchmarkConfig,
    backend: AttentionBackend,
    device: torch.device,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=getattr(torch, config.dtype),
        attn_implementation=backend,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _make_bundle(
    model: nn.Module,
    tokenizer: object,
    adapter_names: list[str],
    config: PerformanceBenchmarkConfig,
    device: torch.device,
) -> ModelBundle:
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        candidate_token_ids=torch.empty(0, dtype=torch.long, device=device),
        adapter_names=adapter_names,
        device=device,
        model_name=config.model_name,
    )


def _resolved_backend(model: nn.Module) -> str:
    model_config = getattr(model, "config", None)
    for name in ("_attn_implementation", "_attn_implementation_internal"):
        value = getattr(model_config, name, None)
        if value:
            return str(value)
    return "unknown"


def _unavailable_result(backend: str, error: BaseException) -> BackendBenchmarkResult:
    return BackendBenchmarkResult(
        backend=backend,
        status="unavailable",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def benchmark_backend_safely(
    backend: str,
    runner: Callable[[], _BackendArtifacts],
) -> _BackendArtifacts | BackendBenchmarkResult:
    """Run one backend, converting initialization/runtime failures to a result row."""

    try:
        return runner()
    except Exception as error:  # noqa: BLE001 - availability is the benchmark output
        return _unavailable_result(backend, error)


def _run_loaded_backend(
    backend: AttentionBackend,
    bundle: ModelBundle,
    prompt_batch: dict[str, Tensor],
    prompts: Sequence[str],
    fixed_rollout: SequenceRolloutBatch | None,
    reference_generation: GenerationArtifact | None,
    reference_log_probs: Tensor | None,
    config: PerformanceBenchmarkConfig,
    load_seconds: float,
) -> _BackendArtifacts:
    generation_metrics, generation = benchmark_rollout_generation(bundle, prompt_batch, config)
    pad_token_id, eos_token_ids = _resolve_special_token_ids(bundle.tokenizer, bundle.model)
    scoring_rollout = fixed_rollout or build_fixed_scoring_rollout(
        prompt_batch,
        generation,
        prompts,
        config,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_token_ids,
    )
    scoring_metrics, log_probs = benchmark_teacher_forced_scoring(bundle, scoring_rollout, config)
    output_equivalence = (
        OutputEquivalence(True, True, 0, 1.0)
        if reference_generation is None
        else compare_generation_outputs(reference_generation, generation)
    )
    logprob_equivalence = (
        LogprobEquivalence(True, scoring_rollout.valid_response_tokens, 0.0, 0.0)
        if reference_log_probs is None
        else compare_log_probs(
            reference_log_probs,
            log_probs,
            scoring_rollout.response_mask,
            atol=config.logprob_atol,
            rtol=config.logprob_rtol,
        )
    )
    result = BackendBenchmarkResult(
        backend=backend,
        status="available",
        resolved_backend=_resolved_backend(bundle.model),
        load_wall_time_seconds=load_seconds,
        generation=generation_metrics,
        teacher_forced_scoring=scoring_metrics,
        output_digest=generation.digest,
        completion_samples=generation.completions[:2],
        output_equivalence=output_equivalence,
        logprob_equivalence=logprob_equivalence,
    )
    return _BackendArtifacts(result, generation, log_probs)


def run_attention_backend_benchmark(
    config: PerformanceBenchmarkConfig | None = None,
    *,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
) -> PerformanceBenchmarkReport:
    """Run eager/SDPA/FlashAttention-2 sequentially on one exact adapter state."""

    active_config = config or PerformanceBenchmarkConfig()
    device = torch.device(active_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer = _load_tokenizer(active_config)
    prompts = _chat_prompts(tokenizer, questions, active_config.prompt_count)
    calibration_prompts = _chat_prompts(
        tokenizer, questions, active_config.calibration_prompt_count
    )
    prompt_batch = _tokenize_prompts(tokenizer, prompts, active_config.max_prompt_tokens)

    results_by_backend: dict[str, BackendBenchmarkResult] = {}
    blueprint: AdapterBlueprint | None = None
    calibration_seconds: float | None = None
    reference_generation: GenerationArtifact | None = None
    reference_log_probs: Tensor | None = None
    fixed_rollout: SequenceRolloutBatch | None = None

    eager_model: nn.Module | None = None
    eager_bundle: ModelBundle | None = None
    eager_load_started = time.perf_counter()
    try:
        eager_model = _load_base_model(active_config, "eager", device)
        eager_base_load_seconds = time.perf_counter() - eager_load_started
        calibration_started = time.perf_counter()
        blueprint = calibrate_adapter_blueprint(
            eager_model,
            tokenizer,
            calibration_prompts,
            active_config,
            device,
        )
        calibration_seconds = time.perf_counter() - calibration_started
        adapter_install_started = time.perf_counter()
        adapter_names = apply_adapter_blueprint(eager_model, blueprint, device)
        eager_adapter_install_seconds = time.perf_counter() - adapter_install_started
        eager_bundle = _make_bundle(eager_model, tokenizer, adapter_names, active_config, device)
        eager_load_seconds = eager_base_load_seconds + eager_adapter_install_seconds
        eager_run = _run_loaded_backend(
            "eager",
            eager_bundle,
            prompt_batch,
            prompts,
            None,
            None,
            None,
            active_config,
            eager_load_seconds,
        )
        results_by_backend["eager"] = eager_run.result
        reference_generation = eager_run.generation
        reference_log_probs = eager_run.scoring_log_probs
        pad_token_id, eos_token_ids = _resolve_special_token_ids(tokenizer, eager_model)
        fixed_rollout = build_fixed_scoring_rollout(
            prompt_batch,
            reference_generation,
            prompts,
            active_config,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )
    except Exception as error:  # noqa: BLE001 - preserve a structured failed report
        results_by_backend["eager"] = _unavailable_result("eager", error)
    finally:
        eager_bundle = None
        eager_model = None
        _release_device_memory(device)

    if blueprint is not None and fixed_rollout is not None:
        for backend in active_config.backends:
            if backend == "eager":
                continue

            def run_backend(selected_backend: AttentionBackend = backend) -> _BackendArtifacts:
                loaded_model: nn.Module | None = None
                bundle: ModelBundle | None = None
                load_started = time.perf_counter()
                try:
                    loaded_model = _load_base_model(active_config, selected_backend, device)
                    adapter_names = apply_adapter_blueprint(loaded_model, blueprint, device)
                    bundle = _make_bundle(
                        loaded_model, tokenizer, adapter_names, active_config, device
                    )
                    load_seconds = time.perf_counter() - load_started
                    return _run_loaded_backend(
                        selected_backend,
                        bundle,
                        prompt_batch,
                        prompts,
                        fixed_rollout,
                        reference_generation,
                        reference_log_probs,
                        active_config,
                        load_seconds,
                    )
                finally:
                    bundle = None
                    loaded_model = None
                    _release_device_memory(device)

            outcome = benchmark_backend_safely(backend, run_backend)
            results_by_backend[backend] = (
                outcome if isinstance(outcome, BackendBenchmarkResult) else outcome.result
            )
    else:
        reference_error = RuntimeError("eager reference initialization failed")
        for backend in active_config.backends:
            if backend != "eager":
                results_by_backend[backend] = _unavailable_result(backend, reference_error)

    ordered_results = tuple(results_by_backend[backend] for backend in active_config.backends)
    return PerformanceBenchmarkReport(
        created_at_utc=datetime.now(UTC).isoformat(),
        config=active_config,
        system=_system_metadata(device),
        adapter_fingerprint=blueprint.fingerprint if blueprint is not None else None,
        calibration_wall_time_seconds=calibration_seconds,
        results=ordered_results,
    )


def write_performance_report(
    report: PerformanceBenchmarkReport,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_questions(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_QUESTIONS
    questions = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not questions:
        raise ValueError("prompt file contains no non-empty lines")
    return questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=SUPPORTED_ATTENTION_BACKENDS,
        default=list(SUPPORTED_ATTENTION_BACKENDS),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-layers", type=int, default=4)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-core-std", type=float, default=0.0)
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--calibration-prompt-count", type=int, default=8)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--scoring-micro-batch-size", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprob-atol", type=float, default=5e-3)
    parser.add_argument("--logprob-rtol", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all-backends", action="store_true")
    parser.add_argument("--strict-equivalence", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = PerformanceBenchmarkConfig(
        model_name=arguments.model,
        model_revision=arguments.revision,
        backends=tuple(arguments.backends),
        dtype=arguments.dtype,
        device=arguments.device,
        adapter_rank=arguments.adapter_rank,
        adapter_layers=arguments.adapter_layers,
        adapter_scale=arguments.adapter_scale,
        adapter_core_std=arguments.adapter_core_std,
        prompt_count=arguments.prompt_count,
        calibration_prompt_count=arguments.calibration_prompt_count,
        group_size=arguments.group_size,
        max_prompt_tokens=arguments.max_prompt_tokens,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        do_sample=not arguments.greedy,
        scoring_micro_batch_size=arguments.scoring_micro_batch_size,
        warmup_iterations=arguments.warmup_iterations,
        benchmark_iterations=arguments.iterations,
        seed=arguments.seed,
        logprob_atol=arguments.logprob_atol,
        logprob_rtol=arguments.logprob_rtol,
    )
    report = run_attention_backend_benchmark(
        config,
        questions=_load_questions(arguments.prompt_file),
    )
    if arguments.output is not None:
        destination = write_performance_report(report, arguments.output)
        print(destination)
    else:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    unavailable = [result for result in report.results if result.status != "available"]
    inequivalent = [
        result
        for result in report.results
        if result.status == "available"
        and (
            result.output_equivalence is None
            or result.logprob_equivalence is None
            or not result.output_equivalence.exact_match
            or not result.logprob_equivalence.allclose
        )
    ]
    if arguments.require_all_backends and unavailable:
        return 2
    if arguments.strict_equivalence and inequivalent:
        return 3
    if report.results[0].status != "available":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = [
    "SUPPORTED_ATTENTION_BACKENDS",
    "AdapterBlueprint",
    "AttentionBackend",
    "BackendBenchmarkResult",
    "GenerationArtifact",
    "LogprobEquivalence",
    "OutputEquivalence",
    "PerformanceBenchmarkConfig",
    "PerformanceBenchmarkReport",
    "PhaseMetrics",
    "apply_adapter_blueprint",
    "benchmark_backend_safely",
    "benchmark_rollout_generation",
    "benchmark_teacher_forced_scoring",
    "build_fixed_scoring_rollout",
    "build_parser",
    "calibrate_adapter_blueprint",
    "compare_generation_outputs",
    "compare_log_probs",
    "main",
    "run_attention_backend_benchmark",
    "write_performance_report",
]
