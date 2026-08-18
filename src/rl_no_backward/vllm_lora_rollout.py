"""vLLM 0.22 rollout backend for repeatedly reloaded standard PEFT LoRA."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .standard_lora import (
    export_vllm_lora_adapter,
    make_vllm_lora_request,
)
from .vllm_plugin import (
    VLLM_BATCH_INVARIANT_ENV,
    configure_trusted_vllm_callable_serialization,
    configure_vllm_batch_invariance,
    configure_vllm_v1_multiprocessing,
)
from .vllm_rollout import (
    VLLMGreedyGeneration,
    VLLMGroupedGeneration,
    parse_vllm_greedy_outputs,
    parse_vllm_grouped_outputs,
)


@dataclass(frozen=True, slots=True)
class LoRAReloadReceipt:
    version: int
    policy_version: str
    state_digest: str
    adapter_model_sha256: str
    parameter_count: int
    adapter_path: str
    active_lora_ids: tuple[int, ...]
    prefix_cache_reset: bool
    load_inplace: bool = True
    adapter_path_transient: bool = True
    durable_hash_fields: tuple[str, ...] = ("state_digest", "adapter_model_sha256")


@dataclass(frozen=True, slots=True)
class LoRANextTokenProbe:
    """Small content-addressed signature used to prove same-ID adapter reloads."""

    policy_version: str
    state_digest: str
    token_id: int
    selected_token_logprob: float


def _resolved_flash_attn_version(llm: Any) -> int | None:
    candidates = (
        getattr(llm, "vllm_config", None),
        getattr(getattr(llm, "llm_engine", None), "vllm_config", None),
        getattr(getattr(getattr(llm, "llm_engine", None), "model_executor", None), "vllm_config", None),
    )
    for candidate in candidates:
        attention = getattr(candidate, "attention_config", None)
        value = getattr(attention, "flash_attn_version", None)
        if isinstance(value, int):
            return value
        if isinstance(attention, dict) and isinstance(attention.get("flash_attn_version"), int):
            return int(attention["flash_attn_version"])
    return None


def create_standard_lora_vllm_engine(
    model: str,
    *,
    revision: str,
    dtype: str,
    max_model_len: int,
    max_lora_rank: int,
    kv_cache_memory_bytes: int,
    enforce_eager: bool,
    flash_attn_version: int,
    max_num_seqs: int,
    seed: int,
) -> Any:
    """Create a protected single-GPU engine with LoRA and explicit FA2."""

    if flash_attn_version != 2:
        raise ValueError("the matched run pins vLLM FlashAttention version 2")
    if max_lora_rank < 1 or max_model_len < 2 or kv_cache_memory_bytes < 1:
        raise ValueError("invalid vLLM LoRA engine dimensions")
    configure_vllm_v1_multiprocessing(enabled=False)
    configure_vllm_batch_invariance(enabled=False)
    configure_trusted_vllm_callable_serialization(enabled=False)
    os.environ.setdefault(VLLM_BATCH_INVARIANT_ENV, "0")
    try:
        from vllm import LLM
    except ImportError as error:  # pragma: no cover - CUDA runtime only
        raise RuntimeError("the matched LoRA rollout backend requires vllm==0.22.0") from error
    llm = LLM(
        model=model,
        revision=revision,
        dtype=dtype,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        skip_tokenizer_init=True,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=max_model_len,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        enforce_eager=enforce_eager,
        logprobs_mode="processed_logprobs",
        seed=seed,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=1,
        max_cpu_loras=1,
        fully_sharded_loras=False,
        max_num_seqs=max_num_seqs,
        disable_log_stats=True,
    )
    resolved = _resolved_flash_attn_version(llm)
    if resolved != 2:
        raise RuntimeError(f"vLLM did not resolve the required FlashAttention version 2: {resolved}")
    return llm


class ReloadableLoRAGenerator:
    """Guard generation behind an explicit in-place adapter reload receipt."""

    def __init__(
        self,
        llm: Any,
        *,
        lora_name: str = "matched-policy",
        lora_int_id: int = 1,
        retain_exports: int = 2,
    ):
        if retain_exports < 1:
            raise ValueError("retain_exports must be positive")
        self.llm = llm
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.request: Any | None = None
        self.policy_version: str | None = None
        self.state_digest: str | None = None
        self.retain_exports = retain_exports
        self._exports: list[Path] = []

    def sync(
        self,
        model: nn.Module,
        export_root: str | Path,
        *,
        version: int,
        policy_version: str,
    ) -> LoRAReloadReceipt:
        adapter_path, export_receipt = export_vllm_lora_adapter(
            model,
            export_root,
            version=version,
        )
        reload_request = make_vllm_lora_request(
            adapter_path,
            lora_name=self.lora_name,
            lora_int_id=self.lora_int_id,
        )
        engine = getattr(self.llm, "llm_engine", None)
        add_lora = getattr(engine, "add_lora", None)
        if not callable(add_lora):
            raise TypeError("vLLM engine does not expose add_lora")
        loaded = add_lora(reload_request)
        if loaded is False:
            raise RuntimeError("vLLM refused the in-place LoRA reload")
        list_loras = getattr(engine, "list_loras", None)
        active = tuple(sorted(int(value) for value in list_loras())) if callable(list_loras) else ()
        if self.lora_int_id not in active:
            raise RuntimeError(f"vLLM did not report active LoRA ID {self.lora_int_id}: {active}")
        reset = self.llm.reset_prefix_cache()
        if reset is False:
            raise RuntimeError("vLLM refused to reset prefix cache after LoRA reload")

        # Generation should reuse the adapter just loaded above rather than
        # perform a second disk reload after the cache-reset boundary.
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as error:  # pragma: no cover - CUDA runtime only
            raise RuntimeError("vLLM LoRA request support is unavailable") from error
        self.request = LoRARequest(
            lora_name=self.lora_name,
            lora_int_id=self.lora_int_id,
            lora_path=str(adapter_path.resolve()),
            load_inplace=False,
        )
        self.policy_version = policy_version
        self.state_digest = str(export_receipt["state_digest"])
        self._exports.append(adapter_path)
        while len(self._exports) > self.retain_exports:
            expired = self._exports.pop(0)
            if expired != adapter_path and expired.is_dir():
                shutil.rmtree(expired)
        return LoRAReloadReceipt(
            version=version,
            policy_version=policy_version,
            state_digest=self.state_digest,
            adapter_model_sha256=str(export_receipt["adapter_model_sha256"]),
            parameter_count=int(export_receipt["parameter_count"]),
            adapter_path=str(adapter_path.resolve()),
            active_lora_ids=active,
            prefix_cache_reset=True,
        )

    def _require_synced(self) -> tuple[Any, str]:
        if self.request is None or self.policy_version is None or self.state_digest is None:
            raise RuntimeError("reload a LoRA policy before generation")
        return self.request, self.policy_version

    def probe_next_token(self, prompt_token_ids: tuple[int, ...]) -> LoRANextTokenProbe:
        """Return one deterministic chosen-token/log-probability reload signature."""

        request, policy_version = self._require_synced()
        if not prompt_token_ids:
            raise ValueError("probe prompt token sequence must be non-empty")
        from vllm import SamplingParams

        params = SamplingParams(
            n=1,
            max_tokens=1,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            ignore_eos=True,
            logprobs=1,
            detokenize=False,
            skip_special_tokens=False,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": list(prompt_token_ids)}],
            sampling_params=params,
            use_tqdm=False,
            lora_request=request,
        )
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("vLLM reload probe returned the wrong prompt count")
        request_output = outputs[0]
        if tuple(int(value) for value in request_output.prompt_token_ids) != prompt_token_ids:
            raise RuntimeError("vLLM changed the reload-probe prompt")
        if len(request_output.outputs) != 1:
            raise RuntimeError("vLLM reload probe returned the wrong candidate count")
        candidate = request_output.outputs[0]
        tokens = tuple(int(value) for value in candidate.token_ids)
        if len(tokens) != 1 or not candidate.logprobs or len(candidate.logprobs) != 1:
            raise RuntimeError("vLLM reload probe did not return one token and its log-probability")
        token_id = tokens[0]
        entry = candidate.logprobs[0].get(token_id)
        if entry is None:
            raise RuntimeError("vLLM reload probe omitted the selected token log-probability")
        raw_logprob = getattr(entry, "logprob", entry)
        selected_logprob = float(raw_logprob)
        if not math.isfinite(selected_logprob):
            raise RuntimeError("vLLM reload probe returned a non-finite log-probability")
        assert self.state_digest is not None
        return LoRANextTokenProbe(
            policy_version=policy_version,
            state_digest=self.state_digest,
            token_id=token_id,
            selected_token_logprob=selected_logprob,
        )

    def generate_grouped(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        pad_token_id: int,
        eos_token_ids: tuple[int, ...],
        device: torch.device | str,
    ) -> VLLMGroupedGeneration:
        request, policy_version = self._require_synced()
        if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt token sequences must be non-empty")
        if group_size < 2 or max_new_tokens < 1:
            raise ValueError("invalid grouped generation dimensions")
        if not isinstance(temperature, Real) or temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be positive and finite")
        from vllm import SamplingParams

        prompts = [{"prompt_token_ids": list(prompt)} for prompt in prompt_token_ids]
        params = [
            SamplingParams(
                n=group_size,
                max_tokens=max_new_tokens,
                temperature=float(temperature),
                top_k=0,
                top_p=1.0,
                min_p=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=seed + prompt_index,
                stop_token_ids=list(eos_token_ids),
                ignore_eos=True,
                logprobs=0,
                detokenize=False,
                skip_special_tokens=False,
            )
            for prompt_index in range(len(prompts))
        ]
        outputs = self.llm.generate(
            prompts,
            sampling_params=params,
            use_tqdm=False,
            lora_request=request,
        )
        return parse_vllm_grouped_outputs(
            outputs,
            prompt_token_ids,
            group_size=group_size,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
            policy_version=policy_version,
            device=device,
        )

    def generate_greedy(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
    ) -> VLLMGreedyGeneration:
        request, policy_version = self._require_synced()
        if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt token sequences must be non-empty")
        from vllm import SamplingParams

        prompts = [{"prompt_token_ids": list(prompt)} for prompt in prompt_token_ids]
        params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            stop_token_ids=list(eos_token_ids),
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
        )
        outputs = self.llm.generate(
            prompts,
            sampling_params=params,
            use_tqdm=False,
            lora_request=request,
        )
        return parse_vllm_greedy_outputs(
            outputs,
            prompt_token_ids,
            eos_token_ids=eos_token_ids,
            policy_version=policy_version,
        )


__all__ = [
    "LoRANextTokenProbe",
    "LoRAReloadReceipt",
    "ReloadableLoRAGenerator",
    "create_standard_lora_vllm_engine",
]
