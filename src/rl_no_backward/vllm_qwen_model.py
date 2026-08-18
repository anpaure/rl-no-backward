"""vLLM 0.22 Qwen2 implementation of the calibrated residual-core policy.

This module intentionally imports vLLM directly and is loaded only through the
lazy model-registry entry installed by ``register_vllm_residual_qwen_model``.
Keeping it separate lets the rest of the project run on systems where vLLM is
not available (including macOS development machines).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen2Config
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
)

from .vllm_rollout import (
    VLLMResidualCoreAdapter,
    apply_adapter_to_vllm_split_state,
)


def _adapter_config(config: Qwen2Config) -> tuple[frozenset[int], int, float]:
    raw_layers = getattr(config, "residual_core_adapter_layers", None)
    raw_rank = getattr(config, "residual_core_adapter_rank", None)
    raw_scale = getattr(config, "residual_core_adapter_scale", None)
    if not isinstance(raw_layers, (list, tuple)) or not raw_layers:
        raise ValueError("residual_core_adapter_layers must be a non-empty list")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_layers):
        raise ValueError("residual_core_adapter_layers must contain integers")
    layers = frozenset(int(index) for index in raw_layers)
    if len(layers) != len(raw_layers) or min(layers) < 0:
        raise ValueError("residual_core_adapter_layers must be unique and non-negative")
    if isinstance(raw_rank, bool) or not isinstance(raw_rank, int) or raw_rank < 1:
        raise ValueError("residual_core_adapter_rank must be a positive integer")
    scale = float(raw_scale)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("residual_core_adapter_scale must be finite")
    return layers, int(raw_rank), scale


class ResidualCoreQwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen block that adds the exact post-block residual-core delta."""

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        adapter_layers, rank, scale = _adapter_config(config)
        layer_index = extract_layer_index(prefix)
        self.residual_core_adapter: VLLMResidualCoreAdapter | None
        if layer_index in adapter_layers:
            self.residual_core_adapter = VLLMResidualCoreAdapter(
                config.hidden_size,
                rank,
                scale=scale,
                layer_index=layer_index,
            )
        else:
            self.residual_core_adapter = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch, output_residual = super().forward(positions, hidden_states, residual)
        adapter = self.residual_core_adapter
        if adapter is not None:
            branch, output_residual = apply_adapter_to_vllm_split_state(
                branch,
                output_residual,
                adapter,
            )
        return branch, output_residual


class ResidualCoreQwen2Model(Qwen2Model):
    """Use the residual-aware decoder layer without changing weight names."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=ResidualCoreQwen2DecoderLayer,
        )


class ResidualCoreQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2 causal LM with four mutable post-block residual-core adapters.

    The constructor mirrors vLLM 0.22's ``Qwen2ForCausalLM`` constructor but
    substitutes :class:`ResidualCoreQwen2Model`.  Inheriting the original class
    preserves its Hugging Face weight loader and LoRA/PP interface declarations.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        # Validate once up front as well as in each decoder constructor so a
        # malformed override fails before any weights are loaded.
        adapter_layers, _, _ = _adapter_config(config)
        if max(adapter_layers) >= config.num_hidden_layers:
            raise ValueError("a residual adapter layer exceeds the Qwen layer count")

        self.config = config
        self.quant_config = quant_config
        self.model = ResidualCoreQwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


__all__ = [
    "ResidualCoreQwen2DecoderLayer",
    "ResidualCoreQwen2ForCausalLM",
    "ResidualCoreQwen2Model",
]
