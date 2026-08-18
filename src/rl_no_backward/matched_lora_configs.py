"""Dependency-neutral optimizer configurations for isolated matched methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchedBackpropConfig:
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    prompt_groups_per_micro_batch: int = 1
    fused_adamw: bool = True
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if (
            isinstance(self.prompt_groups_per_micro_batch, bool)
            or not isinstance(self.prompt_groups_per_micro_batch, int)
            or self.prompt_groups_per_micro_batch < 1
        ):
            raise ValueError("prompt_groups_per_micro_batch must be positive")
        if not isinstance(self.fused_adamw, bool):
            raise TypeError("fused_adamw must be boolean")
        if self.scheduler_type not in {"linear", "cosine", "constant"}:
            raise ValueError("scheduler_type must be linear, cosine, or constant")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")


__all__ = ["MatchedBackpropConfig"]
