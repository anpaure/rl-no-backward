"""Public optimizer facade for fixed autoregressive sequence rollouts.

The implementations are deliberately split: :mod:`sequence_backprop` contains
the conventional reverse-mode baseline, while :mod:`sequence_forward_only` is
independently auditable as strict inference-only code.
"""

from .sequence_backprop import (
    BackpropSequenceConfig,
    BackpropSequenceStepResult,
    make_sequence_grpo_optimizer,
    sequence_grpo_step,
)
from .sequence_fastpath import FusedProbeConfig, enable_batched_probe_adapters
from .sequence_forward_only import (
    ForwardSequenceConfig,
    ForwardSequenceMethod,
    ForwardSequenceStepResult,
    SequenceActiveSubspace,
    directional_sequence_score_statistics,
    forward_sequence_step,
    sampled_sequence_kl,
)

SequenceStepResult = BackpropSequenceStepResult | ForwardSequenceStepResult

__all__ = [
    "BackpropSequenceConfig",
    "BackpropSequenceStepResult",
    "ForwardSequenceConfig",
    "ForwardSequenceMethod",
    "ForwardSequenceStepResult",
    "FusedProbeConfig",
    "SequenceActiveSubspace",
    "SequenceStepResult",
    "directional_sequence_score_statistics",
    "enable_batched_probe_adapters",
    "forward_sequence_step",
    "make_sequence_grpo_optimizer",
    "sampled_sequence_kl",
    "sequence_grpo_step",
]
