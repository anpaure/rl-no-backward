"""Small, reproducible building blocks for forward-only RLVR experiments."""

from .config import RolloutConfig, TaskConfig
from .task import (
    CANDIDATE_ACTIONS,
    CODEBOOK_ACTIONS,
    CODEBOOK_WORDS,
    ChecksumExample,
    ChecksumSplits,
    checksum_target,
    codebook_context,
    codebook_target,
    format_codebook_prompt,
    format_prompt,
    generate_examples,
    group_leave_one_out_advantages,
    make_checksum_splits,
    parse_candidate_action,
    score_candidate,
    score_candidates,
)

__all__ = [
    "CANDIDATE_ACTIONS",
    "CODEBOOK_ACTIONS",
    "CODEBOOK_WORDS",
    "ChecksumExample",
    "ChecksumSplits",
    "RolloutConfig",
    "TaskConfig",
    "checksum_target",
    "codebook_context",
    "codebook_target",
    "format_codebook_prompt",
    "format_prompt",
    "generate_examples",
    "group_leave_one_out_advantages",
    "make_checksum_splits",
    "parse_candidate_action",
    "score_candidate",
    "score_candidates",
]
