"""Canonical cryptographic provenance for fixed-policy sequence rollouts.

The training JSONL deliberately does not embed full rollout tensors.  These
digests provide a compact, tamper-evident identity for the exact prompt and
response tensors, the behavior-policy log probabilities, and the sampling
seed used to produce them.  Tensor values are normalized to explicit little-
endian wire dtypes before hashing so the result does not depend on their
source device or incidental integer dtype.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

ROLLOUT_PROVENANCE_VERSION = "rl-no-backward-rollout-v1"
ROLLOUT_DIGEST_ALGORITHM = "sha256"


class RolloutTensorSource(Protocol):
    """The rollout fields covered by :func:`build_rollout_provenance`."""

    prompt_input_ids: Tensor
    prompt_attention_mask: Tensor
    response_input_ids: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("rollout seed must be a non-negative integer")
    return seed


def _update_header(digest: Any, name: str, wire_dtype: str, shape: tuple[int, ...]) -> None:
    name_bytes = name.encode("ascii")
    dtype_bytes = wire_dtype.encode("ascii")
    digest.update(struct.pack("<I", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack("<I", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack("<I", len(shape)))
    digest.update(struct.pack(f"<{len(shape)}q", *shape))


def _update_tensor(
    digest: Any,
    name: str,
    tensor: Tensor,
    wire_kind: Literal["int64", "bool", "float32"],
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    value = tensor.detach().to(device="cpu")
    shape = tuple(int(size) for size in value.shape)
    if wire_kind == "int64":
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must contain integer token IDs")
        canonical = value.to(dtype=torch.int64).contiguous().numpy().astype("<i8", copy=False)
    elif wire_kind == "bool":
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have boolean dtype")
        canonical = value.to(dtype=torch.uint8).contiguous().numpy()
    else:
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        canonical_tensor = value.to(dtype=torch.float32).contiguous()
        if not torch.isfinite(canonical_tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        canonical = canonical_tensor.numpy().astype("<f4", copy=False)
    _update_header(digest, name, wire_kind, shape)
    digest.update(canonical.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class RolloutProvenance:
    """SHA-256 identities persisted beside one optimizer-step rollout."""

    rollout_seed: int
    token_digest: str
    behavior_logprob_digest: str
    digest: str

    def __post_init__(self) -> None:
        _validate_seed(self.rollout_seed)
        for name, value in (
            ("token_digest", self.token_digest),
            ("behavior_logprob_digest", self.behavior_logprob_digest),
            ("digest", self.digest),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")

    def as_record_fields(self) -> dict[str, str | int]:
        """Return stable JSONL keys for a training-step record."""

        return {
            "rollout_provenance_version": ROLLOUT_PROVENANCE_VERSION,
            "rollout_digest_algorithm": ROLLOUT_DIGEST_ALGORITHM,
            "rollout_seed": self.rollout_seed,
            "rollout_token_digest": self.token_digest,
            "behavior_logprob_digest": self.behavior_logprob_digest,
            "rollout_digest": self.digest,
        }


def build_rollout_provenance(
    rollout: RolloutTensorSource,
    *,
    seed: int,
) -> RolloutProvenance:
    """Hash one fixed rollout using a canonical, versioned wire format."""

    rollout_seed = _validate_seed(seed)
    seed_bytes = struct.pack("<Q", rollout_seed)

    token_hash = hashlib.sha256()
    token_hash.update(f"{ROLLOUT_PROVENANCE_VERSION}/tokens\0".encode("ascii"))
    token_hash.update(seed_bytes)
    _update_tensor(token_hash, "prompt_input_ids", rollout.prompt_input_ids, "int64")
    _update_tensor(
        token_hash,
        "prompt_attention_mask",
        rollout.prompt_attention_mask,
        "bool",
    )
    _update_tensor(token_hash, "response_input_ids", rollout.response_input_ids, "int64")
    _update_tensor(token_hash, "response_mask", rollout.response_mask, "bool")
    token_digest = token_hash.hexdigest()

    logprob_hash = hashlib.sha256()
    logprob_hash.update(f"{ROLLOUT_PROVENANCE_VERSION}/behavior-logprobs\0".encode("ascii"))
    logprob_hash.update(seed_bytes)
    _update_tensor(
        logprob_hash,
        "old_token_log_probs",
        rollout.old_token_log_probs,
        "float32",
    )
    behavior_logprob_digest = logprob_hash.hexdigest()

    combined_hash = hashlib.sha256()
    combined_hash.update(f"{ROLLOUT_PROVENANCE_VERSION}/combined\0".encode("ascii"))
    combined_hash.update(seed_bytes)
    combined_hash.update(bytes.fromhex(token_digest))
    combined_hash.update(bytes.fromhex(behavior_logprob_digest))
    combined_digest = combined_hash.hexdigest()
    return RolloutProvenance(
        rollout_seed=rollout_seed,
        token_digest=token_digest,
        behavior_logprob_digest=behavior_logprob_digest,
        digest=combined_digest,
    )


__all__ = [
    "ROLLOUT_DIGEST_ALGORITHM",
    "ROLLOUT_PROVENANCE_VERSION",
    "RolloutProvenance",
    "RolloutTensorSource",
    "build_rollout_provenance",
]
