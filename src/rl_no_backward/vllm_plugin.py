"""vLLM general-plugin registration for the residual-core Qwen policy.

vLLM may force its V1 EngineCore to use Python's ``spawn`` start method after
the Hugging Face trainer has initialized CUDA.  An in-memory ModelRegistry
mutation in the parent process is therefore insufficient: the fresh child must
discover the architecture through installed package metadata.  vLLM loads the
``vllm.general_plugins`` entry-point group in the parent, EngineCore, and worker
processes, making this the supported process-safe registration hook.
"""

from __future__ import annotations

import os
from importlib.metadata import entry_points
from typing import Any

VLLM_PLUGIN_NAME = "rl_no_backward_residual_qwen"
VLLM_RESIDUAL_ARCHITECTURE = "ResidualCoreQwen2ForCausalLM"
VLLM_MODEL_QUALNAME = "rl_no_backward.vllm_qwen_model:ResidualCoreQwen2ForCausalLM"
VLLM_PLUGIN_ENTRY_POINT = "rl_no_backward.vllm_plugin:register_residual_qwen_model"
VLLM_INSECURE_SERIALIZATION_ENV = "VLLM_ALLOW_INSECURE_SERIALIZATION"
VLLM_BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
VLLM_V1_MULTIPROCESSING_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


def _validated_vllm_registry() -> Any:
    try:
        import vllm
        from vllm import ModelRegistry
    except ImportError as error:  # pragma: no cover - exercised in the CUDA runtime
        raise RuntimeError("vLLM is not installed; install the pinned 0.22.x CUDA wheel") from error

    version_text = str(getattr(vllm, "__version__", ""))
    numeric = version_text.split("+", 1)[0].split(".")
    try:
        version_pair = (int(numeric[0]), int(numeric[1]))
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"could not parse vLLM version {version_text!r}") from error
    if version_pair != (0, 22):
        raise RuntimeError(
            "the residual Qwen integration was validated against vLLM 0.22.x; "
            f"found {version_text!r}"
        )
    return ModelRegistry


def register_residual_qwen_model() -> None:
    """Idempotently register the lazy custom model in the current process."""

    registry = _validated_vllm_registry()
    if VLLM_RESIDUAL_ARCHITECTURE in registry.get_supported_archs():
        return
    # A lazy string is required: importing the CUDA model implementation while
    # plugin discovery runs would defeat vLLM's spawn/fork safety guarantees.
    registry.register_model(VLLM_RESIDUAL_ARCHITECTURE, VLLM_MODEL_QUALNAME)


def ensure_vllm_plugin_discoverable() -> None:
    """Fail before engine spawn if editable-install metadata is stale or filtered."""

    matches = [
        candidate
        for candidate in entry_points(group="vllm.general_plugins")
        if candidate.name == VLLM_PLUGIN_NAME
    ]
    if len(matches) != 1 or matches[0].value != VLLM_PLUGIN_ENTRY_POINT:
        raise RuntimeError(
            "the rl-no-backward vLLM plugin entry point is not installed; reinstall the "
            "project into the vLLM environment (for example: "
            "`uv pip install --python .venv-vllm/bin/python -e .`)"
        )

    allowed = os.environ.get("VLLM_PLUGINS")
    if allowed is not None:
        enabled = {name.strip() for name in allowed.split(",") if name.strip()}
        if VLLM_PLUGIN_NAME not in enabled:
            raise RuntimeError(
                f"VLLM_PLUGINS filters out required plugin {VLLM_PLUGIN_NAME!r}; "
                "add it to the comma-separated allowlist or unset VLLM_PLUGINS"
            )


def configure_trusted_vllm_callable_serialization(*, enabled: bool) -> None:
    """Configure vLLM callable RPC serialization before importing vLLM.

    ``LLM.apply_model`` sends a Python callable to spawned EngineCore/worker
    processes.  vLLM 0.22 intentionally requires pickle/cloudpickle for that
    payload and disables it by default because deserializing untrusted data can
    execute arbitrary code.  This opt-in is suitable only for this benchmark's
    trusted, local process tree; it must never be used for an RPC endpoint that
    accepts payloads from another user or host.
    """

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    configured = os.environ.get(VLLM_INSECURE_SERIALIZATION_ENV)
    if configured is not None and configured not in {"0", "1"}:
        raise RuntimeError(
            f"{VLLM_INSECURE_SERIALIZATION_ENV} must be exactly '0' or '1', not {configured!r}"
        )
    if enabled:
        if configured == "0":
            raise RuntimeError(
                f"config enables trusted callable serialization but "
                f"{VLLM_INSECURE_SERIALIZATION_ENV}=0 explicitly disables it"
            )
        if configured is None:
            # A spawned EngineCore inherits this environment.  This assignment
            # deliberately happens before the first vLLM import in the factory.
            os.environ[VLLM_INSECURE_SERIALIZATION_ENV] = "1"
        return
    if configured == "1":
        raise RuntimeError(
            f"{VLLM_INSECURE_SERIALIZATION_ENV}=1 conflicts with the default-off "
            "vllm_allow_insecure_serialization config"
        )


def configure_vllm_batch_invariance(*, enabled: bool) -> None:
    """Set scheduling-invariant vLLM execution before the first vLLM import.

    vLLM 0.22 does not promise reproducible offline output under its default
    multiprocessing scheduler.  Its H100 batch-invariance mode makes seeded
    output insensitive to batch order and scheduling, at a possible throughput
    cost from deterministic kernels and disabled nondeterministic optimizations.
    Spawned EngineCore and worker processes inherit this explicit environment.
    """

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    configured = os.environ.get(VLLM_BATCH_INVARIANT_ENV)
    if configured is not None and configured not in {"0", "1"}:
        raise RuntimeError(
            f"{VLLM_BATCH_INVARIANT_ENV} must be exactly '0' or '1', not {configured!r}"
        )
    if enabled:
        if configured == "0":
            raise RuntimeError(
                f"config enables vLLM batch invariance but "
                f"{VLLM_BATCH_INVARIANT_ENV}=0 explicitly disables it"
            )
        if configured is None:
            os.environ[VLLM_BATCH_INVARIANT_ENV] = "1"
        return
    if configured == "1":
        raise RuntimeError(
            f"{VLLM_BATCH_INVARIANT_ENV}=1 conflicts with the default-off "
            "vllm_batch_invariant config"
        )


def configure_vllm_v1_multiprocessing(*, enabled: bool) -> None:
    """Select vLLM's EngineCore process mode before importing vLLM.

    vLLM 0.22 reads this environment control while constructing its V1
    ``EngineCoreClient``.  Disabling it selects the in-process client, so
    ``LLM.apply_model`` invokes the mutable adapter update directly and does
    not require insecure callable serialization.  An explicit value is used
    for both modes so a contradictory shell environment cannot silently win.
    """

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    configured = os.environ.get(VLLM_V1_MULTIPROCESSING_ENV)
    if configured is not None and configured not in {"0", "1"}:
        raise RuntimeError(
            f"{VLLM_V1_MULTIPROCESSING_ENV} must be exactly '0' or '1', "
            f"not {configured!r}"
        )
    requested = "1" if enabled else "0"
    if configured is not None and configured != requested:
        raise RuntimeError(
            f"{VLLM_V1_MULTIPROCESSING_ENV}={configured} conflicts with "
            f"vllm_enable_v1_multiprocessing={str(enabled).lower()}"
        )
    if configured is None:
        # This must happen before plugin discovery or any other vLLM import.
        os.environ[VLLM_V1_MULTIPROCESSING_ENV] = requested


__all__ = [
    "VLLM_BATCH_INVARIANT_ENV",
    "VLLM_INSECURE_SERIALIZATION_ENV",
    "VLLM_MODEL_QUALNAME",
    "VLLM_PLUGIN_ENTRY_POINT",
    "VLLM_PLUGIN_NAME",
    "VLLM_RESIDUAL_ARCHITECTURE",
    "VLLM_V1_MULTIPROCESSING_ENV",
    "configure_trusted_vllm_callable_serialization",
    "configure_vllm_batch_invariance",
    "configure_vllm_v1_multiprocessing",
    "ensure_vllm_plugin_discoverable",
    "register_residual_qwen_model",
]
