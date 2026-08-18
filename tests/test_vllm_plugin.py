from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

import rl_no_backward.vllm_plugin as plugin


def test_pyproject_installs_process_wide_vllm_general_plugin() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["entry-points"]["vllm.general_plugins"] == {
        plugin.VLLM_PLUGIN_NAME: plugin.VLLM_PLUGIN_ENTRY_POINT
    }


def test_general_plugin_registration_is_lazy_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class Registry:
        architectures: ClassVar[set[str]] = set()

        @classmethod
        def get_supported_archs(cls) -> set[str]:
            return cls.architectures

        @classmethod
        def register_model(cls, architecture: str, model_qualname: str) -> None:
            calls.append((architecture, model_qualname))
            cls.architectures.add(architecture)

    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.22.0"
    fake_vllm.ModelRegistry = Registry
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    plugin.register_residual_qwen_model()
    plugin.register_residual_qwen_model()

    assert calls == [(plugin.VLLM_RESIDUAL_ARCHITECTURE, plugin.VLLM_MODEL_QUALNAME)]
    # The plugin registers a lazy module path; it never imports the CUDA model.
    assert "rl_no_backward.vllm_qwen_model" not in sys.modules


def test_plugin_preflight_rejects_stale_editable_metadata_and_env_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin, "entry_points", lambda **_kwargs: [])
    with pytest.raises(RuntimeError, match="entry point is not installed"):
        plugin.ensure_vllm_plugin_discoverable()

    installed = SimpleNamespace(
        name=plugin.VLLM_PLUGIN_NAME,
        value=plugin.VLLM_PLUGIN_ENTRY_POINT,
    )
    monkeypatch.setattr(plugin, "entry_points", lambda **_kwargs: [installed])
    monkeypatch.setenv("VLLM_PLUGINS", "some_other_plugin")
    with pytest.raises(RuntimeError, match="filters out required plugin"):
        plugin.ensure_vllm_plugin_discoverable()

    monkeypatch.setenv(
        "VLLM_PLUGINS",
        f"some_other_plugin,{plugin.VLLM_PLUGIN_NAME}",
    )
    plugin.ensure_vllm_plugin_discoverable()


def test_trusted_callable_serialization_is_explicit_and_contradiction_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = plugin.VLLM_INSECURE_SERIALIZATION_ENV
    monkeypatch.delenv(key, raising=False)

    plugin.configure_trusted_vllm_callable_serialization(enabled=False)
    assert key not in os.environ

    plugin.configure_trusted_vllm_callable_serialization(enabled=True)
    assert os.environ[key] == "1"

    monkeypatch.setenv(key, "0")
    with pytest.raises(RuntimeError, match="explicitly disables"):
        plugin.configure_trusted_vllm_callable_serialization(enabled=True)

    monkeypatch.setenv(key, "1")
    with pytest.raises(RuntimeError, match="conflicts with the default-off"):
        plugin.configure_trusted_vllm_callable_serialization(enabled=False)

    monkeypatch.setenv(key, "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        plugin.configure_trusted_vllm_callable_serialization(enabled=True)


def test_batch_invariance_is_explicit_and_contradiction_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = plugin.VLLM_BATCH_INVARIANT_ENV
    monkeypatch.delenv(key, raising=False)

    plugin.configure_vllm_batch_invariance(enabled=False)
    assert key not in os.environ

    plugin.configure_vllm_batch_invariance(enabled=True)
    assert os.environ[key] == "1"

    monkeypatch.setenv(key, "0")
    with pytest.raises(RuntimeError, match="explicitly disables"):
        plugin.configure_vllm_batch_invariance(enabled=True)

    monkeypatch.setenv(key, "1")
    with pytest.raises(RuntimeError, match="conflicts with the default-off"):
        plugin.configure_vllm_batch_invariance(enabled=False)

    monkeypatch.setenv(key, "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        plugin.configure_vllm_batch_invariance(enabled=True)


def test_v1_multiprocessing_is_explicit_and_contradiction_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = plugin.VLLM_V1_MULTIPROCESSING_ENV
    monkeypatch.delenv(key, raising=False)

    plugin.configure_vllm_v1_multiprocessing(enabled=False)
    assert os.environ[key] == "0"

    monkeypatch.setenv(key, "1")
    with pytest.raises(RuntimeError, match="conflicts with"):
        plugin.configure_vllm_v1_multiprocessing(enabled=False)

    monkeypatch.setenv(key, "0")
    with pytest.raises(RuntimeError, match="conflicts with"):
        plugin.configure_vllm_v1_multiprocessing(enabled=True)

    monkeypatch.setenv(key, "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        plugin.configure_vllm_v1_multiprocessing(enabled=True)

    monkeypatch.delenv(key)
    plugin.configure_vllm_v1_multiprocessing(enabled=True)
    assert os.environ[key] == "1"
