#!/usr/bin/env bash
set -euo pipefail

# Isolated uv runtime for the official TRL + PEFT LoRA-GRPO baseline.  The
# lower-level script supplies vLLM 0.22, Torch 2.11/cu130 and the verified
# prebuilt FlashAttention wheel requested for this project.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${1:-$project_dir/.venv-standard-grpo}"
uv_bin="${UV_BIN:-}"

if [[ -z "$uv_bin" ]]; then
  uv_bin="$(command -v uv || true)"
  if [[ -z "$uv_bin" ]]; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
  fi
fi

if [[ ! -x "$runtime_dir/bin/python" ]] || ! "$runtime_dir/bin/python" -c 'import vllm, flash_attn' >/dev/null 2>&1; then
  "$project_dir/scripts/install_h100_fastpath.sh" "$runtime_dir"
fi

"$uv_bin" pip install \
  --python "$runtime_dir/bin/python" \
  --torch-backend cu130 \
  -e "$project_dir" \
  -r "$project_dir/requirements/standard-grpo-h100.txt"

"$uv_bin" pip check --python "$runtime_dir/bin/python"
"$runtime_dir/bin/python" - <<'PY'
import importlib.metadata as metadata

expected = {
    "torch": "2.11.0",
    "transformers": "5.15.0",
    "vllm": "0.22.0",
    "flash-attn": "2.8.3+cu130torch2.11",
    "trl": "1.10.0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "datasets": "4.8.5",
    "wandb": "0.28.2",
}
actual = {package: metadata.version(package) for package in expected}
for package, wanted in expected.items():
    value = actual[package]
    if package == "torch":
        if not value.startswith(wanted + "+cu130"):
            raise RuntimeError(f"unexpected {package}: {value!r}")
    elif value != wanted:
        raise RuntimeError(f"unexpected {package}: {value!r} != {wanted!r}")
print("standard GRPO runtime ready:", actual)
PY
