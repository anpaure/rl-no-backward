#!/usr/bin/env bash
set -euo pipefail

# Reproducible H100 runtime for the vLLM/FlashAttention training fast path.
# vLLM 0.22.0 pins Torch 2.11, so its FlashAttention wheel must use the same
# Torch and CUDA ABI.  The wheel is downloaded from the prebuild repository
# requested for this experiment and verified before installation.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${1:-$project_dir/.venv-vllm}"
uv_bin="${UV_BIN:-}"

if [[ -z "$uv_bin" ]]; then
  uv_bin="$(command -v uv || true)"
  if [[ -z "$uv_bin" ]]; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
  fi
fi

# vLLM's Triton launcher is compiled lazily and needs Python.h. Minimal H100
# images often omit the system python-dev package, while uv's managed Python
# distribution includes its matching headers without requiring sudo.
"$uv_bin" python install 3.12
"$uv_bin" venv --managed-python --python 3.12 "$runtime_dir"
python_include="$($runtime_dir/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "$python_include/Python.h" ]]; then
  echo "uv-managed Python is missing $python_include/Python.h" >&2
  exit 1
fi
"$uv_bin" pip install \
  --python "$runtime_dir/bin/python" \
  --torch-backend cu130 \
  -e "$project_dir" \
  'vllm==0.22.0'

wheel_name='flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl'
wheel_url='https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl'
wheel_sha256='173b0e1a5d6a0becb4ce11b755605c4b20289a6d17816a00cfd7a69078c3e612'
download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT
wheel_path="$download_dir/$wheel_name"

curl --fail --location --retry 3 --output "$wheel_path" "$wheel_url"
printf '%s  %s\n' "$wheel_sha256" "$wheel_path" | sha256sum --check --strict
"$uv_bin" pip install --python "$runtime_dir/bin/python" --no-deps "$wheel_path"

"$runtime_dir/bin/python" - <<'PY'
import flash_attn
import torch
import vllm

expected = ("2.8.3", "0.22.0", "2.11.0+cu130")
actual = (flash_attn.__version__, vllm.__version__, torch.__version__)
if actual != expected:
    raise RuntimeError(f"unexpected optimized runtime versions: {actual!r} != {expected!r}")
print(f"ready: flash-attn={actual[0]} vllm={actual[1]} torch={actual[2]}")
PY
