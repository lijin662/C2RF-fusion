#!/usr/bin/env bash
# Official CUDA 12.8 wheels supporting Blackwell; generic setup_env.sh remains available.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  printf 'This preset targets the Linux RTX 5090 server. Use setup_env.sh locally.\n' >&2
  exit 1
fi
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
bash scripts/setup_env.sh
.venv/bin/python check_gpu.py
