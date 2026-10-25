#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Set TORCH_INDEX_URL to the PyTorch wheel index matching the server driver.
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  .venv/bin/python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
printf 'Environment ready. Use: source .venv/bin/activate\n'
