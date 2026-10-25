#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-dataset}"
RESOLUTION="${RESOLUTION:-256}"
DATASET_NAME="${DATASET_NAME:-OSDataset}"
LIMIT="${LIMIT:-0}"
ARCHIVE="${ARCHIVE:-downloads/OSdataset.zip}"
for split in train val test; do
  "$PYTHON" prepare_osdataset.py --archive "$ARCHIVE" --output_root "$DATA_ROOT" \
    --dataset_name "$DATASET_NAME" --resolution "$RESOLUTION" --split "$split" --limit "$LIMIT"
done
