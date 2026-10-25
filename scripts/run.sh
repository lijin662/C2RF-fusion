#!/usr/bin/env bash
# Usage: bash scripts/run.sh {check|smoke|train|fuse|evaluate|reg-test} {RoadScene|OSDataset} [python options...]
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:?Specify check, smoke, train, fuse, or reg-test}"
dataset="${2:?Specify RoadScene or OSDataset}"
shift 2
PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT="${EXPERIMENT:-baseline}"
common=(--dataset "$dataset" --train_dataroot "$DATA_ROOT/train" --test_dataroot "$DATA_ROOT/test"
        --output_dir "$OUTPUT_ROOT" --experiment "$EXPERIMENT" --device "$DEVICE"
        --batch_size "${BATCH_SIZE:-2}" --nThreads "${WORKERS:-4}")
case "$action" in
  check)
    "$PYTHON" check_dataset.py "${common[@]}" --phase train "$@"
    "$PYTHON" check_dataset.py "${common[@]}" --phase test --generate_test_warp "$@"
    ;;
  smoke)
    "$PYTHON" smoke_test.py "${common[@]}" --phase train "$@"
    "$PYTHON" smoke_test.py "${common[@]}" --phase test "$@"
    ;;
  train)
    "$PYTHON" -u train_Fu.py "${common[@]}" "$@"
    ;;
  fuse)
    "$PYTHON" fuse.py "${common[@]}" \
      --fusion_checkpoint "$OUTPUT_ROOT/$dataset/$EXPERIMENT/latest.pt" "$@"
    ;;
  reg-test)
    "$PYTHON" test.py "${common[@]}" "$@"
    ;;
  evaluate)
    "$PYTHON" evaluate_fusion.py "${common[@]}" "$@"
    ;;
  *) printf 'Unknown action: %s\n' "$action" >&2; exit 2 ;;
esac
