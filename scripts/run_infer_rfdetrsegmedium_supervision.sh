#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

CHECKPOINT="$ROOT_DIR/runs/lados_rfdetrsegmedium/checkpoint_best_regular.pth"
SPLIT_FILE="$ROOT_DIR/datasets/new_baseline/train.txt"
COCO_ANNOTATION="$ROOT_DIR/datasets/lados_432/test/_annotations.coco.json"
OUTPUT_DIR="$ROOT_DIR/runs/lados_rfdetrsegmedium/inference_supervision/train"

"$PYTHON_BIN" "$ROOT_DIR/scripts/infer_rfdetrsegmedium_supervision.py" \
  --checkpoint "$CHECKPOINT" \
  --split-file "$SPLIT_FILE" \
  --coco-annotation "$COCO_ANNOTATION" \
  --device cpu \
  --output-dir "$OUTPUT_DIR" \
  "$@"
