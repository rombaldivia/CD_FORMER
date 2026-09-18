#!/usr/bin/env bash
set -euo pipefail

echo "=== CD-Former manuscript-aligned evaluation ==="

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-/data/nturgbd}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
PKL_PATH="${PKL_PATH:-$DATA_DIR/ntu120_3danno.pkl}"
SCRIPT_PATH="${SCRIPT_PATH:-$CODE_DIR/graphormer_frames_reset_eval.py}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-32}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

if [ ! -f "$PKL_PATH" ]; then
  echo "Downloading NTU RGB+D 120 annotation file..."
  wget -O "$PKL_PATH" \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
fi

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "ERROR: evaluation script not found at $SCRIPT_PATH"
  exit 1
fi

for FRAMES in 16 24 32; do
  VAR="WEIGHTS_${FRAMES}"
  WEIGHTS_PATH="${!VAR:-}"

  if [ -z "$WEIGHTS_PATH" ]; then
    echo "ERROR: $VAR is not set."
    echo "Public model weights are intentionally not distributed in this repository."
    echo "Provide the local checkpoint corresponding to each temporal configuration."
    exit 1
  fi
  if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "ERROR: local checkpoint not found: $WEIGHTS_PATH"
    exit 1
  fi

  echo "=== Evaluating ${FRAMES} frames ==="
  python "$SCRIPT_PATH" \
    --pkl "$PKL_PATH" \
    --weights "$WEIGHTS_PATH" \
    --val_xsub xsub_val \
    --val_xset xset_val \
    --frames "$FRAMES" \
    --d_model 192 \
    --heads 8 \
    --layers 12 \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --num_workers 2 \
    --outdir "$RESULTS_DIR/metrics_eval_${FRAMES}f"
done

echo "=== Evaluation finished ==="
echo "Results saved under: $RESULTS_DIR"
