#!/usr/bin/env bash
set -e

echo "=== CD-Former Evaluation ==="

CODE_DIR="${CODE_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-/data/nturgbd}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

PKL_PATH="${PKL_PATH:-$DATA_DIR/ntu120_3danno.pkl}"
WEIGHTS_PATH="${WEIGHTS_PATH:-}"
SCRIPT_PATH="${SCRIPT_PATH:-$CODE_DIR/graphormer_frames_reset_eval.py}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-32}"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

if [ ! -f "$PKL_PATH" ]; then
  echo "Downloading NTU RGB+D 120 annotation file..."
  wget -O "$PKL_PATH" \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
else
  echo "Annotation file already exists: $PKL_PATH"
fi

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "ERROR: evaluation script not found at $SCRIPT_PATH"
  echo "Available Python files:"
  find "$CODE_DIR" -maxdepth 3 -name "*.py" || true
  exit 1
fi

if [ -z "$WEIGHTS_PATH" ]; then
  echo "ERROR: no checkpoint path supplied."
  echo "Model weights are not distributed in this repository."
  echo "Set WEIGHTS_PATH to a local checkpoint file before running evaluation."
  exit 1
fi

if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "ERROR: local checkpoint not found at $WEIGHTS_PATH"
  exit 1
fi

echo "Using script: $SCRIPT_PATH"
echo "Using checkpoint: $WEIGHTS_PATH"
echo "Using PKL: $PKL_PATH"
echo "Using device: $DEVICE"

for FRAMES in 16 24 32; do
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
