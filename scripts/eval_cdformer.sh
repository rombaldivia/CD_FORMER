#!/usr/bin/env bash
set -e

CODE_DIR="${CODE_DIR:-/code}"
PKL_PATH="${PKL_PATH:-/data/nturgbd/ntu120_3danno.pkl}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$CODE_DIR/checkpoints/CD_former.pth}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$(dirname "$PKL_PATH")"
mkdir -p /results

if [ ! -f "$PKL_PATH" ]; then
  echo "Downloading NTU RGB+D 120 annotation file..."
  wget -O "$PKL_PATH" \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
fi

if [ -n "${SCRIPT_PATH:-}" ]; then
  SELECTED_SCRIPT="$SCRIPT_PATH"
elif [ -f "$CODE_DIR/train-fine tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine tunning.py"
elif [ -f "$CODE_DIR/train-fine_tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine_tunning.py"
elif [ -f "$CODE_DIR/train-fine-tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine-tunning.py"
elif [ -f "$CODE_DIR/graphormer_spatial_reset_full.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/graphormer_spatial_reset_full.py"
else
  echo "ERROR: CD-Former script was not found. Available Python files:"
  find "$CODE_DIR" -maxdepth 3 -name "*.py" || true
  exit 1
fi

echo "Using script: $SELECTED_SCRIPT"
echo "Using checkpoint: $WEIGHTS_PATH"
echo "Using PKL: $PKL_PATH"
echo "Using device: $DEVICE"

python "$SELECTED_SCRIPT" \
  --eval_only \
  --weights "$WEIGHTS_PATH" \
  --pkl "$PKL_PATH" \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --device "$DEVICE"
