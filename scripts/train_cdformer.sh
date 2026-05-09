#!/usr/bin/env bash
set -e

echo "=== CD-Former Training / Fine-tuning ==="

CODE_DIR="${CODE_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-/data/nturgbd}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

PKL_PATH="${PKL_PATH:-$DATA_DIR/ntu120_3danno.pkl}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-90}"
EPOCHS="${EPOCHS:-120}"
SAVE_BEST="${SAVE_BEST:-$RESULTS_DIR/best_graphormer.pth}"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

if [ ! -f "$PKL_PATH" ]; then
  echo "Downloading NTU RGB+D 120 annotation file..."
  wget -O "$PKL_PATH" \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
else
  echo "Annotation file already exists: $PKL_PATH"
fi

if [ -n "${SCRIPT_PATH:-}" ]; then
  SELECTED_SCRIPT="$SCRIPT_PATH"
elif [ -f "$CODE_DIR/train-fine tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine tunning.py"
elif [ -f "$CODE_DIR/train-fine_tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine_tunning.py"
elif [ -f "$CODE_DIR/train-fine-tunning.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/train-fine-tunning.py"
elif [ -f "$CODE_DIR/graphormer_spatial.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/graphormer_spatial.py"
elif [ -f "$CODE_DIR/graphormer_spatial_reset_full.py" ]; then
  SELECTED_SCRIPT="$CODE_DIR/graphormer_spatial_reset_full.py"
else
  echo "ERROR: training script was not found. Available Python files:"
  find "$CODE_DIR" -maxdepth 3 -name "*.py" || true
  exit 1
fi

echo "Using script: $SELECTED_SCRIPT"
echo "Using PKL: $PKL_PATH"
echo "Using device: $DEVICE"
echo "Saving best checkpoint to: $SAVE_BEST"

python "$SELECTED_SCRIPT" \
  --pkl "$PKL_PATH" \
  --train_split xsub_train \
  --val_split xsub_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --freeze_n 12 \
  --unfreeze_epoch 15 \
  --batch "$BATCH" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --save_best "$SAVE_BEST"
