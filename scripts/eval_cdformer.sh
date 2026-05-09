#!/usr/bin/env bash
set -e

PKL_PATH="${PKL_PATH:-/data/nturgbd/ntu120_3danno.pkl}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/code/checkpoints/CD_former.pth}"
SCRIPT_PATH="${SCRIPT_PATH:-/code/graphormer_spatial_reset_full.py}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "$(dirname "$PKL_PATH")"

if [ ! -f "$PKL_PATH" ]; then
  echo "Downloading NTU RGB+D 120 annotation file..."
  wget -O "$PKL_PATH" \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
fi

python "$SCRIPT_PATH" \
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
