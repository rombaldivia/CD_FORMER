#!/usr/bin/env bash
set -e

python graphormer_spatial_reset_full.py \
  --eval_only \
  --weights best_graphormer.pth \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --device cuda
