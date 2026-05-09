#!/usr/bin/env bash
set -e

python graphormer_spatial_reset_full.py \
  --pkl ntu120_3danno.pkl \
  --train_split xsub_train \
  --val_split xsub_val \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --reset_head \
  --freeze_n 12 \
  --unfreeze_epoch 15 \
  --device cuda
