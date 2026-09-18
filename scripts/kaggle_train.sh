#!/usr/bin/env bash
set -euo pipefail

# Dual-T4 Kaggle launcher for protocol-separated CD-Former runs.
# XSUB runs on GPU 0 and XSET runs on GPU 1. No checkpoint is stored in Git.

PKL_PATH="${PKL_PATH:-/kaggle/input/ntu120/ntu120_3danno.pkl}"
FRAMES="${FRAMES:-32}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/cdformer_runs}"
MICRO_BATCH="${MICRO_BATCH:-20}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-20}"
WORKERS="${WORKERS:-2}"

EXTRA_XSUB=()
EXTRA_XSET=()

# Optional second-stage head reset. Point these variables to private/local
# checkpoints created by an earlier stage.
if [[ -n "${INIT_XSUB:-}" ]]; then
  EXTRA_XSUB+=(--init-checkpoint "$INIT_XSUB" --reset-head)
fi
if [[ -n "${INIT_XSET:-}" ]]; then
  EXTRA_XSET+=(--init-checkpoint "$INIT_XSET" --reset-head)
fi

mkdir -p "$OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES=0 python -u train_cdformer.py   --pkl "$PKL_PATH"   --protocol xsub   --frames "$FRAMES"   --seed "$SEED"   --micro-batch "$MICRO_BATCH"   --accumulation-steps "$ACCUMULATION_STEPS"   --workers "$WORKERS"   --outdir "$OUTPUT_ROOT/xsub/T${FRAMES}_seed${SEED}"   "${EXTRA_XSUB[@]}"   > "$OUTPUT_ROOT/xsub_T${FRAMES}_seed${SEED}.log" 2>&1 &
PID_XSUB=$!

CUDA_VISIBLE_DEVICES=1 python -u train_cdformer.py   --pkl "$PKL_PATH"   --protocol xset   --frames "$FRAMES"   --seed "$SEED"   --micro-batch "$MICRO_BATCH"   --accumulation-steps "$ACCUMULATION_STEPS"   --workers "$WORKERS"   --outdir "$OUTPUT_ROOT/xset/T${FRAMES}_seed${SEED}"   "${EXTRA_XSET[@]}"   > "$OUTPUT_ROOT/xset_T${FRAMES}_seed${SEED}.log" 2>&1 &
PID_XSET=$!

echo "XSUB PID: $PID_XSUB (GPU 0)"
echo "XSET PID: $PID_XSET (GPU 1)"
echo "Logs: $OUTPUT_ROOT"

wait "$PID_XSUB"
wait "$PID_XSET"

echo "Both protocol runs finished."
