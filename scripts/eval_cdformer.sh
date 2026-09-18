#!/usr/bin/env bash
set -euo pipefail

# Evaluate the six NTU RGB+D 120 protocol/frame checkpoints used by the paper.
# Each checkpoint is evaluated only on the protocol for which it was trained.
#
# Required environment variables:
#   WEIGHTS_XSUB_16  WEIGHTS_XSUB_24  WEIGHTS_XSUB_32
#   WEIGHTS_XSET_16  WEIGHTS_XSET_24  WEIGHTS_XSET_32

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PKL_PATH="${PKL_PATH:-/data/nturgbd/ntu120_3danno.pkl}"
RESULTS_DIR="${RESULTS_DIR:-$CODE_DIR/results}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-32}"

if [[ ! -f "$PKL_PATH" ]]; then
  echo "ERROR: ntu120_3danno.pkl not found: $PKL_PATH"
  echo "Set PKL_PATH to the local PySKL NTU RGB+D 120 annotation file."
  exit 1
fi

mkdir -p "$RESULTS_DIR"

for PROTOCOL in xsub xset; do
  UPPER_PROTOCOL="${PROTOCOL^^}"

  for FRAMES in 16 24 32; do
    VAR="WEIGHTS_${UPPER_PROTOCOL}_${FRAMES}"
    WEIGHTS_PATH="${!VAR:-}"

    if [[ -z "$WEIGHTS_PATH" ]]; then
      echo "ERROR: $VAR is not set."
      exit 1
    fi
    if [[ ! -f "$WEIGHTS_PATH" ]]; then
      echo "ERROR: checkpoint not found: $WEIGHTS_PATH"
      exit 1
    fi

    echo "=== ${UPPER_PROTOCOL} | T=${FRAMES} ==="

    python "$CODE_DIR/graphormer_frames_reset_eval.py"       --pkl "$PKL_PATH"       --weights "$WEIGHTS_PATH"       --protocol "$PROTOCOL"       --frames "$FRAMES"       --d-model 192       --heads 8       --layers 12       --d-ff 2048       --dropout 0.15       --batch "$BATCH"       --device "$DEVICE"       --outdir "$RESULTS_DIR/${PROTOCOL}_T${FRAMES}"
  done
done

echo "Evaluation complete: $RESULTS_DIR"
