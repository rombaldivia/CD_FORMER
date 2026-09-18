#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PKL_PATH="${PKL_PATH:-/data/nturgbd/ntu120_3danno.pkl}"
RESULTS_DIR="${RESULTS_DIR:-$CODE_DIR/results}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-32}"

if [[ ! -f "$PKL_PATH" ]]; then
  echo "ntu120_3danno.pkl not found: $PKL_PATH" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

for PROTOCOL in xsub xset; do
  PROTOCOL_UPPER="${PROTOCOL^^}"

  for FRAMES in 16 24 32; do
    VAR="WEIGHTS_${PROTOCOL_UPPER}_${FRAMES}"
    WEIGHTS_PATH="${!VAR:-}"

    if [[ -z "$WEIGHTS_PATH" ]]; then
      echo "$VAR is not set" >&2
      exit 1
    fi

    if [[ ! -f "$WEIGHTS_PATH" ]]; then
      echo "checkpoint not found: $WEIGHTS_PATH" >&2
      exit 1
    fi

    echo "${PROTOCOL_UPPER} T${FRAMES}"

    python "$CODE_DIR/graphormer_frames_reset_eval.py" \
      --pkl "$PKL_PATH" \
      --weights "$WEIGHTS_PATH" \
      --protocol "$PROTOCOL" \
      --frames "$FRAMES" \
      --d-model 192 \
      --heads 8 \
      --layers 12 \
      --d-ff 2048 \
      --dropout 0.15 \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --outdir "$RESULTS_DIR/${PROTOCOL}_T${FRAMES}"
  done
done
