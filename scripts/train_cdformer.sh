#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PKL_PATH="${PKL_PATH:-/data/nturgbd/ntu120_3danno.pkl}"
PROTOCOL="${PROTOCOL:-xsub}"
FRAMES="${FRAMES:-16}"
RESULTS_DIR="${RESULTS_DIR:-$CODE_DIR/runs/$PROTOCOL/T${FRAMES}_seed42_split42}"

if [ ! -f "$PKL_PATH" ]; then
  echo "Set PKL_PATH to the existing ntu120_3danno.pkl annotation file."
  exit 1
fi

exec python "$CODE_DIR/cd_former_official.py" \
  --pkl "$PKL_PATH" --protocol "$PROTOCOL" --frames "$FRAMES" \
  --d_model "${D_MODEL:-192}" --heads "${HEADS:-6}" --layers "${LAYERS:-8}" \
  --batch "${BATCH:-32}" --epochs "${EPOCHS:-120}" --stop "${PATIENCE:-10}" \
  --device "${DEVICE:-cuda}" --outdir "$RESULTS_DIR" --resume auto "$@"
