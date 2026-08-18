#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-30}"
RESULT_CSV="runlog/ett_imputation_mcar20.csv"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2)

mkdir -p runlog

echo "[MCAR-0.2][step=12][ETT] Preparing missing/imputed test sets"
"${PYTHON_BIN}" src/data_imputation.py \
  --device "${DEVICE}" \
  --epochs "${IMPUTATION_EPOCHS}" \
  --drop_rate 0.2 \
  --missing_type MCAR \
  --datasets "${DATASETS[@]}" \
  --result_path "${RESULT_CSV}"

echo "Saved imputation summary to ${RESULT_CSV}"
