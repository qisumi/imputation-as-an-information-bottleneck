#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-100}"
RATES="${RATES:-0.1 0.2 0.3}"

mkdir -p runlog

for rate in ${RATES}; do
  echo "[MCAR] Preparing missing and imputed test sets at rate=${rate}"
  "${PYTHON_BIN}" src/data_imputation.py \
    --device "${DEVICE}" \
    --epochs "${IMPUTATION_EPOCHS}" \
    --drop_rate "${rate}" \
    --missing_type MCAR \
    --datasets ECG METR-LA SOLAR \
    --result_path "runlog/imputation_mcar_${rate}.csv"
done

