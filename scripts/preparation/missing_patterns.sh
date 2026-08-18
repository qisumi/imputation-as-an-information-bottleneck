#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-100}"
MISSING_TYPES="${MISSING_TYPES:-MAR MNAR}"

mkdir -p runlog

for missing_type in ${MISSING_TYPES}; do
  suffix="$(echo "${missing_type}" | tr 'A-Z' 'a-z')"
  echo "[${missing_type}] Preparing ECG and METR-LA at missing rate 0.2"
  "${PYTHON_BIN}" src/data_imputation.py \
    --device "${DEVICE}" \
    --epochs "${IMPUTATION_EPOCHS}" \
    --drop_rate 0.2 \
    --missing_type "${missing_type}" \
    --datasets ECG METR-LA \
    --result_path "runlog/imputation_${suffix}_0.2.csv"
done

