#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"
EPOCHS="${EPOCHS:-100}"

OUT_DIR="runlog/backbone_mem"
mkdir -p "${OUT_DIR}"
mkdir -p runlog

DATASETS=("ECG" "METR-LA" "SOLAR")
BACKBONES=("DLinear" "TQNet")

MISSING_TYPE="MCAR"
MASK_BACKBONE="conv2d"

public_args="--seeds ${SEEDS} --epochs ${EPOCHS} --drop_rate 0.2 --device ${DEVICE}"
cmd="${PYTHON_BIN} src/run_mtgnn.py"

for DS in "${DATASETS[@]}"; do
  for BK in "${BACKBONES[@]}"; do
    bk_lower=$(echo "${BK}" | tr 'A-Z' 'a-z')

    RESULT_CSV="runlog/backbone_mem.csv"
    echo 'pred_model,dataset,imputation_model,mae,rmse,mape,kernel_size,dropout' > "${RESULT_CSV}"

    log_name="${OUT_DIR}/exp3_${DS}_${bk_lower}_${MASK_BACKBONE}.log"

    echo "==============================================="
    echo "Dataset=${DS}, missing_type=${MISSING_TYPE}, mask_backbone=${MASK_BACKBONE}, backbone=${BK}"
    echo "Log: ${log_name}"
    echo "==============================================="

    ${cmd} \
      --dataset "${DS}" \
      --expid 300 \
      --mask_layer True \
      --mask_backbone "${MASK_BACKBONE}" \
      --missing_type "${MISSING_TYPE}" \
      --backbone "${BK}" \
      --result_path "${RESULT_CSV}" \
      ${public_args} \
      > "${log_name}" 2>&1

    cp "${RESULT_CSV}" "${OUT_DIR}/exp3_${DS}_${bk_lower}_${MASK_BACKBONE}.csv"
  done
done
