#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"
read -r -a SEED_VALUES <<< "${SEEDS}"
EPOCHS="${EPOCHS:-30}"
RESULT_CSV="runlog/ett_forecasting_mcar20.csv"
LOG_DIR="runlog/logs/ett_forecasting"
METHODS=(BRITS MRNN SAITS Transformer USGAN)

mkdir -p runlog "${LOG_DIR}"
echo 'pred_model,dataset,imputation_model,mae,rmse,mape,kernel_size,dropout' > "${RESULT_CSV}"

public_args=(
  --seeds "${SEED_VALUES[@]}"
  --epochs "${EPOCHS}"
  --drop_rate 0.2
  --missing_type MCAR
  --seq_in_len 12
  --seq_out_len 12
  --device "${DEVICE}"
  --result_path "${RESULT_CSV}"
)
cmd=("${PYTHON_BIN}" src/run_mtgnn.py)

run_suite() {
  local dataset="$1"
  local oracle_expid="$2"
  local proposed_expid="$3"

  echo "[MCAR-0.2][step=12][ETT] Running ${dataset}"

  "${cmd[@]}" \
    --dataset "${dataset}" \
    --expid "${oracle_expid}" \
    "${public_args[@]}" \
    > "${LOG_DIR}/${dataset}_oracle.log" 2>&1

  local idx=1
  for method in "${METHODS[@]}"; do
    "${cmd[@]}" \
      --dataset "${dataset}" \
      --imputation_method "${method}" \
      --pretrained_expid "${oracle_expid}" \
      --expid "$((oracle_expid + idx))" \
      "${public_args[@]}" \
      > "${LOG_DIR}/${dataset}_${method}.log" 2>&1
    idx=$((idx + 1))
  done

  "${cmd[@]}" \
    --dataset "${dataset}" \
    --expid "${proposed_expid}" \
    --mask_layer True \
    "${public_args[@]}" \
    > "${LOG_DIR}/${dataset}_proposed.log" 2>&1
}

run_suite ETTh1 610 616
run_suite ETTh2 620 626
run_suite ETTm1 630 636
run_suite ETTm2 640 646

echo "Saved prediction benchmark to ${RESULT_CSV}"
