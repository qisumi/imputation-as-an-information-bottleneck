#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"
read -r -a SEED_VALUES <<< "${SEEDS}"
EPOCHS="${EPOCHS:-100}"
MISSING_TYPES="${MISSING_TYPES:-MCAR MAR MNAR}"
METHODS=(BRITS MRNN SAITS Transformer USGAN)

mkdir -p runlog runlog/logs/missing_patterns

run_dataset() {
  local missing_type="$1"
  local dataset="$2"
  local oracle_expid="$3"
  local proposed_expid="$4"
  local result_csv="$5"
  local log_prefix="runlog/logs/missing_patterns/${missing_type}_${dataset}"
  local common_args=(
    --dataset "${dataset}"
    --seeds "${SEED_VALUES[@]}"
    --epochs "${EPOCHS}"
    --drop_rate 0.2
    --device "${DEVICE}"
    --missing_type "${missing_type}"
    --result_path "${result_csv}"
  )

  "${PYTHON_BIN}" src/run_mtgnn.py \
    --expid "${oracle_expid}" \
    "${common_args[@]}" \
    > "${log_prefix}_oracle.log" 2>&1

  local index=1
  for method in "${METHODS[@]}"; do
    "${PYTHON_BIN}" src/run_mtgnn.py \
      --imputation_method "${method}" \
      --expid "$((oracle_expid + index))" \
      "${common_args[@]}" \
      > "${log_prefix}_${method}.log" 2>&1
    index=$((index + 1))
  done

  "${PYTHON_BIN}" src/run_mtgnn.py \
    --expid "${proposed_expid}" \
    --mask_layer True \
    "${common_args[@]}" \
    > "${log_prefix}_mem.log" 2>&1
}

for missing_type in ${MISSING_TYPES}; do
  suffix="$(echo "${missing_type}" | tr 'A-Z' 'a-z')"
  result_csv="runlog/forecast_${suffix}20.csv"
  echo 'pred_model,dataset,imputation_model,mae,rmse,mape,kernel_size,dropout' > "${result_csv}"

  run_dataset "${missing_type}" ECG 10 15 "${result_csv}"
  run_dataset "${missing_type}" METR-LA 20 25 "${result_csv}"
done
