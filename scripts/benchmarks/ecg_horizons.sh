#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MISSING_TYPE="${MISSING_TYPE:-MCAR}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"
read -r -a SEED_VALUES <<< "${SEEDS}"
EPOCHS="${EPOCHS:-30}"
RESULT_CSV="${RESULT_CSV:-runlog/ecg_horizons_mcar20.csv}"
LOG_DIR="${LOG_DIR:-runlog/logs/ecg_horizons}"
BENCHMARK_TARGET="${BENCHMARK_TARGET:-all}"
METHODS=(BRITS MRNN SAITS Transformer USGAN)

mkdir -p runlog "${LOG_DIR}"

RUN_ORACLE=0
RUN_BASELINE=0
RUN_PROPOSED=0

case "${BENCHMARK_TARGET}" in
  all)
    RUN_ORACLE=1
    RUN_BASELINE=1
    RUN_PROPOSED=1
    ;;
  baseline)
    RUN_BASELINE=1
    ;;
  *)
    echo "[ERROR] Unsupported BENCHMARK_TARGET='${BENCHMARK_TARGET}'. Use 'all' or 'baseline'." >&2
    exit 1
    ;;
esac

if [[ "${BENCHMARK_TARGET}" == "all" || ! -s "${RESULT_CSV}" ]]; then
  echo 'pred_model,dataset,imputation_model,mae,rmse,mape,kernel_size,dropout' > "${RESULT_CSV}"
fi

cmd=("${PYTHON_BIN}" src/run_mtgnn.py)

run_suite() {
  local dataset="$1"
  local seq_len="$2"
  local oracle_expid="$3"
  local proposed_expid="$4"

  echo "[ECG-length][${MISSING_TYPE}-0.2] Running ${dataset}"

  local public_args=(
    --seeds "${SEED_VALUES[@]}"
    --epochs "${EPOCHS}"
    --drop_rate 0.2
    --missing_type "${MISSING_TYPE}"
    --seq_in_len "${seq_len}"
    --seq_out_len "${seq_len}"
    --device "${DEVICE}"
    --result_path "${RESULT_CSV}"
  )

  if [[ "${RUN_ORACLE}" == "1" ]]; then
    "${cmd[@]}" \
      --dataset "${dataset}" \
      --expid "${oracle_expid}" \
      "${public_args[@]}" \
      > "${LOG_DIR}/${dataset}_oracle.log" 2>&1
  fi

  if [[ "${RUN_BASELINE}" == "1" ]]; then
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
  fi

  if [[ "${RUN_PROPOSED}" == "1" ]]; then
    "${cmd[@]}" \
      --dataset "${dataset}" \
      --expid "${proposed_expid}" \
      --mask_layer True \
      "${public_args[@]}" \
      > "${LOG_DIR}/${dataset}_proposed.log" 2>&1
  fi
}

run_suite ECG24 24 724 730
run_suite ECG36 36 736 742
run_suite ECG48 48 748 754

echo "Saved ECG length prediction benchmark to ${RESULT_CSV} using ${MISSING_TYPE}"
