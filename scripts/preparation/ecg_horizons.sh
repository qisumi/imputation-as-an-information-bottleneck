#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MISSING_TYPE="${MISSING_TYPE:-MCAR}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-30}"
RESULT_CSV="runlog/imputation_ecg_horizons_mcar20.csv"
LENGTHS=(24 36 48)

dataset_npz_ready() {
  local dataset_dir="$1"
  [[ -f "${dataset_dir}/train.npz" ]] && [[ -f "${dataset_dir}/val.npz" ]] && [[ -f "${dataset_dir}/test.npz" ]]
}

for length in "${LENGTHS[@]}"; do
  output_dir="rawdata/ECG${length}"
  if ! dataset_npz_ready "${output_dir}"; then
    echo "[ERROR] Missing prepared dataset files in ${output_dir}" >&2
    echo "[ERROR] Expected train.npz, val.npz, and test.npz to already exist." >&2
    exit 1
  fi

  echo "[ECG-length] Reusing existing dataset files in ${output_dir}"
done

"${PYTHON_BIN}" src/data_imputation.py \
  --device "${DEVICE}" \
  --epochs "${IMPUTATION_EPOCHS}" \
  --drop_rate 0.2 \
  --missing_type "${MISSING_TYPE}" \
  --datasets ECG24 ECG36 ECG48 \
  --result_path "${RESULT_CSV}"

echo "Saved ECG length imputation summary to ${RESULT_CSV} using ${MISSING_TYPE}"
