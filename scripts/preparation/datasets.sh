#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ECG_SOURCE="${ECG_SOURCE:-rawdata/ECG5000/ECG5000_TRAIN.txt}"
METR_SOURCE="${METR_SOURCE:-rawdata/metr-la.h5}"
SOLAR_SOURCE="${SOLAR_SOURCE:-rawdata/solar_AL.txt}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required raw file: ${path}" >&2
    exit 1
  fi
}

require_file "${ECG_SOURCE}"
require_file "${ECG_SOURCE/_TRAIN/_TEST}"
require_file "${METR_SOURCE}"
require_file "${SOLAR_SOURCE}"

"${PYTHON_BIN}" src/generate_data.py \
  --ds_name ecg \
  --length 12 \
  --dataset_filename "${ECG_SOURCE}" \
  --output_dir rawdata/ECG

for length in 24 36 48; do
  "${PYTHON_BIN}" src/generate_data.py \
    --ds_name ecg \
    --length "${length}" \
    --dataset_filename "${ECG_SOURCE}" \
    --output_dir "rawdata/ECG${length}"
done

"${PYTHON_BIN}" src/generate_data.py \
  --ds_name metr-la \
  --length 12 \
  --dataset_filename "${METR_SOURCE}" \
  --output_dir rawdata/METR-LA

"${PYTHON_BIN}" src/generate_data.py \
  --ds_name solar \
  --length 12 \
  --dataset_filename "${SOLAR_SOURCE}" \
  --output_dir rawdata/SOLAR

bash scripts/preparation/ett.sh

