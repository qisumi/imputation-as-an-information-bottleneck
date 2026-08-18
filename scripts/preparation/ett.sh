#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2)

for dataset in "${DATASETS[@]}"; do
  csv_path="rawdata/${dataset}/${dataset}.csv"
  output_dir="rawdata/${dataset}"

  if [[ ! -f "${csv_path}" ]]; then
    echo "[ERROR] Missing raw csv for ${dataset}: ${csv_path}" >&2
    exit 1
  fi

  echo "[ETT] Preparing ${dataset} from ${csv_path}"
  "${PYTHON_BIN}" src/generate_data.py \
    --ds_name "${dataset}" \
    --length 12 \
    --output_dir "${output_dir}" \
    --dataset_filename "${csv_path}"
done
