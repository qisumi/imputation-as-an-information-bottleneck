#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MISSING_TYPE="${MISSING_TYPE:-MCAR}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-30}"
PREDICTION_EPOCHS="${PREDICTION_EPOCHS:-30}"
STEP12_RESULT="${STEP12_RESULT:-runlog/pred_with_imputation_bench.csv}"
STEP12_ARCHIVE="${STEP12_ARCHIVE:-${STEP12_RESULT}}"
LENGTH_RESULT="${LENGTH_RESULT:-runlog/ecg_horizons_mcar20.csv}"
OUTPUT_LONG="${OUTPUT_LONG:-runlog/ecg_horizons_mcar20_long.csv}"
OUTPUT_WIDE="${OUTPUT_WIDE:-runlog/ecg_horizons_mcar20_wide.csv}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"

mkdir -p runlog

dedupe_length_results() {
  "${PYTHON_BIN}" - "${LENGTH_RESULT}" <<'PY'
import os
import sys

import pandas as pd

path = sys.argv[1]
if not os.path.exists(path):
    raise SystemExit(0)

df = pd.read_csv(path)
subset = [column for column in ["pred_model", "dataset", "imputation_model"] if column in df.columns]
if subset:
    df = df.drop_duplicates(subset=subset, keep="last")
df.to_csv(path, index=False)
PY
}

if [[ ! -f "${STEP12_ARCHIVE}" ]]; then
  echo "[ERROR] Missing step-12 result for ${MISSING_TYPE}: ${STEP12_ARCHIVE}" >&2
  exit 1
fi

if [[ "${STEP12_ARCHIVE}" != "${STEP12_RESULT}" ]]; then
  cp "${STEP12_ARCHIVE}" "${STEP12_RESULT}"
fi
echo "[reuse-step12] Reusing ${MISSING_TYPE} step-12 results: ${STEP12_ARCHIVE}"

DEVICE="${DEVICE}" MISSING_TYPE="${MISSING_TYPE}" IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS}" \
  bash scripts/preparation/ecg_horizons.sh "$@"
DEVICE="${DEVICE}" MISSING_TYPE="${MISSING_TYPE}" EPOCHS="${PREDICTION_EPOCHS}" SEEDS="${SEEDS}" RESULT_CSV="${LENGTH_RESULT}" BENCHMARK_TARGET=all \
  bash scripts/benchmarks/ecg_horizons.sh

dedupe_length_results

"${PYTHON_BIN}" src/summarize_ecg_length_results.py \
  --step12_result "${STEP12_RESULT}" \
  --length_result "${LENGTH_RESULT}" \
  --output_long "${OUTPUT_LONG}" \
  --output_wide "${OUTPUT_WIDE}"
