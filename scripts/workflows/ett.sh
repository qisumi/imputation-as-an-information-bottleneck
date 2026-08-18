#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-cuda:0}"
IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS:-30}"
PREDICTION_EPOCHS="${PREDICTION_EPOCHS:-30}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"

DEVICE="${DEVICE}" bash scripts/preparation/ett.sh
DEVICE="${DEVICE}" IMPUTATION_EPOCHS="${IMPUTATION_EPOCHS}" bash scripts/preparation/ett_imputation.sh
DEVICE="${DEVICE}" EPOCHS="${PREDICTION_EPOCHS}" SEEDS="${SEEDS}" bash scripts/benchmarks/ett_mcar20.sh
