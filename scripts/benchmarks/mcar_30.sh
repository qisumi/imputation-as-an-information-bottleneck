#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-42 2023 2024 2025 2026}"
EPOCHS="${EPOCHS:-100}"
public_args="--seeds ${SEEDS} --epochs ${EPOCHS} --drop_rate 0.3 --device ${DEVICE}"
cmd="${PYTHON_BIN} src/run_mtgnn.py"

$cmd --dataset ECG --expid 10 $public_args
$cmd --dataset ECG --imputation_method BRITS --expid 11 $public_args
$cmd --dataset ECG --imputation_method MRNN --expid 12 $public_args
$cmd --dataset ECG --imputation_method SAITS --expid 13 $public_args
$cmd --dataset ECG --imputation_method Transformer --expid 14 $public_args
$cmd --dataset ECG --imputation_method USGAN --expid 14 $public_args
$cmd --dataset METR-LA --expid 20 $public_args
$cmd --dataset METR-LA --imputation_method BRITS --expid 21 $public_args
$cmd --dataset METR-LA --imputation_method MRNN --expid 22 $public_args
$cmd --dataset METR-LA --imputation_method SAITS --expid 23 $public_args
$cmd --dataset METR-LA --imputation_method Transformer --expid 24 $public_args
$cmd --dataset METR-LA --imputation_method USGAN --expid 24 $public_args
$cmd --dataset SOLAR --expid 30 $public_args
$cmd --dataset SOLAR --imputation_method BRITS --expid 31 $public_args
$cmd --dataset SOLAR --imputation_method MRNN --expid 32 $public_args
$cmd --dataset SOLAR --imputation_method SAITS --expid 33 $public_args
$cmd --dataset SOLAR --imputation_method Transformer --expid 34 $public_args
$cmd --dataset SOLAR --imputation_method USGAN --expid 34 $public_args

$cmd --dataset ECG --expid 15 --mask_layer True $public_args
$cmd --dataset SOLAR --expid 35 --mask_layer True $public_args
$cmd --dataset METR-LA --expid 25 --mask_layer True $public_args
