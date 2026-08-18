#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/preparation/main_mcar.sh
bash scripts/benchmarks/mcar_10.sh
bash scripts/benchmarks/mcar_20.sh
bash scripts/benchmarks/mcar_30.sh

