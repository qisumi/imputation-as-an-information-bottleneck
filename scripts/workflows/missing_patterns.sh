#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RATES=0.2 bash scripts/preparation/main_mcar.sh
bash scripts/preparation/missing_patterns.sh
bash scripts/benchmarks/missing_patterns.sh
