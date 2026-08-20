#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CSV_PATH:?Set CSV_PATH to the preprocessed CSV manifest.}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/ab-$(date +%Y%m%d-%H%M%S)}"
MODES="${MODES:-baseline optimized-core optimized-cache}"
export RESULT_ROOT
mkdir -p "${RESULT_ROOT}"

for mode in ${MODES}; do
  "${ROOT}/scripts/run_benchmark_variant.sh" "${mode}"
done
python "${ROOT}/scripts/summarize_benchmarks.py" "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary.md"
echo "Results: ${RESULT_ROOT}"
