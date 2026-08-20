#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/config.env" ]] && source "${ROOT}/config.env"
: "${CSV_PATH:?Copy config.example.env to config.env and set CSV_PATH}"

bash "${ROOT}/scripts/bootstrap.sh"
bash "${ROOT}/scripts/check_env.sh"

RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/ab-$(date +%Y%m%d-%H%M%S)}"
MODES="${MODES:-baseline optimized-core optimized-data optimized-zero}"
export RESULT_ROOT
mkdir -p "${RESULT_ROOT}"

for mode in ${MODES}; do
  echo "========== ${mode} =========="
  RUN_NAME="${mode}" bash "${ROOT}/scripts/run_variant.sh" "${mode}"
done

python "${ROOT}/scripts/summarize_benchmarks.py" "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary.md"
echo "A/B results: ${RESULT_ROOT}"
