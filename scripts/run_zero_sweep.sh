#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/config.env" ]] && source "${ROOT}/config.env"
: "${CSV_PATH:?Set CSV_PATH in config.env}"
bash "${ROOT}/scripts/bootstrap.sh" >/dev/null
FULL="${ROOT}/.work/optimized/examples/minimax_h3/model_training/full"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/zero-sweep-$(date +%Y%m%d-%H%M%S)}"
export RESULT_ROOT DATA_WORKERS=0
mkdir -p "${RESULT_ROOT}"
run_one() {
  local name="$1" cfg="$2"
  RUN_NAME="${name}" ZERO3_ACCELERATE_CONFIG="${cfg}" \
    bash "${ROOT}/scripts/run_variant.sh" optimized-zero
}
run_one zero-original "${FULL}/accelerate_config_zero3.yaml"
run_one zero-prefetch-50m "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_50m.yaml"
run_one zero-prefetch-160m "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_160m.yaml"
run_one zero-prefetch-320m "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_320m.yaml"
python "${ROOT}/scripts/summarize_benchmarks.py" "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary.md"
