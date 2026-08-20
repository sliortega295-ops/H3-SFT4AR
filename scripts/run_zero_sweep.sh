#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL="${ROOT}/optimized/DiffSynth-Studio/examples/minimax_h3/model_training/full"
: "${CSV_PATH:?Set CSV_PATH to the preprocessed CSV manifest.}"

RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/zero-sweep-$(date +%Y%m%d-%H%M%S)}"
export RESULT_ROOT
# Keep workers=0 and all data-path settings fixed so this sweep changes only
# the DeepSpeed/ZeRO scheduling configuration.
export DATA_WORKERS=0
mkdir -p "${RESULT_ROOT}"

run_one() {
  local name="$1"
  local config="$2"
  echo "=== ${name} ==="
  RUN_NAME="${name}" \
  ZERO3_ACCELERATE_CONFIG="${config}" \
  bash "${ROOT}/scripts/run_benchmark_variant.sh" optimized-zero
}

run_one zero-original \
  "${FULL}/accelerate_config_zero3.yaml"
run_one zero-prefetch-50m \
  "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_50m.yaml"
run_one zero-prefetch-160m \
  "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_160m.yaml"
run_one zero-prefetch-320m \
  "${FULL}/zero3_sweeps/accelerate_zero3_prefetch_320m.yaml"

python "${ROOT}/scripts/summarize_benchmarks.py" "${RESULT_ROOT}" \
  | tee "${RESULT_ROOT}/summary.md"
echo "ZeRO sweep outputs: ${RESULT_ROOT}"
