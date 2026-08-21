#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v accelerate >/dev/null || {
  echo "Activate the repository Python environment before running this gate." >&2
  exit 2
}

RUN_ROOT="${RUN_ROOT:-${ROOT}/results/correctness-$(date +%Y%m%d-%H%M%S)}"
CORRECTNESS_STEPS="${CORRECTNESS_STEPS:-3}"
COMMON_ENV=(
  CORRECTNESS_GATE=1
  WARMUP_STEPS=0
  MEASURE_STEPS=0
  MAX_TRAIN_STEPS="${CORRECTNESS_STEPS}"
  DATA_WORKERS=0
  DATASET_REPEAT=100
  RESULT_ROOT="${RUN_ROOT}"
)

env "${COMMON_ENV[@]}" RUN_NAME=baseline-a bash "${ROOT}/scripts/run_variant.sh" baseline
env "${COMMON_ENV[@]}" RUN_NAME=baseline-b bash "${ROOT}/scripts/run_variant.sh" baseline

python "${ROOT}/scripts/compare_correctness.py" \
  "${RUN_ROOT}/baseline-a" \
  "${RUN_ROOT}/baseline-b" \
  --steps "${CORRECTNESS_STEPS}" \
  --output "${RUN_ROOT}/baseline_repeat_comparison.json"

env "${COMMON_ENV[@]}" RUN_NAME=optimized-zero bash "${ROOT}/scripts/run_variant.sh" optimized-zero

python "${ROOT}/scripts/compare_correctness.py" \
  "${RUN_ROOT}/baseline-a" \
  "${RUN_ROOT}/baseline-b" \
  "${RUN_ROOT}/optimized-zero" \
  --steps "${CORRECTNESS_STEPS}" \
  --output "${RUN_ROOT}/correctness_comparison.json"

printf 'correctness_gate_pass run_root=%s\n' "${RUN_ROOT}"
