#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/config.env" ]] && source "${ROOT}/config.env"
: "${CSV_PATH:?Set CSV_PATH in config.env}"
command -v nsys >/dev/null 2>&1 || { echo 'nsys not found in PATH' >&2; exit 1; }
MODE="${PROFILE_MODE:-optimized-core}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export MEASURE_STEPS="${PROFILE_STEPS:-5}"
export RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/nsys-$(date +%Y%m%d-%H%M%S)}"
export RUN_NAME="${RUN_NAME:-${MODE}}"
export ENABLE_NVTX=1 PROFILE_CUDA_CAPTURE=1
mkdir -p "${RESULT_ROOT}"
REPORT="${REPORT:-${RESULT_ROOT}/${RUN_NAME}}"
nsys profile --force-overwrite=true --output="${REPORT}" \
  --trace="${NSYS_TRACE:-cuda,nvtx,osrt,nccl,cublas,cudnn}" \
  --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  bash "${ROOT}/scripts/run_variant.sh" "${MODE}"
echo "Nsight report: ${REPORT}.nsys-rep"
