#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CSV_PATH:?Set CSV_PATH to the preprocessed CSV manifest.}"
command -v nsys >/dev/null 2>&1 || { echo "nsys not found in PATH" >&2; exit 1; }

PROFILE_MODE="${PROFILE_MODE:-optimized-core}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURE_STEPS="${PROFILE_STEPS:-5}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/nsys-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${PROFILE_MODE}}"
REPORT="${REPORT:-${RESULT_ROOT}/${RUN_NAME}}"
mkdir -p "${RESULT_ROOT}"

export WARMUP_STEPS MEASURE_STEPS RESULT_ROOT RUN_NAME
export ENABLE_NVTX=1
export PROFILE_CUDA_CAPTURE=1

nsys profile \
  --force-overwrite=true \
  --output="${REPORT}" \
  --trace="${NSYS_TRACE:-cuda,nvtx,osrt,nccl,cublas,cudnn}" \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  bash "${ROOT}/scripts/run_benchmark_variant.sh" "${PROFILE_MODE}"

echo "Nsight Systems report: ${REPORT}.nsys-rep"
