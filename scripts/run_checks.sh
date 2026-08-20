#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_BOOTSTRAP=1 bash "${ROOT}/scripts/bootstrap.sh" >/dev/null
python -m py_compile \
  "${ROOT}/.work/baseline/examples/minimax_h3/model_training/train_preprocessed_csv.py" \
  "${ROOT}/.work/optimized/examples/minimax_h3/model_training/train_preprocessed_csv.py" \
  "${ROOT}/.work/optimized/diffsynth/models/minimax_h3_dit.py"
bash -n "${ROOT}/run_ab.sh" "${ROOT}/scripts/bootstrap.sh" \
  "${ROOT}/scripts/run_ab.sh" "${ROOT}/scripts/run_variant.sh" \
  "${ROOT}/scripts/profile_nsys.sh" "${ROOT}/scripts/run_zero_sweep.sh"
echo 'Bootstrap and static checks passed.'
