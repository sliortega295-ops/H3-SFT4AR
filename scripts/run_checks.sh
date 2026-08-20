#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_BOOTSTRAP=1 bash "${ROOT}/scripts/bootstrap.sh" >/dev/null

python -m py_compile \
  "${ROOT}/.work/baseline/examples/minimax_h3/model_training/train_preprocessed_csv.py" \
  "${ROOT}/.work/optimized/examples/minimax_h3/model_training/train_preprocessed_csv.py" \
  "${ROOT}/.work/optimized/diffsynth/models/minimax_h3_dit.py" \
  "${ROOT}/.work/optimized/diffsynth/pipelines/minimax_h3_audio_video.py"

bash -n "${ROOT}/run_ab.sh" "${ROOT}/scripts/bootstrap.sh" \
  "${ROOT}/scripts/run_ab.sh" "${ROOT}/scripts/run_variant.sh" \
  "${ROOT}/scripts/profile_nsys.sh" "${ROOT}/scripts/run_zero_sweep.sh"

while read -r expected path; do
  actual="$(git hash-object "${ROOT}/.work/optimized/${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "optimized hash mismatch: ${path}" >&2
    echo "  expected ${expected}" >&2
    echo "  actual   ${actual}" >&2
    exit 1
  fi
done < "${ROOT}/experiments/optimized_expected_git_hashes.txt"

echo 'Bootstrap/static checks passed; all 17 optimized blobs match the validated tree.'
