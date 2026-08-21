#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/.work"
UPSTREAM_REPO="${DIFFSYNTH_UPSTREAM_REPO:-https://github.com/modelscope/DiffSynth-Studio.git}"
UPSTREAM_COMMIT="6343deda06b3e09efc9b1ce23c135c35a341d143"
LOCAL_UPSTREAM="${DIFFSYNTH_SOURCE:-}"

if [[ "${FORCE_BOOTSTRAP:-0}" == 1 ]]; then rm -rf "${WORK}"; fi
mkdir -p "${WORK}"
if [[ -f "${WORK}/.bootstrap-v6-ok" ]]; then
  echo "baseline=${WORK}/baseline"
  echo "baseline_measured=${WORK}/baseline-measured"
  echo "optimized=${WORK}/optimized"
  exit 0
fi
rm -rf "${WORK}/upstream" "${WORK}/baseline" "${WORK}/baseline-measured" "${WORK}/optimized"

if [[ -n "${LOCAL_UPSTREAM}" ]]; then
  [[ -d "${LOCAL_UPSTREAM}/.git" ]] || {
    echo "DIFFSYNTH_SOURCE must point to a git checkout: ${LOCAL_UPSTREAM}" >&2; exit 1;
  }
  git -C "${LOCAL_UPSTREAM}" cat-file -e "${UPSTREAM_COMMIT}^{commit}"
  mkdir -p "${WORK}/baseline"
  git -C "${LOCAL_UPSTREAM}" archive "${UPSTREAM_COMMIT}" | tar -x -C "${WORK}/baseline"
else
  git clone --filter=blob:none --no-checkout "${UPSTREAM_REPO}" "${WORK}/upstream"
  git -C "${WORK}/upstream" checkout --detach "${UPSTREAM_COMMIT}"
  mkdir -p "${WORK}/baseline"
  git -C "${WORK}/upstream" archive "${UPSTREAM_COMMIT}" | tar -x -C "${WORK}/baseline"
fi

# Reconstruct the user's uploaded Ref2VA baseline on the exact pinned upstream.
cp -a "${ROOT}/baseline/DiffSynth-Studio/." "${WORK}/baseline/"

# Baseline measurement tree = baseline + timing-only instrumentation. Apply
# from the repository root with an explicit generated-tree prefix; otherwise
# git apply resolves paths against the containing worktree instead of $PWD.
cp -a "${WORK}/baseline" "${WORK}/baseline-measured"
git -C "${ROOT}" apply --directory=.work/baseline-measured --recount --check \
  "${ROOT}/experiments/baseline_measurement.patch"
git -C "${ROOT}" apply --directory=.work/baseline-measured --recount \
  "${ROOT}/experiments/baseline_measurement.patch"
git -C "${ROOT}" apply --directory=.work/baseline-measured --recount --check \
  "${ROOT}/experiments/correctness_gate_baseline.patch"
git -C "${ROOT}" apply --directory=.work/baseline-measured --recount \
  "${ROOT}/experiments/correctness_gate_baseline.patch"

# Optimized always starts from exactly the same baseline. Experiment-facing
# examples/configs are stored as a readable overlay, while the five core runtime
# changes are applied from the exact verified patch set.
cp -a "${WORK}/baseline" "${WORK}/optimized"
if [[ -d "${ROOT}/optimized/DiffSynth-Studio/examples" ]]; then
  cp -a "${ROOT}/optimized/DiffSynth-Studio/examples/." "${WORK}/optimized/examples/"
fi
for patch_file in "${ROOT}"/patches/optimized-core/*.patch; do
  git -C "${ROOT}" apply --directory=.work/optimized --recount --check "${patch_file}"
  git -C "${ROOT}" apply --directory=.work/optimized --recount "${patch_file}"
done
git -C "${ROOT}" apply --directory=.work/optimized --recount --check \
  "${ROOT}/experiments/correctness_gate_optimized.patch"
git -C "${ROOT}" apply --directory=.work/optimized --recount \
  "${ROOT}/experiments/correctness_gate_optimized.patch"

touch "${WORK}/.bootstrap-v6-ok"
echo "baseline=${WORK}/baseline"
echo "baseline_measured=${WORK}/baseline-measured"
echo "optimized=${WORK}/optimized"
