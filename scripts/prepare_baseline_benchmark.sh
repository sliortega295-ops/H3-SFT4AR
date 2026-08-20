#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/.work/baseline-measured}"
rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"
cp -a "${ROOT}/baseline/DiffSynth-Studio" "${DEST}"
(
  cd "${DEST}"
  patch -s -p1 < "${ROOT}/experiments/baseline_measurement.patch"
)
printf '%s\n' "${DEST}"
