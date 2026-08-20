#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${ROOT}/baseline/DiffSynth-Studio"
OPT="${ROOT}/optimized/DiffSynth-Studio"

export PYTHONDONTWRITEBYTECODE=1
(
  cd "${BASE}"
  sha256sum -c "${ROOT}/BASELINE_SHA256SUMS.txt" >/dev/null
)
echo "Baseline checksum validation passed."
python "${ROOT}/tests/test_semantic_equivalence.py"
python "${ROOT}/tests/compare_semantic_probe.py"

python - <<PY
from pathlib import Path
root = Path(${ROOT@Q})
paths = [
    root / "optimized/DiffSynth-Studio/examples/minimax_h3/model_training/preprocessed_csv_dataset.py",
    root / "optimized/DiffSynth-Studio/examples/minimax_h3/model_training/train_preprocessed_csv.py",
    root / "optimized/DiffSynth-Studio/diffsynth/diffusion/training_module.py",
    root / "optimized/DiffSynth-Studio/diffsynth/diffusion/runner.py",
    root / "optimized/DiffSynth-Studio/diffsynth/diffusion/logger.py",
    root / "optimized/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py",
    root / "optimized/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py",
    root / "scripts/summarize_benchmarks.py",
    root / "tests/semantic_probe.py",
    root / "tests/compare_semantic_probe.py",
]
for path in paths:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Python syntax validation passed.")
PY

for script in \
  "${OPT}/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV.sh" \
  "${OPT}/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV-Optimized.sh" \
  "${ROOT}/scripts/prepare_baseline_benchmark.sh" \
  "${ROOT}/scripts/run_benchmark_variant.sh" \
  "${ROOT}/scripts/run_ab.sh" \
  "${ROOT}/scripts/profile_nsys.sh" \
  "${ROOT}/scripts/run_zero_sweep.sh" \
  "${ROOT}/scripts/run_checks.sh"; do
  bash -n "${script}"
done

echo "Shell syntax validation passed."

python - <<PY
import json
from pathlib import Path
import yaml
root = Path(${ROOT@Q})
full = root / "optimized/DiffSynth-Studio/examples/minimax_h3/model_training/full"
for path in [full / "deepspeed_zero3_overlap.json", *sorted((full / "zero3_sweeps").glob("*.json"))]:
    json.loads(path.read_text(encoding="utf-8"))
for path in [full / "accelerate_config_zero3_optimized.yaml", *sorted((full / "zero3_sweeps").glob("*.yaml"))]:
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("JSON/YAML validation passed.")
PY

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
cp -a "${BASE}" "${TMP}/baseline-measured"
(
  cd "${TMP}/baseline-measured"
  patch --dry-run -s -p1 < "${ROOT}/experiments/baseline_measurement.patch"
  patch -s -p1 < "${ROOT}/experiments/baseline_measurement.patch"
)
python - <<PY
from pathlib import Path
root = Path(${TMP@Q}) / "baseline-measured"
for rel in [
    "diffsynth/diffusion/runner.py",
    "diffsynth/diffusion/logger.py",
    "examples/minimax_h3/model_training/train_preprocessed_csv.py",
]:
    path = root / rel
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Baseline instrumentation patch validation passed.")
PY

cmp \
  "${BASE}/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV.sh" \
  "${OPT}/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV.sh"

PATCH_TMP="$(mktemp -d)"
cp -a "${BASE}" "${PATCH_TMP}/DiffSynth-Studio"
(
  cd "${PATCH_TMP}/DiffSynth-Studio"
  patch --dry-run -s -p1 < "${ROOT}/patches/scheduling_optimizations.patch"
)
rm -rf "${PATCH_TMP}"

if find "${ROOT}/baseline" "${ROOT}/optimized" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.ckpt' \) \
  | grep -q .; then
  echo "Unexpected model/cache artifact found in source trees." >&2
  exit 1
fi

if find "${ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' \) | grep -q .; then
  echo "Generated Python bytecode found in repository tree." >&2
  exit 1
fi

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "${ROOT}"/scripts/*.sh \
    "${OPT}/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA-Preprocessed-CSV-Optimized.sh"
fi

echo "All semantic, patch, and static checks passed."
