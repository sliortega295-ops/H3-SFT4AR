#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/config.env" ]] && source "${ROOT}/config.env"

missing=0
for cmd in python accelerate patch git; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "missing command: ${cmd}" >&2; missing=1
  fi
done
: "${CSV_PATH:=}"
if [[ -z "${CSV_PATH}" || ! -f "${CSV_PATH}" ]]; then
  echo "CSV_PATH is missing or not a file: ${CSV_PATH:-<unset>}" >&2; missing=1
fi
if [[ -n "${DIT_MODEL_PATH:-}" ]] && ! compgen -G "${DIT_MODEL_PATH}" >/dev/null; then
  echo "warning: DIT_MODEL_PATH matched no files; MODEL_SPEC fallback will be used." >&2
fi
python - <<'PY' || missing=1
mods=['torch','accelerate','deepspeed']
for m in mods:
    try:
        x=__import__(m)
        print(f'{m}={getattr(x,"__version__","unknown")}')
    except Exception as e:
        raise SystemExit(f'missing python dependency {m}: {e}')
PY
[[ "${missing}" == 0 ]] || exit 1
