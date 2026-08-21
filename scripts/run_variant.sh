#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/config.env" ]] && source "${ROOT}/config.env"
MODE="${1:?Usage: $0 baseline|optimized-core|optimized-data|optimized-zero}"
: "${CSV_PATH:?Set CSV_PATH in config.env or environment}"

bash "${ROOT}/scripts/bootstrap.sh" >/dev/null

SEED="${SEED:-20260820}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-30}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$((WARMUP_STEPS + MEASURE_STEPS))}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DATA_WORKERS="${DATA_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TEXT_CACHE_SIZE="${TEXT_CACHE_SIZE:-64}"
PACKED_LAYOUT_CACHE_SIZE="${PACKED_LAYOUT_CACHE_SIZE:-64}"
MODEL_SPEC="${MODEL_SPEC:-MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors}"
DIT_MODEL_PATH="${DIT_MODEL_PATH:-}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/benchmark-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${MODE}}"
OUTPUT_PATH="${RESULT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS=()
if [[ -n "${DIT_MODEL_PATH}" ]] && compgen -G "${DIT_MODEL_PATH}" >/dev/null; then
  MODEL_ARGS=(--dit_model_path "${DIT_MODEL_PATH}")
else
  MODEL_ARGS=(--model_id_with_origin_paths "${MODEL_SPEC}")
fi

DATA_ARGS=()
case "${MODE}" in
  baseline)
    CODE_ROOT="${ROOT}/.work/baseline-measured"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    ;;
  optimized-core)
    CODE_ROOT="${ROOT}/.work/optimized"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    DATA_ARGS=(
      --dataset_num_workers 0
      --no-dataloader_pin_memory --no-dataloader_persistent_workers
      --no-dataloader_non_blocking --no-cache_mmap
      --text_cache_size 0 --packed_layout_cache_size 0
    )
    ;;
  optimized-data)
    CODE_ROOT="${ROOT}/.work/optimized"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    DATA_ARGS=(
      --dataset_num_workers "${DATA_WORKERS}"
      --dataloader_pin_memory --dataloader_non_blocking --cache_mmap
      --text_cache_size "${TEXT_CACHE_SIZE}"
      --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
    )
    if (( DATA_WORKERS > 0 )); then
      DATA_ARGS+=(--dataloader_persistent_workers --dataloader_prefetch_factor "${PREFETCH_FACTOR}")
    else
      DATA_ARGS+=(--no-dataloader_persistent_workers)
    fi
    ;;
  optimized-zero)
    CODE_ROOT="${ROOT}/.work/optimized"
    ACCELERATE_CONFIG="${ZERO3_ACCELERATE_CONFIG:-${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3_optimized.yaml}"
    DATA_ARGS=(
      --dataset_num_workers "${DATA_WORKERS}"
      --dataloader_pin_memory --dataloader_non_blocking --cache_mmap
      --text_cache_size "${TEXT_CACHE_SIZE}"
      --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
    )
    if (( DATA_WORKERS > 0 )); then
      DATA_ARGS+=(--dataloader_persistent_workers --dataloader_prefetch_factor "${PREFETCH_FACTOR}")
    else
      DATA_ARGS+=(--no-dataloader_persistent_workers)
    fi
    ;;
  *) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

PROFILE_ARGS=()
[[ "${ENABLE_NVTX:-0}" != 1 ]] || PROFILE_ARGS+=(--enable_nvtx)
[[ "${PROFILE_CUDA_CAPTURE:-0}" != 1 ]] || PROFILE_ARGS+=(--profile_cuda_capture)

CORRECTNESS_ARGS=()
if [[ "${CORRECTNESS_GATE:-0}" == 1 ]]; then
  if (( WARMUP_STEPS != 0 || MEASURE_STEPS != 0 )); then
    echo "Correctness gate requires WARMUP_STEPS=0 and MEASURE_STEPS=0." >&2
    exit 2
  fi
  CORRECTNESS_ARGS=(
    --fixed_sample_schedule
    --fixed_chunk_seed "${SEED}"
    --correctness_report "${OUTPUT_PATH}/correctness"
  )
fi

CMD=(
  accelerate launch --config_file "${ACCELERATE_CONFIG}"
  examples/minimax_h3/model_training/train_preprocessed_csv.py
  --dataset_metadata_path "${CSV_PATH}"
  --dataset_repeat "${DATASET_REPEAT}"
  --chunk_num_frames 56 --video_chunk_latents 15 --audio_chunk_latents 93
  "${MODEL_ARGS[@]}"
  --learning_rate "${LEARNING_RATE}" --num_epochs 1
  --remove_prefix_in_ckpt 'pipe.dit.' --output_path "${OUTPUT_PATH}"
  --trainable_models dit --use_gradient_checkpointing --find_unused_parameters
  --save_steps 1000000000
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --benchmark_warmup_steps "${WARMUP_STEPS}"
  --benchmark_steps "${MEASURE_STEPS}"
  --skip_final_save --task sft
  "${DATA_ARGS[@]}" "${PROFILE_ARGS[@]}" "${CORRECTNESS_ARGS[@]}"
)

{
  echo "mode=${MODE}"
  echo "code_root=${CODE_ROOT}"
  echo "accelerate_config=${ACCELERATE_CONFIG}"
  echo "seed=${SEED}"
  echo "warmup_steps=${WARMUP_STEPS}"
  echo "measure_steps=${MEASURE_STEPS}"
  echo "max_train_steps=${MAX_TRAIN_STEPS}"
  echo "correctness_gate=${CORRECTNESS_GATE:-0}"
  echo 'semantic_contract=full Ref2VA tokens; same target/ref tensors; same loss; full DiT; same optimizer/update order'
  printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
} | tee "${OUTPUT_PATH}/run_contract.txt"

cd "${CODE_ROOT}"
export H3_EXPERIMENT_SEED="${SEED}"
export H3_CORRECTNESS_CAPTURE="${CORRECTNESS_GATE:-0}"
export PYTHONPATH="${ROOT}/scripts:${CODE_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' -o "${OUTPUT_PATH}/process.time" \
    "${CMD[@]}" 2>&1 | tee "${OUTPUT_PATH}/train.log"
else
  "${CMD[@]}" 2>&1 | tee "${OUTPUT_PATH}/train.log"
fi
