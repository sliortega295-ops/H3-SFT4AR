#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?Usage: $0 baseline|optimized-core|optimized-cache|optimized-data|optimized-zero}"
: "${CSV_PATH:?Set CSV_PATH to the preprocessed CSV manifest.}"

DIT_MODEL_PATH="${DIT_MODEL_PATH:-}"
MODEL_SPEC="${MODEL_SPEC:-MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEED="${SEED:-20260820}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-30}"
MAX_TRAIN_STEPS="$((WARMUP_STEPS + MEASURE_STEPS))"
DATA_WORKERS="${DATA_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TEXT_CACHE_SIZE="${TEXT_CACHE_SIZE:-64}"
PACKED_LAYOUT_CACHE_SIZE="${PACKED_LAYOUT_CACHE_SIZE:-64}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/benchmark-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${MODE}}"
OUTPUT_PATH="${RESULT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS=()
if [[ -n "${DIT_MODEL_PATH}" ]] && compgen -G "${DIT_MODEL_PATH}" > /dev/null; then
  MODEL_ARGS=(--dit_model_path "${DIT_MODEL_PATH}")
else
  MODEL_ARGS=(--model_id_with_origin_paths "${MODEL_SPEC}")
fi

OPT_DATA_ARGS=()
case "${MODE}" in
  baseline)
    CODE_ROOT="$(${ROOT}/scripts/prepare_baseline_benchmark.sh "${ROOT}/.work/baseline-measured")"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    ;;
  optimized-core)
    CODE_ROOT="${ROOT}/optimized/DiffSynth-Studio"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    OPT_DATA_ARGS=(
      --dataset_num_workers 0
      --no-dataloader_pin_memory
      --no-dataloader_persistent_workers
      --no-dataloader_non_blocking
      --no-cache_mmap
      --text_cache_size 0
      --packed_layout_cache_size 0
    )
    ;;
  optimized-cache)
    CODE_ROOT="${ROOT}/optimized/DiffSynth-Studio"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    OPT_DATA_ARGS=(
      --dataset_num_workers 0
      --dataloader_pin_memory
      --no-dataloader_persistent_workers
      --dataloader_non_blocking
      --cache_mmap
      --text_cache_size "${TEXT_CACHE_SIZE}"
      --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
    )
    ;;
  optimized-data)
    CODE_ROOT="${ROOT}/optimized/DiffSynth-Studio"
    ACCELERATE_CONFIG="${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml"
    OPT_DATA_ARGS=(
      --dataset_num_workers "${DATA_WORKERS}"
      --dataloader_pin_memory
      --dataloader_non_blocking
      --cache_mmap
      --text_cache_size "${TEXT_CACHE_SIZE}"
      --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
    )
    if (( DATA_WORKERS > 0 )); then
      OPT_DATA_ARGS+=(
        --dataloader_persistent_workers
        --dataloader_prefetch_factor "${PREFETCH_FACTOR}"
      )
    else
      OPT_DATA_ARGS+=(--no-dataloader_persistent_workers)
    fi
    ;;
  optimized-zero)
    CODE_ROOT="${ROOT}/optimized/DiffSynth-Studio"
    ACCELERATE_CONFIG="${ZERO3_ACCELERATE_CONFIG:-${CODE_ROOT}/examples/minimax_h3/model_training/full/accelerate_config_zero3_optimized.yaml}"
    OPT_DATA_ARGS=(
      --dataset_num_workers "${DATA_WORKERS}"
      --dataloader_pin_memory
      --dataloader_non_blocking
      --cache_mmap
      --text_cache_size "${TEXT_CACHE_SIZE}"
      --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
    )
    if (( DATA_WORKERS > 0 )); then
      OPT_DATA_ARGS+=(
        --dataloader_persistent_workers
        --dataloader_prefetch_factor "${PREFETCH_FACTOR}"
      )
    else
      OPT_DATA_ARGS+=(--no-dataloader_persistent_workers)
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

PROFILE_ARGS=()
if [[ "${ENABLE_NVTX:-0}" == "1" ]]; then
  PROFILE_ARGS+=(--enable_nvtx)
fi
if [[ "${PROFILE_CUDA_CAPTURE:-0}" == "1" ]]; then
  PROFILE_ARGS+=(--profile_cuda_capture)
fi

CMD=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  examples/minimax_h3/model_training/train_preprocessed_csv.py
  --dataset_metadata_path "${CSV_PATH}"
  --dataset_repeat "${DATASET_REPEAT}"
  --chunk_num_frames 56
  --video_chunk_latents 15
  --audio_chunk_latents 93
  "${MODEL_ARGS[@]}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs 1
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  --trainable_models dit
  --use_gradient_checkpointing
  --find_unused_parameters
  --save_steps 1000000000
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --benchmark_warmup_steps "${WARMUP_STEPS}"
  --benchmark_steps "${MEASURE_STEPS}"
  --skip_final_save
  --task sft
  "${OPT_DATA_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
)

{
  printf 'mode=%s\n' "${MODE}"
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'code_root=%s\n' "${CODE_ROOT}"
  printf 'accelerate_config=%s\n' "${ACCELERATE_CONFIG}"
  printf 'output=%s\n' "${OUTPUT_PATH}"
  printf 'seed=%s\n' "${SEED}"
  printf 'warmup_steps=%s\n' "${WARMUP_STEPS}"
  printf 'measure_steps=%s\n' "${MEASURE_STEPS}"
  printf 'semantic_contract=full Ref2VA tokens, full DiT, same loss and update order\n'
  printf 'sample_sequence_contract=workers=0 preserves the baseline RNG call order; workers>0 preserves the sampling distribution but not a per-step sample sequence\n'
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} | tee "${OUTPUT_PATH}/run_contract.txt"

cd "${CODE_ROOT}"
export H3_EXPERIMENT_SEED="${SEED}"
export PYTHONPATH="${ROOT}/experiments:${CODE_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' -o "${OUTPUT_PATH}/process.time" \
    "${CMD[@]}" 2>&1 | tee "${OUTPUT_PATH}/train.log"
else
  "${CMD[@]}" 2>&1 | tee "${OUTPUT_PATH}/train.log"
fi
