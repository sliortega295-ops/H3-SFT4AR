#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:?Set CSV_PATH to the preprocessed CSV manifest.}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-Ref2VA-full-preprocessed-csv-optimized}"
MODEL_SPEC="${MODEL_SPEC:-MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors}"
DIT_MODEL_PATH="${DIT_MODEL_PATH:-}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ACCELERATE_CONFIG_FILE:-examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml}}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
TEXT_CACHE_SIZE="${TEXT_CACHE_SIZE:-64}"
PACKED_LAYOUT_CACHE_SIZE="${PACKED_LAYOUT_CACHE_SIZE:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
BENCHMARK_WARMUP_STEPS="${BENCHMARK_WARMUP_STEPS:-0}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
DATALOADER_NON_BLOCKING="${DATALOADER_NON_BLOCKING:-1}"
CACHE_MMAP="${CACHE_MMAP:-1}"
ENABLE_NVTX="${ENABLE_NVTX:-0}"
PROFILE_CUDA_CAPTURE="${PROFILE_CUDA_CAPTURE:-0}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-0}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-1}"

MODEL_ARGS=()
if [[ -n "${DIT_MODEL_PATH}" ]] && compgen -G "${DIT_MODEL_PATH}" > /dev/null; then
  MODEL_ARGS=(--dit_model_path "${DIT_MODEL_PATH}")
else
  MODEL_ARGS=(--model_id_with_origin_paths "${MODEL_SPEC}")
fi

bool_flag() {
  local enabled="$1" positive="$2" negative="$3"
  if [[ "${enabled}" == "1" ]]; then printf '%s\n' "${positive}"; else printf '%s\n' "${negative}"; fi
}

DATA_ARGS=(
  --dataset_metadata_path "${CSV_PATH}"
  --dataset_repeat "${DATASET_REPEAT}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --text_cache_size "${TEXT_CACHE_SIZE}"
  --packed_layout_cache_size "${PACKED_LAYOUT_CACHE_SIZE}"
  "$(bool_flag "${DATALOADER_PIN_MEMORY}" --dataloader_pin_memory --no-dataloader_pin_memory)"
  "$(bool_flag "${DATALOADER_NON_BLOCKING}" --dataloader_non_blocking --no-dataloader_non_blocking)"
  "$(bool_flag "${CACHE_MMAP}" --cache_mmap --no-cache_mmap)"
)
if (( DATASET_NUM_WORKERS > 0 )) && [[ "${DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
  DATA_ARGS+=(--dataloader_persistent_workers)
else
  DATA_ARGS+=(--no-dataloader_persistent_workers)
fi

RUN_ARGS=()
[[ -z "${MAX_TRAIN_STEPS}" ]] || RUN_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
if (( BENCHMARK_STEPS > 0 )); then
  RUN_ARGS+=(--benchmark_warmup_steps "${BENCHMARK_WARMUP_STEPS}" --benchmark_steps "${BENCHMARK_STEPS}")
fi
[[ "${ENABLE_NVTX}" != "1" ]] || RUN_ARGS+=(--enable_nvtx)
[[ "${PROFILE_CUDA_CAPTURE}" != "1" ]] || RUN_ARGS+=(--profile_cuda_capture)
[[ "${SKIP_FINAL_SAVE}" != "1" ]] || RUN_ARGS+=(--skip_final_save)
[[ "${FIND_UNUSED_PARAMETERS}" != "1" ]] || RUN_ARGS+=(--find_unused_parameters)

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  examples/minimax_h3/model_training/train_preprocessed_csv.py \
  "${DATA_ARGS[@]}" \
  --chunk_num_frames 56 \
  --video_chunk_latents 15 \
  --audio_chunk_latents 93 \
  "${MODEL_ARGS[@]}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --save_steps "${SAVE_STEPS}" \
  --task "sft" \
  "${RUN_ARGS[@]}"
