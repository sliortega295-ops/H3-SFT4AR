#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:-/share/project/zhj/iclr26/DiffSynth-Studio/published_dynamic_segshots_v2_fps24_50_h3_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/MiniMax-H3-Ref2VA-full-preprocessed-csv}"
MODEL_SPEC="${MODEL_SPEC:-MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors}"
DIT_MODEL_PATH="${DIT_MODEL_PATH:-/share/project/zhj/project/MiniMax-H3/Ref2VA/transformer/model*.safetensors}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

if compgen -G "${DIT_MODEL_PATH}" > /dev/null; then
  MODEL_ARGS=(--dit_model_path "${DIT_MODEL_PATH}")
else
  MODEL_ARGS=(--model_id_with_origin_paths "${MODEL_SPEC}")
fi

accelerate launch \
  --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train_preprocessed_csv.py \
  --dataset_metadata_path "${CSV_PATH}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers 0 \
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
  --find_unused_parameters \
  --save_steps 100 \
  --task "sft"
