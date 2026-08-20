import argparse
import glob
import os
from pathlib import Path

import accelerate
import torch

try:
    from .preprocessed_csv_dataset import (
        DEFAULT_AUDIO_CHUNK_LATENTS,
        DEFAULT_CHUNK_NUM_FRAMES,
        DEFAULT_VIDEO_CHUNK_LATENTS,
        MiniMaxH3PreprocessedCSVDataset,
    )
except ImportError:
    from preprocessed_csv_dataset import (
        DEFAULT_AUDIO_CHUNK_LATENTS,
        DEFAULT_CHUNK_NUM_FRAMES,
        DEFAULT_VIDEO_CHUNK_LATENTS,
        MiniMaxH3PreprocessedCSVDataset,
    )

from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTMiniMaxH3AudioVideoLoss,
    ModelLogger,
    add_gradient_config,
    add_logger_config,
    add_lora_config,
    add_model_config,
    add_offload_training_config,
    add_output_config,
    add_training_config,
    launch_training_task,
)
from diffsynth.models.minimax_h3_dit import pack_audio, patchify_video
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline,
    MiniMaxH3Unit_PackedSequenceBuilder,
    ModelConfig,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MiniMaxH3PreprocessedTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        dit_model_path=None,
        trainable_models="dit",
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        min_timestep_boundary=0.0,
        max_timestep_boundary=1.0,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        if task != "sft":
            raise ValueError("The preprocessed CSV trainer supports only --task sft.")

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        if dit_model_path is not None:
            expanded_path = os.path.expanduser(dit_model_path)
            matched_paths = sorted(glob.glob(expanded_path))
            if not matched_paths and Path(expanded_path).is_file():
                matched_paths = [str(Path(expanded_path).resolve())]
            if not matched_paths:
                raise ValueError(
                    f"--dit_model_path {dit_model_path!r} does not match any files."
                )
            model_configs.insert(
                0,
                ModelConfig(
                    path=matched_paths[0] if len(matched_paths) == 1 else matched_paths
                ),
            )
        self.pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            processor_config=None,
        )
        if self.pipe.dit is None:
            raise ValueError("No MiniMax-H3 DiT was loaded. Check the transformer model path.")

        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=lora_checkpoint,
            preset_lora_path=preset_lora_path,
            preset_lora_model=preset_lora_model,
            task=task,
        )
        self.pipe.scheduler_audio.set_timesteps(1000, training=True)

        # A value of 1.0 in model_fn is the clean endpoint, i.e. diffusion timestep 0.
        self.pipe.imgvid_cond_noise_aug = 1.0
        self.pipe.audio_cond_noise_aug = 1.0
        self.packed_sequence_builder = MiniMaxH3Unit_PackedSequenceBuilder()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.min_timestep_boundary = float(min_timestep_boundary)
        self.max_timestep_boundary = float(max_timestep_boundary)
        if not 0 <= self.min_timestep_boundary < self.max_timestep_boundary <= 1:
            raise ValueError("Timestep boundaries must satisfy 0 <= min < max <= 1.")

    def forward(self, data):
        data = self.transfer_data_to_device(data, self.pipe.device, self.pipe.torch_dtype)
        video_latents = data["video_latents"]
        audio_latents = data["audio_latents"]
        prompt_embeds = data["prompt_embeds"]
        text_token_tags = data["text_token_tags"]

        ref_blocks = None
        ref_visual_anchor = None
        ref_audio_anchor = None
        if data["has_reference"]:
            ref_video_latents = data["ref_video_latents"]
            ref_audio_latents = data["ref_audio_latents"]
            ref_blocks = [
                {
                    "kind": "video_audio",
                    "latent_t": int(ref_video_latents.shape[2]),
                    "latent_h": int(ref_video_latents.shape[3]),
                    "latent_w": int(ref_video_latents.shape[4]),
                    "ref_audio_t": int(ref_audio_latents.shape[-1]),
                }
            ]
            # These are the clean cached latents; no scheduler noise is applied to refs.
            ref_visual_anchor = patchify_video(ref_video_latents)
            ref_audio_anchor = pack_audio(ref_audio_latents)

        packed = self.packed_sequence_builder.process(
            self.pipe,
            prompt_embeds=prompt_embeds,
            video_latents=video_latents,
            audio_latents=audio_latents,
            text_token_tags=text_token_tags,
            ref_blocks=ref_blocks,
        )["packed"]

        return FlowMatchSFTMiniMaxH3AudioVideoLoss(
            self.pipe,
            input_latents=video_latents,
            audio_input_latents=audio_latents,
            packed=packed,
            prompt_embeds=prompt_embeds,
            ref_visual_anchor=ref_visual_anchor,
            ref_audio_anchor=ref_audio_anchor,
            imgvid_cond_noise_aug=1.0,
            audio_cond_noise_aug=1.0,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            min_timestep_boundary=self.min_timestep_boundary,
            max_timestep_boundary=self.max_timestep_boundary,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train MiniMax-H3 directly from preprocessed video/audio/text caches."
    )
    parser.add_argument("--dataset_metadata_path", required=True, help="Preprocessed CSV manifest.")
    parser.add_argument(
        "--dataset_base_path",
        default=None,
        help="Base directory for relative cache paths. Defaults to the CSV directory.",
    )
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--chunk_num_frames", type=int, default=DEFAULT_CHUNK_NUM_FRAMES)
    parser.add_argument("--video_chunk_latents", type=int, default=DEFAULT_VIDEO_CHUNK_LATENTS)
    parser.add_argument("--audio_chunk_latents", type=int, default=DEFAULT_AUDIO_CHUNK_LATENTS)
    parser.add_argument("--video_latents_column", default="h3_video_latents")
    parser.add_argument("--audio_latents_column", default="h3_audio_latents")
    parser.add_argument("--text_emb_column", default="h3_text_emb")
    parser.add_argument("--status_column", default="h3_status")
    parser.add_argument("--initialize_model_on_cpu", action="store_true")
    parser.add_argument(
        "--dit_model_path",
        default=None,
        help="Local MiniMax-H3 transformer file or glob. All matched shards load together.",
    )
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)

    for config_adder in (
        add_model_config,
        add_training_config,
        add_output_config,
        add_lora_config,
        add_gradient_config,
        add_offload_training_config,
        add_logger_config,
    ):
        parser = config_adder(parser)
    return parser


def main():
    args = build_parser().parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = MiniMaxH3PreprocessedCSVDataset(
        metadata_path=args.dataset_metadata_path,
        dataset_base_path=args.dataset_base_path,
        repeat=args.dataset_repeat,
        chunk_num_frames=args.chunk_num_frames,
        video_chunk_latents=args.video_chunk_latents,
        audio_chunk_latents=args.audio_chunk_latents,
        video_latents_column=args.video_latents_column,
        audio_latents_column=args.audio_latents_column,
        text_emb_column=args.text_emb_column,
        status_column=args.status_column,
    )
    model = MiniMaxH3PreprocessedTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        dit_model_path=args.dit_model_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        min_timestep_boundary=args.min_timestep_boundary,
        max_timestep_boundary=args.max_timestep_boundary,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        task=args.task,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launch_training_task(
        accelerator,
        dataset,
        model,
        model_logger,
        args=args,
    )


if __name__ == "__main__":
    main()
