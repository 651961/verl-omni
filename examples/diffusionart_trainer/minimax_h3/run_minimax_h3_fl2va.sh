#!/usr/bin/env bash
# MiniMax H3 FL2VA full-parameter DiffusionART with separate actor/rollout pools.
set -euo pipefail

export HF_HUB_OFFLINE=1
export LOCAL_KERNELS="kernels-community/flash-attn2=/models/kernels-community/flash-attn2"

# Train and validation rows must set extra_info.height, width and num_frames.
# Geometry may vary between prompts; generation always uses 24 fps.

train_path=${TRAIN_FILE:?Set TRAIN_FILE to the training parquet}
test_path=${VAL_FILE:?Set VAL_FILE to the validation parquet}
model_name=${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 repository root}
reward_function_path=${REWARD_FUNCTION_PATH:?Set REWARD_FUNCTION_PATH to your video/audio reward implementation}
reward_function_name=${REWARD_FUNCTION_NAME:?Set REWARD_FUNCTION_NAME to its callable name}
actor_config_path=${ACTOR_CONFIG_PATH:-$model_name/transformer}

ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ACTOR_NNODES=${ACTOR_NNODES:-1}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-8}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}

PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
GROUP_SIZE=${GROUP_SIZE:-8}
ART_LR=${ART_LR:-1e-5}
TRAIN_STEPS=${TRAIN_STEPS:-1000}

ART_STEP_INDICES=${ART_STEP_INDICES:-null}

output_dir=${OUTPUT_DIR:-outputs/minimax_h3_diffusionart_fl2va}

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size="$PROMPTS_PER_BATCH" \
    data.val_batch_size="$PROMPTS_PER_BATCH" \
    data.max_prompt_length=8192 \
    data.filter_overlong_prompts=false \
    data.return_multi_modal_inputs=false \
    actor_rollout_ref.model.path="$model_name/FL2VA" \
    actor_rollout_ref.model.config_path="$actor_config_path" \
    actor_rollout_ref.model.tokenizer_path="$model_name/FL2VA/tokenizer" \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.algorithm=diffusion_art \
    actor_rollout_ref.model.model_type=diffusion_art_model \
    actor_rollout_ref.model.attn_backend=flash_varlen_hub \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_regional_compile=false \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.policy_state_adapters='[default]' \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PROMPTS_PER_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.art_restore_zero_signal=true \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    actor_rollout_ref.actor.optim.lr="$ART_LR" \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.enable_timestep_staging=true \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.text_encoder_tp_size="$TEXT_ENCODER_TP" \
    actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN_HUB \
    actor_rollout_ref.rollout.step_execution=false \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_art_agent \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.vae_use_tiling=true \
    actor_rollout_ref.rollout.pipeline.task="fl2va" \
    actor_rollout_ref.rollout.pipeline.frame_indices='[0]' \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=8192 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.art_step_indices="$ART_STEP_INDICES" \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=10 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    reward.reward_model.enable=false \
    reward.custom_reward_function.path="$reward_function_path" \
    reward.custom_reward_function.name="$reward_function_name" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.sync_compatible=false \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=false \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    trainer.v1.sampler.max_off_policy_threshold=2 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    trainer.n_gpus_per_node="$ACTOR_GPUS_PER_NODE" \
    trainer.nnodes="$ACTOR_NNODES" \
    trainer.logger='[console,tensorboard]' \
    trainer.project_name=diffusion_art \
    trainer.experiment_name="minimax_h3_fl2va" \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.video_fps=24 \
    trainer.val_before_train=true \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir=null \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps="$TRAIN_STEPS" \
    "$@"
