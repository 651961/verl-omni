#!/usr/bin/env bash
# MiniMax H3 Ref2VA (multi-reference) DiffusionNFT LoRA recipe.
# References may mix images, videos and standalone audio (up to twelve files).
set -euo pipefail

export WANDB_MODE=disabled
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 checkpoint directory}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to the training parquet file}"
: "${VAL_DATA_PATH:?Set VAL_DATA_PATH to the validation parquet file}"

if [[ ! -d "$MODEL_PATH/Ref2VA" || ! -d "$MODEL_PATH/transformer_ref" ]]; then
    echo "MODEL_PATH must point to a MiniMax-H3 repo root containing Ref2VA/ (fused rollout) and transformer_ref/ (diffusers actor) (got: '$MODEL_PATH')" >&2
    exit 1
fi

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/models/Qwen3.8-27B}
REWARD_TP=${REWARD_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
INFER_STEPS=${INFER_STEPS:-10}
MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-12288}
REF_IMAGE_SHORT_EDGE=${REF_IMAGE_SHORT_EDGE:-2048}
VAL_REF_IMAGE_SHORT_EDGE=${VAL_REF_IMAGE_SHORT_EDGE:-$REF_IMAGE_SHORT_EDGE}
export REF_IMAGE_SHORT_EDGE
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-_flash_3_varlen_hub}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-FLASH_ATTN_3_HUB}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}

if [[ ! -d "$REWARD_MODEL_PATH" ]]; then
    echo "REWARD_MODEL_PATH must point to the VLM reward checkpoint (got: '$REWARD_MODEL_PATH')" >&2
    exit 1
fi
if (( REWARD_TP <= 0 || N_GPUS % REWARD_TP != 0 )); then
    echo "REWARD_TP must be positive and divide N_GPUS (got: REWARD_TP=$REWARD_TP, N_GPUS=$N_GPUS)" >&2
    exit 1
fi

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
output_dir=${OUTPUT_DIR:-$(dirname "$script_path")/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
mkdir -p "$checkpoint_dir"

lora_warmstart_arg=()
if [[ -n "${LORA_WARMSTART_PATH:-}" ]]; then
    lora_warmstart_arg=(actor_rollout_ref.model.lora_adapter_path=$LORA_WARMSTART_PATH)
fi

python3 -m verl_omni.trainer.main_diffusion \
  algorithm.trainer_type=direct_preference \
  algorithm.sample_source=online \
  algorithm.adv_mode=continuous \
  algorithm.timestep_fraction=1.0 \
  algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
  algorithm.old_policy_update_interval=2 \
  data.train_files="$TRAIN_DATA_PATH" \
  data.val_files="$VAL_DATA_PATH" \
  data.train_batch_size=32 \
  data.val_max_samples=128 \
  data.max_prompt_length=8192 \
  data.truncation=error \
  data.seed=42 \
  actor_rollout_ref.model.path="$MODEL_PATH/Ref2VA" \
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/Ref2VA/tokenizer" \
  actor_rollout_ref.model.config_path="$MODEL_PATH/transformer_ref" \
  +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
  actor_rollout_ref.model.external_lib=verl_omni.pipelines.minimax_h3_diffusion_nft \
  actor_rollout_ref.model.algorithm=diffusion_nft \
  actor_rollout_ref.model.model_type=diffusion_nft_model \
  actor_rollout_ref.model.attn_backend="$ACTOR_ATTN_BACKEND" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=128 \
  actor_rollout_ref.model.lora_alpha=256 \
  "${lora_warmstart_arg[@]}" \
  actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
  actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]' \
  actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=3e-4 \
  actor_rollout_ref.actor.optim.weight_decay=1e-4 \
  actor_rollout_ref.actor.optim.betas="[0.9,0.999]" \
  actor_rollout_ref.actor.optim.override_optimizer_config="{eps: 1e-8}" \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
  actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
  actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
  actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
  actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.rollout_attn_backend="$ROLLOUT_ATTN_BACKEND" \
  actor_rollout_ref.rollout.rollout_adapter=old \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.text_encoder_tp_size="$TEXT_ENCODER_TP" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.agent.num_workers=$((N_GPUS / ROLLOUT_TP)) \
  actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload=True \
  actor_rollout_ref.rollout.max_prompt_embed_length="$MAX_PROMPT_EMBEDS" \
  actor_rollout_ref.rollout.pipeline.task=ref2va \
  actor_rollout_ref.rollout.pipeline.num_inference_steps="$INFER_STEPS" \
  actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
  actor_rollout_ref.rollout.pipeline.max_sequence_length="$MAX_PROMPT_EMBEDS" \
  actor_rollout_ref.rollout.pipeline.reference_image_short_edge="$REF_IMAGE_SHORT_EDGE" \
  actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
  +actor_rollout_ref.rollout.pipeline.output_type=pt \
  actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=33 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.reference_image_short_edge="$VAL_REF_IMAGE_SHORT_EDGE" \
  +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  data.val_batch_size=4 \
  reward.reward_model.enable=True \
  reward.reward_model.model_path="$REWARD_MODEL_PATH" \
  reward.reward_model.rollout.name=vllm \
  reward.reward_model.enable_resource_pool=False \
  reward.reward_model.rollout.tensor_model_parallel_size="$REWARD_TP" \
  reward.reward_model.rollout.max_num_seqs=1 \
  reward.reward_model.rollout.free_cache_engine=True \
  reward.num_workers=$((N_GPUS / REWARD_TP)) \
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.vlm_reward \
  reward.custom_reward_function.name=compute_score_vlm \
  reward.reward_manager.name=VisualRewardManager \
  reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=diffusion_nft \
  trainer.experiment_name=minimax_h3_ref2va_lora \
  trainer.default_local_dir="$checkpoint_dir" \
  trainer.validation_data_dir="$output_dir/validation_data" \
  trainer.rollout_data_dir="$output_dir/rollout_data" \
  trainer.rollout_data_save_freq=10 \
  trainer.rollout_data_max_samples=8 \
  trainer.log_val_generations=8 \
  trainer.video_fps=24 \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  "$@"
