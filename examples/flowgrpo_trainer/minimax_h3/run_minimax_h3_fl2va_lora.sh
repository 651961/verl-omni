#!/usr/bin/env bash
# MiniMax H3 FL2VA (first-frame conditioned) LoRA FlowGRPO with a Qwen3.8-27B visual reward.
set -x

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export LOCAL_KERNELS=kernels-community/flash-attn3=/datasets/codes_zsqiao/verl-omni/.kernels/flash-attn3:kernels-community/flash-attn2=/datasets/codes_zsqiao/verl-omni/.kernels/flash-attn2

MODEL_PATH=${MODEL_PATH:-/models/MiniMax-H3/FL2VA}
DATA_DIR=${DATA_DIR:-/datasets/codes_zsqiao/fl2va_rl_dataset/reject/verl_omni_parquet}
ACTOR_CONFIG_PATH=${ACTOR_CONFIG_PATH:-$(dirname "$MODEL_PATH")/transformer}
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-${NUM_GPUS:-8}}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_SP=${ROLLOUT_SP:-4}
ROLLOUT_WORLD_SIZE=$((ROLLOUT_TP * ROLLOUT_SP))
ROLLOUT_INIT_TIMEOUT=${ROLLOUT_INIT_TIMEOUT:-1800}
ROLLOUT_STAGE_INIT_TIMEOUT=${ROLLOUT_STAGE_INIT_TIMEOUT:-$ROLLOUT_INIT_TIMEOUT}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_WORLD_SIZE}
VAE_PP=${VAE_PP:-$ROLLOUT_WORLD_SIZE}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-flash_varlen_hub}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-FLASH_ATTN_HUB}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/models/Qwen3.8-27B}
REWARD_TP=${REWARD_TP:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ROLLOUT_N=${ROLLOUT_N:-8}
INFER_STEPS=${INFER_STEPS:-10}

if (( NNODES <= 0 || GPUS_PER_NODE <= 0 || ROLLOUT_TP <= 0 || ROLLOUT_SP <= 0 || GPUS_PER_NODE % ROLLOUT_WORLD_SIZE != 0 )); then
    echo "NNODES/GPUS_PER_NODE and rollout TP/SP must be positive; ROLLOUT_TP*ROLLOUT_SP must divide GPUS_PER_NODE (got: NNODES=$NNODES, GPUS_PER_NODE=$GPUS_PER_NODE, TP=$ROLLOUT_TP, SP=$ROLLOUT_SP)" >&2
    exit 1
fi
if (( TEXT_ENCODER_TP != 1 && TEXT_ENCODER_TP != ROLLOUT_WORLD_SIZE )) ||
    [[ ! "$TEXT_ENCODER_TP" =~ ^(1|2|4|8)$ ]]; then
    echo "TEXT_ENCODER_TP must be 1 or ROLLOUT_WORLD_SIZE and one of 1, 2, 4, 8 (got: $TEXT_ENCODER_TP)" >&2
    exit 1
fi
if (( VAE_PP != 1 && VAE_PP != ROLLOUT_WORLD_SIZE )); then
    echo "VAE_PP must be 1 or ROLLOUT_WORLD_SIZE (got: VAE_PP=$VAE_PP, ROLLOUT_WORLD_SIZE=$ROLLOUT_WORLD_SIZE)" >&2
    exit 1
fi

if [[ ! -d "$REWARD_MODEL_PATH" ]]; then
    echo "REWARD_MODEL_PATH must point to the VLM reward checkpoint (got: '$REWARD_MODEL_PATH')" >&2
    exit 1
fi
if (( REWARD_TP <= 0 || GPUS_PER_NODE % REWARD_TP != 0 )); then
    echo "REWARD_TP must be positive and divide GPUS_PER_NODE (got: REWARD_TP=$REWARD_TP, GPUS_PER_NODE=$GPUS_PER_NODE)" >&2
    exit 1
fi

train_path=$DATA_DIR/train.parquet
test_path=$DATA_DIR/test.parquet

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=${OUTPUT_DIR:-$repo_root/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

echo "Nodes=$NNODES, GPUs per node=$GPUS_PER_NODE, total GPUs=$TOTAL_GPUS"
echo "Rollout workers=$((TOTAL_GPUS / ROLLOUT_WORLD_SIZE)), GPUs per replica=$ROLLOUT_WORLD_SIZE, DiT TP=$ROLLOUT_TP, DiT Ulysses SP=$ROLLOUT_SP, text encoder TP=$TEXT_ENCODER_TP, VAE tile parallel=$VAE_PP"
echo "Rollout init timeout=${ROLLOUT_INIT_TIMEOUT}s, stage init timeout=${ROLLOUT_STAGE_INIT_TIMEOUT}s"

h3_lora_targets="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

# Keep the judge's model generation defaults, including its thinking budget.
# An empty override replaces VeRL's sampling overrides and 2048-token cap.
# VeRL uses tensor_model_parallel_size to reserve each replica's GPUs;
# engine_kwargs separately sets the DiT's actual TP and Ulysses degrees.
python3 -m verl_omni.trainer.main_diffusion \
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_KERNELS='$LOCAL_KERNELS'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='${VLLM_LOGGING_LEVEL:-INFO}'" \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=4 \
    data.val_max_samples=128 \
    data.max_prompt_length=6400 \
    data.truncation=error \
    data.seed=42 \
    data.val_batch_size=4 \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.config_path=$ACTOR_CONFIG_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.transformer_subfolder=transformer \
    actor_rollout_ref.model.attn_backend=$ACTOR_ATTN_BACKEND \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=256 \
    actor_rollout_ref.model.target_modules="$h3_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$GPUS_PER_NODE \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_WORLD_SIZE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.tensor_parallel_size=$ROLLOUT_TP \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.ulysses_degree=$ROLLOUT_SP \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.ring_degree=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.num_gpus=$ROLLOUT_WORLD_SIZE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.text_encoder_tp_size=$TEXT_ENCODER_TP \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_patch_parallel_size=$VAE_PP \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=$ROLLOUT_INIT_TIMEOUT \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=$ROLLOUT_STAGE_INIT_TIMEOUT \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((TOTAL_GPUS / ROLLOUT_WORLD_SIZE)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.max_prompt_embed_length=6400 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.task=fl2va \
    actor_rollout_ref.rollout.pipeline.frame_indices='[0]' \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$INFER_STEPS \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=6400 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=33 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,8]' \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_contiguous=True \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="$REWARD_MODEL_PATH" \
    reward.reward_model.rollout.name=vllm \
    "+reward.reward_model.rollout.engine_kwargs.vllm.override_generation_config='{}'" \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.rollout.tensor_model_parallel_size="$REWARD_TP" \
    reward.reward_model.rollout.max_num_seqs=16 \
    reward.reward_model.rollout.free_cache_engine=True \
    reward.num_workers=$((TOTAL_GPUS / REWARD_TP)) \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.vlm_reward \
    reward.custom_reward_function.name=compute_score_vlm \
    reward.reward_manager.name=VisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=minimax_h3_fl2va_lora_${NNODES}x${GPUS_PER_NODE} \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    trainer.rollout_data_save_freq=10 \
    trainer.log_val_generations=8 \
    trainer.video_fps=24 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
