#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet}"
: "${REWARD_FUNCTION_PATH:?Set REWARD_FUNCTION_PATH to your reward implementation}"
: "${REWARD_FUNCTION_NAME:?Set REWARD_FUNCTION_NAME to its callable name}"

: "${ART_MODEL_PATH:?Set ART_MODEL_PATH to a Qwen-Image-2.1 checkpoint}"
# ACTOR_GPUS / ROLLOUT_GPUS denote total GPUs in their separate pools.
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
for value in "$ACTOR_GPUS_PER_NODE" "${ACTOR_GPUS:-8}" "${ACTOR_NNODES:-1}" \
    "${ROLLOUT_GPUS:-2}" "${ROLLOUT_GPUS_PER_NODE:-2}" "${ROLLOUT_NNODES:-1}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU counts and node counts must be positive integers." >&2
        exit 1
    fi
done
ACTOR_NNODES=${ACTOR_NNODES:-$(( (${ACTOR_GPUS:-8} + ACTOR_GPUS_PER_NODE - 1) / ACTOR_GPUS_PER_NODE ))}
ACTOR_GPUS=${ACTOR_GPUS:-$((ACTOR_NNODES * ACTOR_GPUS_PER_NODE))}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-$((ROLLOUT_GPUS < 8 ? ROLLOUT_GPUS : 8))}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( (ROLLOUT_GPUS + ROLLOUT_GPUS_PER_NODE - 1) / ROLLOUT_GPUS_PER_NODE ))}
if (( ACTOR_GPUS_PER_NODE != 8 || ACTOR_NNODES < 1 || ACTOR_GPUS != ACTOR_NNODES * ACTOR_GPUS_PER_NODE )); then
    echo "ART HSDP requires 8 actor GPUs per node and ACTOR_GPUS=ACTOR_NNODES*8." >&2
    exit 1
fi
if (( ROLLOUT_GPUS_PER_NODE < 1 || ROLLOUT_GPUS_PER_NODE > 8 || ROLLOUT_NNODES < 1 || ROLLOUT_GPUS != ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE )); then
    echo "Rollout GPUs must equal ROLLOUT_NNODES*ROLLOUT_GPUS_PER_NODE (1..8 per node)." >&2
    exit 1
fi
GROUP_SIZE=${GROUP_SIZE:-8}
PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
VAL_PROMPTS_PER_BATCH=${VAL_PROMPTS_PER_BATCH:-$PROMPTS_PER_BATCH}
VAL_GROUP_SIZE=${VAL_GROUP_SIZE:-1}
# Keep storage bounded to a few live batches. Units shard data; they are not replicas.
TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-$((ACTOR_NNODES + ROLLOUT_NNODES))}
TQ_STORAGE_SIZE=${TQ_STORAGE_SIZE:-$((4 * PROMPTS_PER_BATCH * (GROUP_SIZE + 1) + 2 * VAL_PROMPTS_PER_BATCH * (VAL_GROUP_SIZE + 1)))}
SYNC_STEPS=${SYNC_STEPS:-1}
CKPT_BACKEND=${CKPT_BACKEND:-omni_delta_sharded}
# Official ART needs no frozen reference model; USE_REFERENCE=true opts into
# additional actor/reference velocity MSE regularization.
if (( SYNC_STEPS < 1 || PROMPTS_PER_BATCH < 1 || GROUP_SIZE < 2 || VAL_PROMPTS_PER_BATCH < 1 || VAL_GROUP_SIZE < 1 || TQ_STORAGE_UNITS < 1 || TQ_STORAGE_SIZE < TQ_STORAGE_UNITS )); then
    echo "Batch, sync, validation and storage sizes must be positive; GROUP_SIZE must be >=2." >&2
    exit 1
fi
if (( PROMPTS_PER_BATCH % SYNC_STEPS != 0 || (PROMPTS_PER_BATCH / SYNC_STEPS * GROUP_SIZE) % ACTOR_GPUS != 0 )); then
    echo "Prompt mini-batches must be integral and their trajectory count divisible by ACTOR_GPUS." >&2
    exit 1
fi

python -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$PROMPTS_PER_BATCH" \
    data.val_batch_size="$VAL_PROMPTS_PER_BATCH" \
    data.max_prompt_length=4096 \
    data.filter_overlong_prompts=false \
    data.return_multi_modal_inputs=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.qwen_image21 \
    data.custom_cls.name=QwenImage21Dataset \
    actor_rollout_ref.model.path="$ART_MODEL_PATH" \
    actor_rollout_ref.model.model_type=diffusion_art_model \
    actor_rollout_ref.model.algorithm=diffusion_art \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.use_regional_compile=false \
    actor_rollout_ref.model.lora_rank=0 \
    'actor_rollout_ref.model.policy_state_adapters=[default]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$((PROMPTS_PER_BATCH / SYNC_STEPS))" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    actor_rollout_ref.actor.optim.lr="${ART_LR:-1e-6}" \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.enable_timestep_staging=true \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=false \
    actor_rollout_ref.actor.use_kl_loss="${USE_REFERENCE:-false}" \
    actor_rollout_ref.actor.kl_loss_coef="${REFERENCE_MSE_COEF:-0.001}" \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-false}" \
    actor_rollout_ref.rollout.step_execution="${ROLLOUT_STEP_EXECUTION:-true}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-4}" \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.agent.default_agent_loop=qwen_image21_agent \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=0 \
    actor_rollout_ref.rollout.pipeline.width=0 \
    actor_rollout_ref.rollout.pipeline.output_resolution="${OUTPUT_RESOLUTION:-1024}" \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=4096 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps="${ROLLOUT_STEPS:-28}" \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.art_step_indices="${ART_STEP_INDICES:-null}" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_GROUP_SIZE" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.output_resolution="${OUTPUT_RESOLUTION:-1024}" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps="${ROLLOUT_STEPS:-28}" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CKPT_BACKEND" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.omni_delta_sharded.encoding=indices \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    reward.reward_model.enable=false \
    reward.custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    reward.custom_reward_function.name="$REWARD_FUNCTION_NAME" \
    transfer_queue.backend.SimpleStorage.num_data_storage_units="$TQ_STORAGE_UNITS" \
    transfer_queue.backend.SimpleStorage.total_storage_size="$TQ_STORAGE_SIZE" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.sync_compatible=false \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=false \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step="$SYNC_STEPS" \
    trainer.v1.sampler.max_off_policy_threshold=2 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    trainer.n_gpus_per_node="$ACTOR_GPUS_PER_NODE" \
    trainer.nnodes="$ACTOR_NNODES" \
    trainer.logger='[console]' \
    trainer.project_name=diffusion_art \
    trainer.experiment_name=qwen_image21_async \
    trainer.val_before_train=false \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_training_steps="${TRAIN_STEPS:-1000}" "$@"
