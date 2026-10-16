#!/usr/bin/env bash
# Qwen-Image 2.1 DiffusionART, V1 separate_async trainer and vllm_omni rollout.
set -x

train_path=${TRAIN_FILE:?Set TRAIN_FILE to the training parquet}
val_path=${VAL_FILE:?Set VAL_FILE to the validation parquet}
model_name=${ART_MODEL_PATH:?Set ART_MODEL_PATH to a Qwen-Image-2.1 checkpoint}
reward_function_path=${REWARD_FUNCTION_PATH:?Set REWARD_FUNCTION_PATH to your reward implementation}
reward_function_name=${REWARD_FUNCTION_NAME:?Set REWARD_FUNCTION_NAME to its callable name}

# Separate actor and rollout pools. Each actor node uses 8 GPUs (HSDP, SP=1).
# ACTOR_GPUS and ROLLOUT_GPUS are total GPU counts across their respective nodes.
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ACTOR_NNODES=${ACTOR_NNODES:-$(( (${ACTOR_GPUS:-8} + ACTOR_GPUS_PER_NODE - 1) / ACTOR_GPUS_PER_NODE ))}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-$((ROLLOUT_GPUS < 8 ? ROLLOUT_GPUS : 8))}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( (ROLLOUT_GPUS + ROLLOUT_GPUS_PER_NODE - 1) / ROLLOUT_GPUS_PER_NODE ))}

PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-8}
GROUP_SIZE=${GROUP_SIZE:-8}
SYNC_STEPS=${SYNC_STEPS:-1}
PPO_MINI_BATCH_SIZE=$((PROMPTS_PER_BATCH / SYNC_STEPS))
ART_LR=${ART_LR:-1e-5}
TRAIN_STEPS=${TRAIN_STEPS:-1000}

OUTPUT_RESOLUTION=${OUTPUT_RESOLUTION:-2048}
ART_STEP_INDICES=${ART_STEP_INDICES:-null}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-28}

CKPT_BACKEND=${CKPT_BACKEND:-omni_delta_sharded}

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$train_path" \
    data.val_files="$val_path" \
    data.train_batch_size="$PROMPTS_PER_BATCH" \
    data.val_batch_size="$PROMPTS_PER_BATCH" \
    data.max_prompt_length=8192 \
    data.filter_overlong_prompts=false \
    data.return_multi_modal_inputs=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.qwen_image21 \
    data.custom_cls.name=QwenImage21Dataset \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.model.model_type=diffusion_art_model \
    actor_rollout_ref.model.algorithm=diffusion_art \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.gradient_checkpointing_save_ops='[aten._scaled_dot_product_flash_attention.default,aten.mm.default]' \
    actor_rollout_ref.model.use_regional_compile=false \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.policy_state_adapters='[default]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.art_restore_zero_signal=true \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    actor_rollout_ref.actor.optim.lr="$ART_LR" \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.step_execution=true \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.agent.default_agent_loop=qwen_image21_agent \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.vae_use_tiling=true \
    actor_rollout_ref.rollout.pipeline.height=0 \
    actor_rollout_ref.rollout.pipeline.width=0 \
    actor_rollout_ref.rollout.pipeline.output_resolution="$OUTPUT_RESOLUTION" \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=8192 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps="$ROLLOUT_STEPS" \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.art_step_indices="$ART_STEP_INDICES" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.max_sequence_length=8192 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CKPT_BACKEND" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.omni_delta_sharded.encoding=indices \
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
    trainer.v1.separate_async.parameter_sync_step="$SYNC_STEPS" \
    trainer.v1.sampler.max_off_policy_threshold=2 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    trainer.n_gpus_per_node="$ACTOR_GPUS_PER_NODE" \
    trainer.nnodes="$ACTOR_NNODES" \
    trainer.logger='[console,tensorboard]' \
    trainer.project_name=diffusion_art \
    trainer.experiment_name=qwen_image21_async \
    trainer.val_before_train=true \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir=null \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps="$TRAIN_STEPS" \
    "$@"
