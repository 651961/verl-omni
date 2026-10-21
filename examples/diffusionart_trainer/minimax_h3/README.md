# MiniMax H3 DiffusionART

Last updated: 10/24/2026

Train MiniMax H3 FL2VA or Ref2VA with full-parameter DiffusionART, FSDP2 actors,
and vLLM-Omni rollout for joint video and audio generation. See the
[DiffusionART guide](../../../docs/algo/diffusionart.md) for the algorithm and
[V1 trainer guide](../../../docs/start/diffusion_v1.md) for separate actor and
rollout pools.

## Install

Follow the project [installation guide](../../../docs/start/install.md).
Use Diffusers and vLLM-Omni versions with MiniMax H3 support. Install
`quack-kernels>=0.6.5` for actor training and FlashAttention Hub kernels
(`kernels-community/flash-attn2`) compatible with your PyTorch environment.

The scripts enable offline Hugging Face loading. Prepare the model assets
before launching and place FlashAttention Hub kernels at
`/models/kernels-community/flash-attn2`, or update `LOCAL_KERNELS` in the scripts.

Defaults use 8 actor GPUs and 8 rollout GPUs, with rollout TP=4 and text encoder
TP=4. Actors use SP=1 and microbatch size 1. Use CFG-distilled checkpoints with
`true_cfg_scale=1.0`.

## Prepare the checkpoint

Set `MODEL_PATH` to the MiniMax H3 repository root containing `FL2VA/` and
`transformer/` for FL2VA, or `Ref2VA/` and `transformer_ref/` for Ref2VA.
`ACTOR_CONFIG_PATH` can override the actor transformer directory.

## Prepare the data

Use [prepare_fl2va_data.py](../../diffusionnft_trainer/minimax_h3/prepare_fl2va_data.py)
for first-frame conditioning, or
[prepare_ref2va_data.py](../../diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py)
for image, video, and audio references. See the
[MiniMax H3 data guide](../../diffusionnft_trainer/minimax_h3/README.md#data-preparation)
for input formats and converter commands.

Add `height`, `width`, and `num_frames` to every training and validation row's
`extra_info`, preserving the other converter metadata:

```json
{"height": 768, "width": 1344, "num_frames": 124}
```

Height and width must be positive multiples of 32. Generation uses 24 fps;
set `num_frames` to a supported `17n+5` frame count, such as 107, 124, or 243,
within the checkpoint's supported duration range. Missing or unaligned geometry
raises an error before generation. Different prompts can use different geometry;
all samples for one prompt use the same geometry.

The custom reward function receives a uint8 TCHW `solution_image`. Select the
named audio artifact from `extra_info["media_artifacts"]` when computing audio
rewards (see the [media contract](../../../docs/advanced/diffusion_media_artifacts.md)).
It must return finite scalar scores for both modalities:

```python
return {"score": logging_score, "video_reward": video_reward, "audio_reward": audio_reward}
```

`score` is the combined score for logging and validation. ART uses the explicit
`video_reward` and `audio_reward` independently. Scalar-only reward functions are
rejected during actor batch preparation. Both regular and TransferQueue reward
transport preserve these fields.

For the author's five-scorer recipe, route the **transformed training components**
as `video_reward = video_align + 1.5 * hpsv3 + desync` and
`audio_reward = 0.5 * audiobox + clap + desync`. Apply the scorer transformations
before these sums; raw scorer units are not interchangeable. Other scorers can
supply their own explicit modality scores.

Each modality has its own complete-prompt covariance, direction RMS,
corrective mask, and feature mean. The joint loss is
`0.5 * video_loss + 0.5 * audio_loss`, regardless of latent dimensions. The model
still performs one joint forward per selected step; Flash (`ART_STEP_INDICES='[0]'`)
uses the same split objective. See the [reward guide](../../../docs/algo/async_reward.md)
for reward configuration.

## Launch training

Run from the repository root with paths accessible to all workers:

```bash
export MODEL_PATH=/models/MiniMax-H3
export TRAIN_FILE=/data/train.parquet
export VAL_FILE=/data/val.parquet
export REWARD_FUNCTION_PATH=/path/to/video_reward.py
export REWARD_FUNCTION_NAME=compute_score

bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_fl2va.sh
```

For Ref2VA:

```bash
bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_ref2va.sh
```

Both recipes use 8 prompts per batch, 8 samples per prompt, learning rate
`1e-5`, 10 denoising steps, and 1000 trainer steps. Set `PROMPTS_PER_BATCH`,
`GROUP_SIZE`, `ART_LR`, and `TRAIN_STEPS` to change these values. Group size
must be even, and each update window must divide evenly across actor GPUs.
`ROLLOUT_TP` and `TEXT_ENCODER_TP` configure rollout parallelism.

Append Hydra overrides to the command, for example:

```bash
bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_fl2va.sh \
    trainer.total_training_steps=20 \
    trainer.logger=console
```

Weight synchronization uses `nccl`. Validation and checkpointing run every 20
trainer steps, with validation also run before training. Logging uses console
and TensorBoard. Checkpoints are written under
`outputs/minimax_h3_diffusionart_fl2va/checkpoints` or
`outputs/minimax_h3_diffusionart_ref2va/checkpoints`; set `OUTPUT_DIR` to change
the output directory.
