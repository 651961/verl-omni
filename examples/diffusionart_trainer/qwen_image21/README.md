# Qwen-Image 2.1 DiffusionART

Last updated: 10/21/2026

Train Qwen-Image 2.1 or Qwen-Image 2.1 Turbo with full-parameter DiffusionART,
FSDP2 actors, and vLLM-Omni rollout. See the
[DiffusionART guide](../../../docs/algo/diffusionart.md) for the algorithm and
[V1 trainer guide](../../../docs/start/diffusion_v1.md) for separate actor and
rollout pools.

## Install

Follow the project [installation guide](../../../docs/start/install.md).
Use Diffusers and vLLM-Omni versions with Qwen-Image 2.1 support, and
Transformers with Qwen3-VL support. Turbo requires support for the checkpoint's
saved `sample_sigmas` schedule. Install `liger-kernel` and
`quack-kernels>=0.6.5` for actor training.

The default Qwen-Image 2.1 recipe uses 8 actor GPUs and 2 rollout GPUs; Turbo
uses 8 actor GPUs and 8 rollout GPUs. Actors use SP=1 and microbatch size 1.
Rollout uses TP=1, so each rollout GPU must fit the model. Allocate additional
resources if your reward function uses a GPU model.

## Prepare the data

Prepare training and validation parquet files with the following fields:

| Field | Contents |
| --- | --- |
| `prompt` | Text or chat messages; one `<image>` placeholder per reference image. |
| `images` | Optional image paths or parquet `bytes`/`path` records, in placeholder order. |
| `data_source` | Task identifier passed to the reward function. |
| `reward_model.ground_truth` | Reference passed to the reward function. |
| `extra_info` | Required output `height` and `width`, both positive multiples of 32. |

For text-to-image training, omit images and placeholders. Image editing accepts
RGBA references and preserves alpha in generated images. Structured image
content blocks are also supported; omit the separate `images` list when using
them.

Set the output dimensions in every training and validation row, for example:

```json
{"height": 1024, "width": 1536}
```

Top-level `height` and `width` columns are also supported and take precedence
over `extra_info`. Missing or invalid dimensions raise an error before generation.
Different prompts can use different dimensions; all samples for one prompt use
the same dimensions.

The custom reward function receives `data_source`, `solution_image`,
`ground_truth`, and `extra_info` by keyword. `solution_image` is a uint8 CHW
image. Return a mapping with `score` and optional scalar metrics. See the
[reward guide](../../../docs/algo/async_reward.md) for reward configuration.

## Launch training

Run from the repository root with paths accessible to all workers:

```bash
export TRAIN_FILE=/data/train.parquet
export VAL_FILE=/data/val.parquet
export REWARD_FUNCTION_PATH=/path/to/reward.py
export REWARD_FUNCTION_NAME=compute_score

ART_MODEL_PATH=/models/Qwen-Image-2.1 \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21.sh
```

For Turbo:

```bash
ART_MODEL_PATH=/models/Qwen-Image-2.1-Turbo \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo.sh
```

Both recipes use 8 prompts per batch, 8 samples per prompt, learning rate
`1e-5`, and 1000 trainer steps. Set `PROMPTS_PER_BATCH`, `GROUP_SIZE`, `ART_LR`,
and `TRAIN_STEPS` to change these values. Group size must be even, and each
update window must divide evenly across actor GPUs. `SYNC_STEPS` defaults to 1
and must divide `PROMPTS_PER_BATCH`.

Qwen-Image 2.1 uses 28 training and 40 validation denoising steps; Turbo uses 8
for both. Turbo follows its checkpoint's saved sigma schedule. Validation and
checkpointing run every 20 trainer steps, with validation also run before
training. Logging uses console and TensorBoard.

Append Hydra overrides to the command, for example:

```bash
ART_MODEL_PATH=/models/Qwen-Image-2.1-Turbo \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo.sh \
    trainer.total_training_steps=20 \
    trainer.logger=console
```

Weight synchronization defaults to `omni_delta_sharded`. Set
`CKPT_BACKEND=nccl` for full-weight synchronization. The recipes enable timestep
staging and selective activation checkpointing; see the
[DiffusionART guide](../../../docs/algo/diffusionart.md#memory-and-precision)
for memory settings.
