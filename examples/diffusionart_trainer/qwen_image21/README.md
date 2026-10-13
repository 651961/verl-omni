# Qwen-Image 2.1 DiffusionART

Last updated: 10/14/2026

Full-parameter RL post-training for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
and [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), using the V1
`separate_async` trainer, FSDP2 actors, and native vLLM-Omni rollout. Both recipes support
text-to-image generation, image editing, RGBA images, and per-sample output sizes.

DiffusionART computes reward/endpoint covariance within each prompt group and trains on
recorded ODE states and rollout velocities. The default objective uses
`velocity_linear_sigma`, RMS directions, and the scheduler-reference corrective mask.
It does not load an EMA policy or a frozen reference model by default.

## Prerequisites

Follow the [installation guide](../../../docs/start/install.md) and the
[V1 trainer guide](../../../docs/start/diffusion_v1.md). Activate the environment before launching:

```bash
source .venv/bin/activate
```

The environment must provide:

- Diffusers with `QwenImage21Pipeline`, `QwenImage21Transformer2DModel`, and
  `AutoencoderKLQwenImage21`. Turbo also requires pipeline-configured `sample_sigmas`
  ([Diffusers PR #14950](https://github.com/huggingface/diffusers/pull/14950)).
- Transformers with `Qwen3VLProcessor` and `Qwen3VLForConditionalGeneration`.
- vLLM-Omni with the native `vllm_omni.diffusion.models.qwen_image_21` pipeline.
  Turbo requires loading `sample_sigmas` from `model_index.json` and forwarding them to
  both full-forward and step execution.
- verl, TransferQueue, and the dependencies for the `omni_delta_sharded` checkpoint backend.

The vLLM-Omni revision pinned in `pyproject.toml` predates this model. Installing the
base project alone does not guarantee these capabilities. The local environment checked
for this recipe uses Diffusers `0.41.0`, Transformers `5.14.1`, vLLM `0.31.0`, and
vLLM-Omni `0.31.0rc2.dev145+gdb9278431`. Model imports and the Turbo scheduler were checked
on CPU; this is not an end-to-end GPU compatibility result.

By default, reserve **8 actor GPUs and 2 separate rollout GPUs**, plus any resources
needed by your reward function. Actor nodes use 8 GPUs each (`fsdp_size=8`, SP=1):
single-node FSDP2 becomes HSDP across multiple nodes. Rollout uses TP=SP=CFG-parallel=1,
so each rollout GPU must fit the native pipeline and its working memory.

## Prepare the dataset

Provide training and validation parquet files. The recipe selects `QwenImage21Dataset`,
which preserves raw prompts and image alpha channels for the model-specific agent loop.

| Field | Contents |
| --- | --- |
| `prompt` | A string or a list of chat messages. For editing, use one `<image>` placeholder per entry in `images`. |
| `images` | Optional reference images, in placeholder order. Use paths or parquet `bytes`/`path` records. |
| `data_source` | Reward/task identifier passed to the scorer and used in validation metrics. |
| `reward_model` | A mapping with `ground_truth`, whose meaning is defined by your reward function. |
| `extra_info` | Optional metadata, including `index`, `height`, `width`, and `output_resolution`. |

For example, an editing row can have the following structure before serialization to parquet:

```json
{
  "prompt": "<image>Change the background to a snowy mountain landscape.",
  "images": ["/data/reference.png"],
  "data_source": "image_edit",
  "reward_model": {"ground_truth": "A snowy mountain landscape behind the original subject."},
  "extra_info": {"index": 0, "height": 2048, "width": 2048}
}
```

For text-to-image, omit `images` and the `<image>` placeholder. Structured image content
blocks are also accepted; when using them, omit the separate `images` list. Reference
paths must be accessible to the workers. `negative_prompt` is optional and unused with
the recipes' `true_cfg_scale=1.0`.

### Resolution and RGBA

Set `OUTPUT_RESOLUTION=2048` for an output area around 2048 × 2048 pixels. The script
default is **1024**, so set this explicitly for 2048-resolution runs. Without explicit
width/height, text-to-image uses a square canvas; editing follows the last reference
image's aspect ratio.

Per-row `extra_info.output_resolution` overrides the global resolution default;
explicit `extra_info.height` and `extra_info.width` take precedence. These values can
also be top-level columns, which take precedence over `extra_info`. The native pipeline
aligns target dimensions down to multiples of 32. Different prompts may use different
resolutions and reference counts; all samples in one prompt group must share the same layout.

RGBA references retain alpha for the VAE; the Qwen3VL encoder receives a white-background
RGB composite. Generated images retain four channels. The reward function must decide
whether to preserve alpha or composite onto a background.

## Prepare the reward

Set `REWARD_FUNCTION_PATH` to a Python file or `pkg://` module and
`REWARD_FUNCTION_NAME` to its callable. The scorer is called by keyword with
`data_source`, `solution_image`, `ground_truth`, and `extra_info`; both synchronous and
asynchronous functions are supported. Return a dictionary containing `score` and optional
scalar metrics. See the [reward guide](../../../docs/algo/async_reward.md) for integration.

For these pixel-output recipes, `solution_image` is a single `torch.uint8` image in
`(C, H, W)` format, cropped to its original dimensions before scoring. Expect RGBA
(`C=4`), rather than assuming RGB. `ground_truth` comes from `reward_model.ground_truth`.

The scripts disable the built-in reward model pool and use your custom scorer. If it
calls a separate GPU service, provision that service independently. For named local
`reward.models`, streaming requires `reward.reward_model.enable_resource_pool=true` and
`offload=false` for every named model. Shared/offloaded models use the batch lifecycle.

## Run training

Launch from the repository root. Both scripts change to that directory, so relative
data and reward paths are resolved from the repository root.

Set the common inputs once, replacing the example paths with your dataset and scorer:

```bash
export TRAIN_FILE=/data/train.parquet
export VAL_FILE=/data/val.parquet
export REWARD_FUNCTION_PATH=/path/to/reward.py
export REWARD_FUNCTION_NAME=compute_score
export OUTPUT_RESOLUTION=2048
```

### Qwen-Image 2.1

```bash
ART_MODEL_PATH=/models/Qwen-Image-2.1 \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh
```

### Qwen-Image 2.1 Turbo

```bash
ART_MODEL_PATH=/models/Qwen-Image-2.1-Turbo \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo_async.sh
```

Both are standalone scripts with complete trainer configurations. The standard recipe
requires `ART_MODEL_PATH`; Turbo defaults to `/models/Qwen-Image-2.1-Turbo` when it is unset.

| Setting | Qwen-Image 2.1 | Qwen-Image 2.1 Turbo |
| --- | --- | --- |
| Launcher | [`run_qwen_image21_async.sh`](run_qwen_image21_async.sh) | [`run_qwen_image21_turbo_async.sh`](run_qwen_image21_turbo_async.sh) |
| Training/validation denoising steps | 28, configurable through `ROLLOUT_STEPS` | 8, using the checkpoint's saved sigma schedule |
| ART replay steps | All 28 by default | All 8 by default |
| CFG scale | 1.0 | 1.0 |
| Experiment name | `qwen_image21_async` | `qwen_image21_turbo_async` |

The standard recipe's 28 denoising steps are a sampling setting, not the number of
training updates or an ART requirement. Turbo automatically loads `sample_sigmas` from
the checkpoint: changing `num_inference_steps` alone does not override that schedule.
The Turbo script does not read `ROLLOUT_STEPS`.

### Training settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `TRAIN_STEPS` | `1000` | Trainer step limit, independent of denoising steps. |
| `PROMPTS_PER_BATCH` | `8` | Prompts per training batch. |
| `GROUP_SIZE` | `8` | Generated samples per prompt; must be at least 2. |
| `ART_LR` | `1e-6` | AdamW learning rate. |
| `ART_STEP_INDICES` | `null` | Zero-based rollout states to replay; `null` selects all. |
| `OUTPUT_RESOLUTION` | `1024` | Default output area is approximately this value squared. |
| `VAL_PROMPTS_PER_BATCH` | `PROMPTS_PER_BATCH` | Validation prompt count. |
| `VAL_GROUP_SIZE` | `1` | Validation samples per prompt. |
| `SYNC_STEPS` | `1` | Actor steps between weight publications; also divides the prompt batch into mini-batches. |
| `CKPT_BACKEND` | `omni_delta_sharded` | Weight publication backend; use `nccl` for a full-sync comparison. |
| `USE_REFERENCE` | `false` | Enable an additional frozen reference for velocity MSE regularization. |
| `REFERENCE_MSE_COEF` | `0.001` | Coefficient for the optional reference loss. |

At the defaults, one batch contains 8 × 8 = 64 trajectories. Actor micro-batches contain
one sample per GPU. Each selected trajectory state contributes a forward/backward pass
before the optimizer update; 8 or 28 replay steps do **not** mean 8 or 28 independent
optimizer updates. Gradient checkpointing adds block recomputation. Optional reference
regularization adds a reference model and one no-gradient forward per selected state.

`ART_STEP_INDICES='[0,2,4,6]'` selects four states for either recipe. Indices must be sorted,
unique, and within the actual schedule. Subsampling changes the training objective's
sampling; it is not a lossless performance optimization.

CLI Hydra overrides take precedence. For example:

```bash
TRAIN_STEPS=100 bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo_async.sh \
    trainer.test_freq=20 \
    trainer.save_freq=20
```

### Multiple nodes

`ACTOR_GPUS` and `ROLLOUT_GPUS` count total GPUs in their separate pools. For two actor
nodes and one rollout node (24 GPUs, excluding rewards):

```bash
ACTOR_GPUS=16 ROLLOUT_GPUS=8 \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo_async.sh
```

`ACTOR_NNODES` can also specify the actor node count; `ACTOR_GPUS_PER_NODE` must remain 8.
Rollout uses at most 8 GPUs per node by default. Override `ROLLOUT_GPUS_PER_NODE` and
`ROLLOUT_NNODES` when needed; the total must equal their product. `PROMPTS_PER_BATCH`
must be divisible by `SYNC_STEPS`, and each mini-batch's trajectory count must be
divisible by `ACTOR_GPUS`. Prepare the Ray cluster and consistent model/data paths on
all nodes as described in the [V1 guide](../../../docs/start/diffusion_v1.md).

## Precision and performance settings

- **Actor precision:** FP32 master parameters, gradient accumulation/reduction, and
  `torch.optim.AdamW` states; BF16 gathered parameters and compute. Rollout velocities
  and ODE states are retained in FP32.
- **FSDP2:** forward parameter prefetch is enabled. Disable it with
  `actor_rollout_ref.actor.fsdp_config.forward_prefetch=false` if needed. Regional
  compilation and deferred gradient synchronization are disabled.
- **Trajectory staging:** `enable_timestep_staging=true` transfers states from CPU one
  timestep at a time. Samples retain their own resolution/layout and use micro-batch 1.
  Transfers are synchronous; this setting does not promise copy/compute overlap.
- **Rollout:** native step execution is enabled, with `ROLLOUT_MAX_NUM_SEQS=4` and
  `ROLLOUT_ENFORCE_EAGER=false`. `ROLLOUT_STEP_EXECUTION=false` selects full-forward execution.
  The integration reuses native batching and per-request prefix KV. Actor replay preserves
  the prefill layout; step execution rejects layout changes within a trajectory.
- **Weight synchronization:** the first publication exports BF16 weights; later publications
  send changed BF16 values relative to the published copy. Small FP32 updates remain in
  the master weights until they cross a BF16 rounding boundary. Savings depend on the
  measured changed-weight ratio.
- **Asynchronous rewards:** each completed sample can be scored immediately. ART covariance
  waits for the full prompt group. Samples retain their generation policy version; the
  recipe uses a staleness threshold of 2 and drops expired samples. This is not strict on-policy training.

TransferQueue SimpleStorage holds training data; Ray handles worker scheduling and RPC.
After scoring, the training queue retains `uid`, rewards, `art_cache`, and small metadata.
Images remain only when needed for configured dumps; validation retains images without
training trajectories. Identical conditions, reference latents, and schedules are shared
within a group. Statistics do not fetch the full trajectory cache.

`TQ_STORAGE_UNITS` defaults to actor nodes plus rollout nodes (2); `TQ_STORAGE_SIZE`
defaults to 320 sample slots, not bytes. Increase capacity when increasing warmup or
in-flight batches. State/velocity buffers alone require about 56 MiB per 1024², 28-step
trajectory, or 224 MiB at 2048². At 2048² and 8 steps, they require about 64 MiB.
These estimates exclude endpoints, conditions, queue overhead, and temporary copies.

## Logging and validation

Logging defaults to `console`, project `diffusion_art`. Checkpoints and validation run
every 100 trainer steps; validation before training is disabled. Override `trainer.logger`,
`trainer.project_name`, and `trainer.experiment_name` as needed. See the
[metrics guide](../../../docs/start/metrics.md). ART MFU excludes reward computation,
VAE/text encoding, and gradient-checkpoint recomputation.

With the data/reward environment variables above, inspect either recipe without launching training:

```bash
ART_MODEL_PATH=/models/Qwen-Image-2.1 \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh --cfg job --resolve
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo_async.sh --cfg job --resolve
```

Core CPU checks (run each command in its own process):

```bash
python -m unittest discover -s tests/trainer/diffusion -p 'test_art_*_on_cpu.py'
python tests/pipelines/test_qwen_image21_diffusion_art_on_cpu.py
python -m unittest discover -s tests/utils -p 'test_qwen_image21_mfu_on_cpu.py'
python -m unittest discover -s tests/reward_loop -p 'test_streaming_reward_router_on_cpu.py'
```

A short GPU run uses the same actor/rollout resource requirements:

```bash
TRAIN_STEPS=2 bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_turbo_async.sh \
    trainer.test_freq=-1 trainer.save_freq=-1
# Standard-model one-step smoke:
ART_MODEL_PATH=/models/Qwen-Image-2.1 bash tests/gpu_smoke/run_qwen_image21_diffusion_art.sh
```

CPU coverage includes ART mathematics, tiny real Diffusers transformer replay gradients,
queue boundaries, and native rollout call contracts with runtime test doubles. Full-model
GPU generation/training, multi-node HSDP, checkpoint resume, and NCCL/delta equivalence
have **not** been validated for this recipe. Acceptance runs should cover text-to-image,
RGB/RGBA editing, mixed aspect ratios, reference on/off, and FP32 optimizer restoration.
There are no measured throughput or convergence results yet.
