# MiniMax H3 T2VA, FL2VA, and Ref2VA FlowGRPO

Last updated: 09/05/2026

These recipes train `MiniMaxAI/MiniMax-H3` LoRA adapters with FlowGRPO for
text-to-audio-video (T2VA), first-frame image-to-audio-video (FL2VA), and
reference-to-audio-video (Ref2VA) generation. The launchers configure a
Diffusers H3 Actor and vLLM-Omni rollout for joint video and audio generation,
with CLAP and ImageBind rewards for T2VA. FL2VA and Ref2VA use Qwen3.8-27B to
score the prompt, reference images and/or videos, and the complete generated
video. This visual reward does not score audio.

T2VA supports NVIDIA GPUs and Ascend NPUs. The FL2VA and full multimodal
Ref2VA paths target NVIDIA GPUs.

## Install

Follow the project [installation guide](../../../docs/start/install.md). In
particular, install the platform backend, the repository-pinned vLLM-Omni
revision, and the training dependencies in that order. Run the commands below
from the verl-omni repository root.

For NVIDIA GPU:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
```

For Ascend NPU:

```bash
uv pip install vllm==0.28.0
uv pip install "vllm-ascend @ git+https://github.com/vllm-project/vllm-ascend.git@$(cat .github/vllm_ascend_pin.txt)"
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
```

Install the tested Diffusers revision that provides
`MiniMaxH3Transformer3DModel`:

```bash
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"
```

## Prepare the checkpoint

Download the complete MiniMax H3 repository rather than only one subfolder:

```bash
export MODEL_ROOT="$HOME/models/MiniMax-H3"

huggingface-cli download MiniMaxAI/MiniMax-H3 \
  --local-dir "$MODEL_ROOT"
```

The recipe uses two representations from that download:

```text
MiniMax-H3/
|-- FL2VA/             # vLLM-Omni T2VA rollout pipeline
|   `-- transformer/
|-- Ref2VA/            # vLLM-Omni Ref2VA rollout pipeline
|   `-- transformer/
|-- transformer/       # Diffusers T2VA Actor weights and config
`-- transformer_ref/   # Diffusers Ref2VA Actor weights and config
```

Set the corresponding paths before launching:

```bash
export MODEL_PATH="$MODEL_ROOT/FL2VA"
export ACTOR_CONFIG_PATH="$MODEL_ROOT/transformer"
```

The scripts derive `ACTOR_CONFIG_PATH` as `$(dirname "$MODEL_PATH")/transformer`
when it is not set explicitly. The Ref2VA launcher takes the repository root
as `MODEL_PATH` and resolves `Ref2VA/` and `transformer_ref/` separately. Do not
replace either official rollout transformer with a symlink to a Diffusers
transformer; rollout and Actor loading use different checkpoint layouts.

## Prepare the data

T2VA uses prompt-only data and reuses the MiniMax H3 DiffusionNFT converter.
Prepare an input directory containing either:

- `train.txt` and `test.txt`, with one prompt per line; or
- `train.jsonl` and `test.jsonl`, with a `prompt`, `text`, or `caption` field.

Convert the splits to verl-omni parquet files:

```bash
export RAW_PROMPT_DIR=/path/to/raw_prompts
export DATA_DIR="$HOME/data/vid_prompt/verl_omni"

python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
  --input_dir "$RAW_PROMPT_DIR" \
  --output_dir "$DATA_DIR"
```

This writes `$DATA_DIR/train.parquet` and `$DATA_DIR/test.parquet`, the paths
consumed by the T2VA launchers. Use `--train_size` or `--val_size` to create a
smaller debugging dataset.

FL2VA conditions each clip on a first frame. Reuse the DiffusionNFT FL2VA
converter (symlinked here as `prepare_fl2va_data.py`), which emits one
`<image>` token and `frame_indices=[0]`:

```bash
export RAW_FL2VA_DIR=/path/to/raw_fl2va
export FL2VA_DATA_DIR="$HOME/data/fl2va/verl_omni"

python3 examples/flowgrpo_trainer/minimax_h3/prepare_fl2va_data.py \
  --input_dir "$RAW_FL2VA_DIR" \
  --output_dir "$FL2VA_DATA_DIR" \
  --frame_mode first
```

Each `train.jsonl` / `test.jsonl` row carries a prompt, a first-frame image
path relative to the input directory, and an explicit output specification:

```json
{"prompt":"The person waves at the camera.","images":["refs/person.png"],"extra_info":{"resolution":"1344x768","num_frames":124,"frame_rate":24.0}}
```

For Ref2VA, create `train.jsonl` and `test.jsonl` under one input directory.
Each row accepts `images`, `videos`, and `audios`; paths may be absolute or
relative to the input directory. At least one image or video is required.
Video entries may be paths or objects with `path` and `start_time_seconds`:

```json
{"prompt":"The subject waves while the scene remains unchanged.","images":["refs/person.png"],"videos":[{"path":"refs/motion.mp4","start_time_seconds":0.0}],"audios":["refs/music.wav"],"extra_info":{"resolution":"768x1344","num_frames":141,"frame_rate":24.0}}
```

Convert both splits with:

```bash
python3 examples/flowgrpo_trainer/minimax_h3/prepare_ref2va_data.py \
  --input_dir /path/to/ref2va_jsonl_and_media \
  --output_dir "$HOME/data/minimax_h3_ref2va"
```

The official limits are at most 9 images, 3 videos, 3 standalone audios, and
12 files total. Each video/audio clip must be 2–15 seconds and total reference
media duration must not exceed 15 seconds. A standalone audio reference needs
at least one visual reference. Video soundtracks are detected and conditioned
automatically; do not list them again under `audios`.

Reference-condition rows are padded to `MAX_PROMPT_EMBEDS`, with explicit row
counts retained for Actor replay. The value must cover the largest per-sample
condition layout; multiple references at the default 2048-pixel short edge may
require a larger cap or a smaller `REF_IMAGE_SHORT_EDGE`. Generated target
trajectories may have different sizes across samples and never repeat reference
rows across SDE steps in Ref2VA.

### Per-row output specifications

FL2VA and Ref2VA use the same required dataset schema for FlowGRPO and
DiffusionNFT. Every row must set all three fields in `extra_info`, with no
fallback defaults:

| Field | Constraint |
| --- | --- |
| `resolution` | Width x height; positive multiples of 32, aspect ratio from 1:4 to 4:1. |
| `num_frames` | Actual frame count, `17*n+5` and 4–15 seconds; e.g. 107, 124, 141, 209. |
| `frame_rate` | Explicitly required; the current H3 engine supports 24 only. The launchers export videos with `trainer.video_fps=24`. |

Choose FL2VA targets to match your first-frame/SFT preprocessing, and select
Ref2VA targets independently of the reference sizes. Train and validation both
use the row's fields. Missing/invalid fields and pipeline duration overrides
fail before generation. One row's `rollout.n` samples retain its specification.

The standard Ray/FSDP path supports mixed resolutions and frame counts within
one logical batch. Each sample's current/next latent trajectories and packed
layouts are transported independently. The actor restores one sample per
forward for both log-prob replay and training. Log-probs and advantages keep
their `[batch, steps]` shape, and gradients accumulate across samples and SDE
steps before an optimizer update. With the FlowGRPO launchers' batch size 32,
rollout n=8 and PPO mini-batch 16, each round generates 256 samples and makes
two optimizer updates of 128 samples each. Increasing n to 16 produces 512
samples and two updates of 256 samples each.

The standard dataset sampler supports `data.shuffle`, `data.seed`, and dataloader
resume. It does not create resolution buckets; only the final incomplete global
training batch is dropped by the dataloader. T2VA launchers retain their global
output settings.

Both H3 adapters declare `ragged_rollout_tensor_dims()` and
`ragged_model_output_dims()`. These hooks describe the single variable axis
of each packed tensor for the shared transport and engine. New H3 algorithms
must provide their own registered adapter and validate replay/loss behavior;
adding a loss name alone does not establish mixed-shape support.
Worker RPCs always place the variable axis immediately after batch, so
splitting, gathering and serialization preserve trajectory layouts. The engine
restores model axes before forwarding each sample. Old/reference policy means
use the same transport and return to per-sample tensors on the driver.

## Install reward dependencies

FL2VA and Ref2VA use the same `compute_score_vlm` scorer as DiffusionNFT.
Place the Qwen3.8-27B checkpoint at `/models/Qwen3.8-27B`, or set
`REWARD_MODEL_PATH` to its local directory. `REWARD_TP` defaults to 4 and must
divide the GPU count (`NUM_GPUS` for FL2VA, `N_GPUS` for Ref2VA). The reward
model runs through vLLM on the shared GPU pool.

The scorer accepts image references, video references, or both, including
Ref2VA rows with only video references. It sends reference videos and the
complete generated video as video inputs and evaluates reference fidelity,
prompt alignment, visual quality and temporal consistency. It does not score
reference or generated audio. Prepare data on the training machine or preserve
the absolute paths in `extra_info.source_images` and `extra_info.source_videos`
on that machine.

The T2VA launchers enable both CLAP and ImageBind rewards. Install their
dependencies before training. CLAP uses `transformers` and `torchaudio`, which
are included in the standard training environment. Its default checkpoint is
downloaded from `laion/larger_clap_general` unless `CLAP_MODEL_PATH` points to
a local copy.

ImageBind is distributed separately under the CC-BY-NC-SA 4.0 non-commercial
license. Install it and its video dependency separately:

```bash
uv pip install 'git+https://github.com/facebookresearch/ImageBind.git'
uv pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Download `imagebind_huge.pth` and set its location:

```bash
export IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth
```

By default, CLAP runs on `$REWARD_DEVICE:0` and ImageBind on
`$REWARD_DEVICE:1`, where `REWARD_DEVICE` is `cuda` for the GPU launcher and
`npu` for the NPU launcher. Both devices must be visible to the reward worker.
These rewards validate generated audio/video alignment but do not directly
measure fidelity to the supplied references.


## Launch

### NVIDIA GPU

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/vid_prompt/verl_omni" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

The GPU launcher uses Actor `_flash_3_varlen_hub` and rollout
`FLASH_ATTN_3_HUB`. On hardware without FA3 support, append compatible Hydra
overrides:

```bash
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh \
  actor_rollout_ref.model.attn_backend=native \
  actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA
```

### NVIDIA GPU (FL2VA)

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/fl2va/verl_omni" \
REWARD_MODEL_PATH=/models/Qwen3.8-27B \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

FL2VA reuses the same CPS FlowGRPO configuration as T2VA. The first-frame
condition rows are held fixed across the reverse-SDE window and re-injected
after every transition, so only the target video/audio rows are scored.

FL2VA defaults to Actor `flash_varlen_hub` and rollout `FLASH_ATTN_HUB`
(FA2), which support A100/A800 GPUs. On Hopper GPUs, set
`ACTOR_ATTN_BACKEND=_flash_3_varlen_hub ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB`
to use FA3 on both sides. The rollout backend's automatic FA3 fallback does
not change the Actor backend.

On the training machine, use a dataset with at least four rows, both
orientations and two valid frame counts for a two-step smoke run. After setting
the model/data/reward paths above and activating `.venv`, run:

```bash
ROLLOUT_N=2 TOTAL_TRAINING_STEPS=2 \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  data.val_max_samples=4 \
  trainer.save_freq=-1 \
  trainer.test_freq=1
```

This command assumes eight actor GPUs. Use the Ref2VA launcher with its model
root for the other task. Check finite losses, near-one replay ratios before
the first update, and the exported per-row video shapes. CPU tests cover tensor
transport/replay/gradient accumulation; real multi-GPU execution must be tested
on the training machine.

### NVIDIA GPU (Ref2VA)

```bash
MODEL_PATH="$MODEL_ROOT" \
DATA_DIR="$HOME/data/minimax_h3_ref2va" \
REWARD_MODEL_PATH=/models/Qwen3.8-27B \
REF_IMAGE_SHORT_EDGE=512 \
VAL_REF_IMAGE_SHORT_EDGE=1024 \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_ref2va_lora.sh
```

Ref2VA preserves the original Agent Loop token IDs and adds the official
reference presentation in rollout. Reference image/video rows use timestep
`0.999`, reference-audio rows use `1.0`, and all reference rows remain fixed
through every stochastic reverse-SDE transition. They are transported once and
reinserted by the Actor; only generated video and audio rows are stored in the
trajectory and contribute to the FlowGRPO log probability. The launcher keeps
the official reference-image short edge of 2048 by default. Set
`REF_IMAGE_SHORT_EDGE` for training and `VAL_REF_IMAGE_SHORT_EDGE` for
validation to multiples of 32 from 256 through 2048. The validation setting
defaults to the training value.

### Ascend NPU

```bash
MODEL_PATH="$MODEL_ROOT/FL2VA" \
DATA_DIR="$HOME/data/vid_prompt/verl_omni" \
IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_npu.sh
```

The NPU launcher uses Actor `_native_npu`, rollout `TORCH_SDPA`, and Actor
parameter and optimizer offload. It sources the Ascend toolkit and ATB
environments from `ASCEND_HOME_PATH`, which defaults to
`/usr/local/Ascend/ascend-toolkit`.

The launchers disable W&B logging by default and write metrics locally.
Checkpoints and logs are written under
`outputs/<launcher-name>/` unless `OUTPUT_DIR` is set.

## Default configuration

| Setting | Default |
| --- | --- |
| Devices | 8 GPU / 16 NPU |
| Rollout DiT TP | 2 GPU / 4 NPU |
| Text-encoder TP | Same as rollout TP |
| Training batch size | 32 |
| PPO mini-batch / per-device micro-batch | 16 / 1 |
| Rollouts per prompt | 8 |
| LoRA rank / alpha | 64 / 128 |
| Learning rate | `3e-4` |
| T2VA training output | `256x384`, 121 requested frames at 24 FPS |
| T2VA validation output | `512x768`, 121 requested frames at 24 FPS, 40 inference steps |
| FL2VA/Ref2VA output | Per-row `resolution`, `num_frames`, `frame_rate` in both training and validation |
| Rollout inference steps | 10 |
| CPS window | 3 contiguous transitions from `[0, 8)` |
| Total training steps | 100 |

The Ref2VA launcher defaults to rollout/text-encoder TP 4,
10 inference steps, `MAX_PROMPT_EMBEDS=12288`, rollout n=8,
and Actor micro-batch 1. It enables layerwise rollout offload and FSDP2 Actor
parameter/optimizer offload because reference presentations can be much longer
than T2VA prompts.

The GPU count must be divisible by the rollout replica size: `ROLLOUT_TP`
for T2VA/Ref2VA, or `ROLLOUT_TP * ROLLOUT_SP` for FL2VA. `TEXT_ENCODER_TP`
cannot exceed that replica size; H3 supports text-encoder TP sizes 1, 2, 4,
and 8. The recipe uses an Actor micro-batch of 1 because samples with different packed
video/audio/text layouts cannot share one H3 forward. For per-row FL2VA and
Ref2VA outputs, the engine also enforces micro-batch size 1 during replay.

The T2VA launcher uses a named `ASPECT_RATIO`, one of `21:9`, `16:9`, `4:3`,
`1:1`, `3:4`, or `9:16`. The explicit height and width select the generated
canvas and must be multiples of 32. FL2VA and Ref2VA take resolution, frame
count and frame rate exclusively from their dataset rows.

### FL2VA rollout throughput

The FL2VA launcher defaults to DiT TP 2 with Ulysses SP 4, giving one rollout
replica on eight GPUs. Text-encoder TP and VAE tile parallelism remain 8.
`ROLLOUT_TP` shards DiT weights, while `ROLLOUT_SP` shards its packed sequence;
Ring remains disabled. The replica size is their product. `VAE_PP` and
`TEXT_ENCODER_TP` each default to that replica size and must be either 1 or
the full replica size because intermediate encoder/VAE groups are not
supported by the current backend.

VeRL's `rollout.tensor_model_parallel_size` reserves the whole replica, so
it remains 8 for TP 2 with SP 4. The actual DiT dimensions are passed through
`engine_kwargs.vllm_omni.tensor_parallel_size=2` and `ulysses_degree=4`.
The FlowGRPO adapter explicitly installs H3's native sequence split/gather
hooks and initializes VAE parallelism because vLLM-Omni's custom-pipeline
loader bypasses the registry setup. Actor sequence parallelism and old-log-prob
recomputation are independent of this rollout configuration.

FL2VA allows 1800 seconds for rollout startup and per-stage initialization.
Concurrent replicas can exceed vLLM-Omni's default 600-second startup limit;
set `ROLLOUT_INIT_TIMEOUT` and `ROLLOUT_STAGE_INIT_TIMEOUT` to adjust these
limits. The launcher forwards `VLLM_LOGGING_LEVEL=INFO` to Ray workers and
defaults `RAY_DEDUP_LOGS=0` so initialization progress remains visible for
each replica. Set `VLLM_LOGGING_LEVEL=WARN RAY_DEDUP_LOGS=1` for quieter logs.
A timeout alone does not distinguish slow loading from a stalled worker;
check the last initialization messages on the affected node if it persists.

VAE tile parallelism distributes video decoding across the replica's GPUs;
it retains a full VAE on each GPU. First-frame image encoding still runs on
rank 0, and audio decoding is replicated. The native decoder falls back to
local decoding when the output has fewer tiles than parallel ranks.

To compare another DiT split on eight GPUs with encoder/VAE settings fixed:

```bash
ROLLOUT_TP=4 ROLLOUT_SP=2 TEXT_ENCODER_TP=8 VAE_PP=8 \
OUTPUT_DIR=outputs/run_minimax_h3_fl2va_lora_tp4_sp2 \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

Compare against the default `ROLLOUT_TP=2 ROLLOUT_SP=4` and the previous
TP-only configuration, `ROLLOUT_TP=8 ROLLOUT_SP=1`, with the same dataset,
inference steps, and rollout count, excluding startup and warmup. All three
use one eight-GPU replica with text-encoder TP 8 and VAE tile parallelism 8.
Reducing DiT TP increases per-GPU weight residency, while SP reduces the
sequence length processed by each rank. Check the longest video shapes;
speedup and peak memory must be measured on the training machine.

With `ROLLOUT_TP=1 ROLLOUT_SP=8`, each GPU retains the full 61.7 GiB DiT
weights. This configuration failed during pipeline weight loading on 80 GiB
GPUs with text-encoder TP 8 and VAE tile parallelism 8: rollout workers reached
about 77 GiB alongside the colocated actor/reward processes. The worker CUDA
OOM appeared before the parent process's `EOFError`. Reducing the video size
or rollout count cannot resolve this weight-loading failure. Keeping TP 1
requires additional weight sharding or component offload; the default TP 2
with SP 4 reduces DiT weight residency while preserving Ulysses inference.

To collect pipeline stage timings, append
`+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_diffusion_pipeline_profiler=True`.

The server must preserve `text_encoder_tp_size` when converting CLI args:
on the pinned vLLM-Omni version, the field is absent from both
`OmniEngineArgs` and `OrchestratorArgs`. Dropping it silently selects TP 1
and places the H3 text encoder on rank 0, even when the launcher prints TP 8.

### Environment overrides

Common environment overrides are:

| Variable | Purpose |
| --- | --- |
| `WORKSPACE` | Base directory for default model and data paths |
| `MODEL_PATH` | Official `MiniMax-H3/FL2VA` rollout pipeline |
| `ACTOR_CONFIG_PATH` | Root Diffusers Actor weights and config directory |
| `DATA_DIR` | Directory containing `train.parquet` and `test.parquet` |
| `OUTPUT_DIR` | Checkpoint and log root |
| `NUM_GPUS` | Devices per node for T2VA/FL2VA; Ref2VA uses `N_GPUS` |
| `ROLLOUT_TP` | vLLM-Omni DiT tensor parallel size |
| `ROLLOUT_SP` | FL2VA DiT Ulysses sequence parallel size; defaults to 4 |
| `TEXT_ENCODER_TP` | H3 text-encoder tensor parallel size |
| `VAE_PP` | FL2VA VAE tile parallel size; defaults to `ROLLOUT_TP * ROLLOUT_SP`, or set 1 to disable |
| `MAX_PROMPT_EMBEDS` | Prompt/reference-row padding cap; defaults to 12288 |
| `REF_IMAGE_SHORT_EDGE` | Ref2VA training image short edge; defaults to 2048 |
| `VAL_REF_IMAGE_SHORT_EDGE` | Ref2VA validation image short edge; defaults to the training value |
| `REWARD_MODEL_PATH` | FL2VA/Ref2VA VLM checkpoint directory; defaults to `/models/Qwen3.8-27B` |
| `REWARD_TP` | FL2VA/Ref2VA reward tensor parallel size; defaults to 4 |
| `REWARD_NUM_WORKERS` | T2VA reward worker count |
| `REWARD_DEVICE` | T2VA reward device type, such as `cuda` or `npu` |
| `CLAP_MODEL_PATH` | T2VA CLAP model ID or local path |
| `IMAGEBIND_MODEL_PATH` | T2VA ImageBind checkpoint path |
| `ROLLOUT_N` | Samples per prompt in FL2VA/Ref2VA; defaults to 8 |
| `ASPECT_RATIO` | T2VA canvas ratio |
| `HEIGHT` | T2VA training output height |
| `WIDTH` | T2VA training output width |
| `NUM_FRAMES` | T2VA training and validation frame count |
| `INFER_STEPS` | Training rollout inference steps |
| `VAL_HEIGHT` | T2VA validation output height |
| `VAL_WIDTH` | T2VA validation output width |
| `TOTAL_TRAINING_STEPS` | Number of trainer steps |

Extra Hydra overrides may be appended to either launcher command.

## Current limitations

- The TransferQueue-specific Agent Loop path does not yet support variable
  target/reference tensors; use the standard Ray trainer and FSDP/FSDP2 engine
  for mixed-shape training. VeOmni is not covered by this integration.
- Distilled checkpoint-specific sigma schedules are rejected because Actor and
  rollout replay currently use the standard H3 video/audio schedules.
- The FL2VA/Ref2VA VLM reward evaluates visual content only. Use an additional
  reward for audio objectives. T2VA launchers require CLAP and ImageBind.
