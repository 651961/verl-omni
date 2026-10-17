# MiniMax H3 FL2VA and Ref2VA DiffusionART

Last updated: 10/17/2026

Full-parameter DiffusionART training for MiniMax H3 first/last-frame-to-audio-video
(FL2VA) and reference-to-audio-video (Ref2VA) generation. The recipes use the V1
`separate_async` trainer, FSDP2 actors, and native vLLM-Omni rollout. Video and audio
share a scalar reward, with their separate Euler schedules preserved during replay.
Resolution and frame count are specified per dataset row; generation uses 24 fps.

## Prerequisites

Follow the [installation guide](../../../docs/start/install.md) and
[V1 trainer guide](../../../docs/start/diffusion_v1.md), then activate the environment:

```bash
source .venv/bin/activate
```

- Diffusers with `MiniMaxH3Transformer3DModel` and vLLM-Omni with the native
  `MiniMaxH3Pipeline`, `_build_denoise_inputs`, and `_unpack_denoised_rows` interfaces.
- FlashAttention 2 Hub kernels (`kernels-community/flash-attn2`) compatible with
  the installed PyTorch/CUDA. Actor uses `flash_varlen_hub`; rollout uses
  `FLASH_ATTN_HUB`, suitable for GPUs such as A800.
- A MiniMax-H3 repository containing `FL2VA/` and `transformer/` for FL2VA, or
  `Ref2VA/` and `transformer_ref/` for Ref2VA. `MODEL_PATH` points to the repository
  root; `ACTOR_CONFIG_PATH` can override the Diffusers transformer directory.
- Separate actor and rollout GPU pools. Defaults use 8 actor GPUs and 8 rollout
  GPUs, with rollout TP=4 and text encoder TP=4. Reserve any additional resources
  required by your reward function.

The recipes use FP32 master weights and Adam states, BF16 compute, actor SP=1,
and one sample per actor microbatch. Use CFG-distilled checkpoints with
`true_cfg_scale=1.0`. LoRA, step execution, multistep samplers, approximate denoiser
caches, latent editing/refinement, and locked source audio are unsupported.

## Prepare the dataset

Reuse the existing H3 converters:

- FL2VA: [prepare_fl2va_data.py](../../diffusionnft_trainer/minimax_h3/prepare_fl2va_data.py).
  The example uses first-frame conditioning (`frame_indices='[0]'`).
- Ref2VA: [prepare_ref2va_data.py](../../diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py),
  supporting reference images, videos, and audio.

Add `height`, `width`, and `num_frames` to every training and validation row's
`extra_info`, preserving the converter's existing metadata. Two rows in the same
batch can use:

```json
{"height": 288, "width": 448, "num_frames": 120}
{"height": 576, "width": 928, "num_frames": 240}
```

Height and width must be positive integer multiples of 32; frame count must be a
positive integer. Missing fields are rejected before generation. Each prompt's
`GROUP_SIZE` responses share the same geometry; different prompts can use different
sizes. H3 rounds frame counts upward to `17n+5`: 120 frames become 124 (about 5.17 s),
and 240 become 243 (about 10.13 s). Native duration limits still apply.

## Launch

Run from the repository root with prepared parquet files and a video reward scorer.
The scorer receives `solution_image` through the visual reward interface, with
generated audio in `extra_info["audio"]` and `extra_info["audio_sample_rate"]` (32000 Hz).

```bash
export MODEL_PATH=/models/MiniMax-H3
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/test.parquet
export REWARD_FUNCTION_PATH=/path/to/video_reward.py
export REWARD_FUNCTION_NAME=compute_score

# Choose the recipe matching your dataset.
bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_fl2va.sh
bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_ref2va.sh
```

Both scripts accept trailing Hydra overrides. Main environment settings are
`PROMPTS_PER_BATCH`, `GROUP_SIZE` (at least 2), `ART_LR`, `TRAIN_STEPS`, and
`OUTPUT_DIR`. `ART_STEP_INDICES=null` records all inference steps; use a subset such
as `[0,2,4]` to reduce trajectory storage. All samples must record the same number
of steps. Media dumping is disabled by default; validation reward scoring remains enabled.

With 8 GPUs per node and `fsdp_size=8`, actor parameters are sharded within a node
and replicated across nodes, with cross-node gradient synchronization:

| Total GPUs | `ACTOR_NNODES` | `ROLLOUT_NNODES` | Actor mesh | Rollout replicas (TP=4) |
| --- | --- | --- | --- | --- |
| 16 (default) | 1 | 1 | FSDP, 8 shards | 2 |
| 32 | 2 | 2 | HSDP, 2 replicas × 8 shards | 4 |

Set `ACTOR_NNODES=2 ROLLOUT_NNODES=2` for the 32-GPU layout. The default batch of
8 prompts × 8 responses divides evenly across both 8 and 16 actor ranks.

## Selective activation checkpointing

The FSDP diffusion engine supports caching GEMMs by full Linear module path,
using the same mechanism as Qwen Image 2.1. H3's MLP down-projection and attention
output projection can be selected with a trailing override on either launcher:

```bash
bash examples/diffusionart_trainer/minimax_h3/run_minimax_h3_fl2va.sh \
  actor_rollout_ref.model.gradient_checkpointing_save_ops='[transformer_blocks.*.ff.net.2,transformer_blocks.*.attn.to_out.0]'
```

These are case-sensitive glob patterns; each must match at least one `nn.Linear`.
The selector caches matrix multiplication outputs inside those modules, leaving
the other projections and norms eligible for recomputation. It does not select
token-refiner projections. To also cache attention-core outputs, add the
operator of the actual backend to this same list;
Qwen's SDPA FlashAttention operator must not be assumed to match H3's FA2 backend.
An empty list means ordinary block checkpointing. H3 launchers retain their
existing behavior until the list is overridden. Named Linear selection
requires `use_regional_compile=false` and is supported only by the FSDP engine.
Measure peak memory and update time for the H3 sequence layout before adopting
this optional configuration; Qwen's memory estimates do not apply to H3.

## Tests

CPU tests cover dual schedules, reference conditions, variable-size transport,
per-row geometry, cleanup on failure, and tiny-model backward:

```bash
python -m pytest -q \
  tests/pipelines/test_minimax_h3_diffusion_art_on_cpu.py \
  tests/trainer/diffusion/test_art_core_on_cpu.py \
  tests/trainer/diffusion/test_art_wiring_on_cpu.py \
  tests/pipelines/test_qwen_image21_diffusion_art_on_cpu.py
```

GPU smoke tests use four GPUs, tiny random weights, synthetic media, and native/SDPA
attention. Each runs one actor update with mixed resolutions and frame counts:

```bash
python tests/special_e2e/run_minimax_h3_diffusion_art_tiny.py --task fl2va
python tests/special_e2e/run_minimax_h3_diffusion_art_tiny.py --task ref2va
```

Both cases are registered in the diffusion GPU smoke group. CPU checks do not
validate distributed weight synchronization, FA2 kernels, or full-model memory use.
