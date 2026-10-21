# DiffusionART

Last updated: 10/24/2026.

DiffusionART trains diffusion models with reward-weighted endpoint covariance and
frozen ODE trajectories. This integration ports the algorithm from
[DiffusionART](https://github.com/vanzll/DiffusionART) and supports Qwen-Image 2.1
(including Turbo) and MiniMax H3 FL2VA/Ref2VA.

The adapted objective, covariance, and window scheduling code is Copyright 2026
DiffusionART contributors, under [Apache 2.0](../../LICENSE). This port changes
batching and replay to use the repository's distributed trainer and adapters.

## Training flow

1. Generate an even number of trajectories per prompt and score every sample.
2. Compute endpoint/reward covariance over each complete prompt group.
3. Randomly divide each group's trajectories into two disjoint update windows.
4. Replay the selected ODE states, comparing actor velocities with frozen rollout
   velocities and reward guidance. Each window performs one optimizer update.
5. Publish the updated weights to the asynchronous rollout pool.

Both windows retain the same frozen guidance and all selected replay steps.
The default objective uses `velocity_linear_sigma`, RMS-normalized directions,
and scheduler-reference corrective masking. Mixed batches retain zero-guidance
restoring anchors and the original loss denominator. If the entire batch has no
usable guidance, both updates and their scheduler steps are skipped without
replacement sampling.

The supported execution path is online direct preference with V1
`separate_async`, FSDP2 full-parameter actors, `ppo_epochs=1`, and actor SP=1:

```yaml
algorithm:
  trainer_type: direct_preference
  sample_source: online
actor_rollout_ref:
  model:
    model_type: diffusion_art_model
    algorithm: diffusion_art
  actor:
    strategy: fsdp2
    diffusion_loss:
      loss_mode: diffusion_art
trainer:
  use_v1: true
  v1:
    trainer_mode: separate_async
```

No old/EMA adapter is maintained. A frozen reference model is needed only when
KL regularization is enabled. Each update window's trajectory count must be
divisible by actor data-parallel size. `parameter_sync_step` divides the prompt
batch into local batches; each local batch can perform two optimizer updates.
The learning-rate schedule accounts for both windows and advances only on updates.
Metrics include `actor/art_update_windows` and `actor/art_skipped_zero_guidance`.

MiniMax H3 consumes explicit `video_reward` and `audio_reward` fields returned by
the custom reward function. It computes covariance, direction RMS, corrective
masks, and feature means separately, then averages the two modality losses with
equal weights. The scalar `score` remains available for logging and validation.
See the [H3 recipe](../../examples/diffusionart_trainer/minimax_h3/README.md)
for the reward return contract and the author's scorer routing.

## Memory and precision

The recipes keep FP32 master weights, Adam states, and gradient reduction while
using BF16 model computation and serving weights. Rollout states and velocities
remain FP32. Timestep staging transfers the selected replay states from CPU one
step at a time. With per-sample rewards, training rows omit pixels unless
rollout dumps are enabled. Validation retains named decoded previews
without recording training trajectories.

`art_step_indices=null` replays every rollout step. Selecting a sorted, unique
subset reduces trajectory storage and replay computation, but changes objective
sampling. Selective activation checkpointing can retain selected operator outputs
instead of recomputing them; this trades more activation memory for less compute.
An empty `gradient_checkpointing_save_ops` list uses ordinary checkpointing.
Named Linear selectors require the ART FSDP engine and regional compilation off.
Neither setting has a workload-independent speed or memory benefit.

BF16 delta publication can reduce transfer volume when updates change only a
fraction of serving values. The seed and delta paths use identical casts;
FP32 master updates accumulate even when a serving value has not yet changed.
Savings depend on the measured changed-value ratio. Serving policy versions are
committed after successful weight application; asynchronous samples can remain stale.

## Examples and validation

- [Qwen-Image 2.1 and Turbo](../../examples/diffusionart_trainer/qwen_image21/README.md)
- [MiniMax H3 FL2VA and Ref2VA](../../examples/diffusionart_trainer/minimax_h3/README.md)

CPU tests cover objective arithmetic, replay gradients, window scheduling,
transport, and media contracts. They do not establish full-model GPU training,
multi-node behavior, convergence, or throughput. Measure those with the model's
recipe and workload before enabling an optimization in a production run.
