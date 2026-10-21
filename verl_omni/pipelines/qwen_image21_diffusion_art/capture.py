# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect Qwen-Image 2.1 replay data from one native forward."""

from contextlib import contextmanager

from verl_omni.pipelines.diffusion_art.capture import ARTTrajectoryRecorder, frozen_cpu


@contextmanager
def capture_art_rollout(pipe, sampling, policy_version):
    """Capture one request and restore its scheduler and hooks on exit."""
    cache = {}
    conditions = []
    recorder = None
    call_index = 0
    endpoint = None
    original_step = pipe.scheduler.step
    cfg = (sampling.true_cfg_scale or 1.0) > 1

    def pre_hook(module, args, kwargs):
        if len(conditions) == (2 if cfg else 1):
            return
        inputs = frozen_cpu(
            {
                key: kwargs[key]
                for key in ("encoder_hidden_states", "encoder_hidden_states_mask", "img_shapes", "img_mask")
            }
        )
        if not conditions:
            frames, height, width = kwargs["img_shapes"][0][-1]
            inputs["image_latents"] = frozen_cpu(kwargs["hidden_states"][:, : -(frames * height * width)])
        conditions.append(inputs)

    def step(model_output, timestep, sample, *args, **kwargs):
        nonlocal recorder, call_index, endpoint
        if recorder is None:
            if sample.shape[0] != 1:
                raise ValueError("Use rollout.n for ART groups; each request must generate one image.")
            recorder = ARTTrajectoryRecorder(
                len(pipe.scheduler.timesteps), (sampling.extra_args or {}).get("art_step_indices")
            )
        index = pipe.scheduler.step_index
        if index is None:
            index = pipe.scheduler.begin_index
        if index is None:
            index = pipe.scheduler.index_for_timestep(timestep)
        sigma = pipe.scheduler.sigmas[index]
        recorder.record(
            call_index, sample[0], model_output[0], timestep, sigma, pipe.scheduler.sigmas[index + 1] - sigma
        )
        result = original_step(model_output, timestep, sample, *args, **kwargs)
        endpoint = result[0] if isinstance(result, tuple) else result.prev_sample
        call_index += 1
        return result

    hook = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    pipe.scheduler.step = step
    try:
        yield cache
        positive = conditions[0]
        reference = positive.pop("image_latents")
        cache.update(
            recorder.finish(endpoint[0]),
            policy_version=policy_version,
            conditions={
                "inputs": positive,
                "negative_inputs": conditions[1] if cfg else None,
                "image_latents": reference,
                "true_cfg_scale": float(sampling.true_cfg_scale or 1.0),
            },
        )
    finally:
        pipe.scheduler.step = original_step
        hook.remove()
