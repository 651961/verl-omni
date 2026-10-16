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

"""Collect Qwen-Image 2.1 ART replay data from native full-forward batching."""

from contextlib import contextmanager

import torch

from verl_omni.trainer.diffusion.art_capture import ARTTrajectoryRecorder, frozen_cpu


@contextmanager
def capture_art_rollout(pipe, sampling_params, policy_version):
    """Scope hooks to one engine forward; keep each batch row's data separate."""
    extras = [sampling.extra_args or {} for sampling in sampling_params]
    caches = [{} for _ in extras]
    saved_inputs = []
    recorders = None
    call_index = 0
    last_endpoint = None
    original_step = pipe.scheduler.step
    cfg = (sampling_params[0].true_cfg_scale or 1.0) > 1

    def pre_hook(module, args, kwargs):
        if len(saved_inputs) >= (2 if cfg else 1):
            return
        saved_inputs.append(
            frozen_cpu(
                {
                    key: kwargs.get(key)
                    for key in ("encoder_hidden_states", "encoder_hidden_states_mask", "img_shapes", "img_mask")
                }
            )
        )
        shape = kwargs["img_shapes"][0][-1]
        n = shape[0] * shape[1] * shape[2]
        saved_inputs[-1]["image_latents"] = frozen_cpu(kwargs["hidden_states"][:, :-n])

    def step(model_output, timestep, sample, *args, **kwargs):
        nonlocal recorders, call_index, last_endpoint
        if recorders is None:
            if sample.shape[0] != len(extras):
                raise ValueError("Use rollout.n for ART groups; each request must generate one image.")
            recorders = [
                ARTTrajectoryRecorder(len(pipe.scheduler.timesteps), extra.get("art_step_indices"))
                if extra.get("art_capture_trajectory", True)
                else None
                for extra in extras
            ]
        index = pipe.scheduler.step_index
        if index is None:
            index = pipe.scheduler.begin_index
        if index is None:
            index = pipe.scheduler.index_for_timestep(timestep)
        sigma = pipe.scheduler.sigmas[index]
        dt = pipe.scheduler.sigmas[index + 1] - sigma
        for row, recorder in enumerate(recorders):
            if recorder is not None:
                recorder.record(call_index, sample[row], model_output[row], timestep, sigma, dt)
        result = original_step(model_output, timestep, sample, *args, **kwargs)
        last_endpoint = (result[0] if isinstance(result, tuple) else result.prev_sample).detach()
        call_index += 1
        return result

    hook = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    pipe.scheduler.step = step
    try:
        yield caches
        for row, recorder in enumerate(recorders):
            if recorder is None:
                continue
            conditions = []
            for inputs in saved_inputs:
                conditions.append(
                    {
                        key: [[tuple(shape) for shape in value[row]]]
                        if key == "img_shapes"
                        else value[row : row + 1].clone()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in inputs.items()
                    }
                )
            positive = conditions[0]
            reference = positive.pop("image_latents")
            negative = conditions[1] if cfg else None
            if negative is not None:
                negative.pop("image_latents")
            caches[row].update(
                recorder.finish(last_endpoint[row]),
                policy_version=policy_version,
                conditions={
                    "inputs": positive,
                    "negative_inputs": negative,
                    "image_latents": reference,
                    "true_cfg_scale": float(sampling_params[row].true_cfg_scale or 1.0),
                },
            )
    finally:
        pipe.scheduler.step = original_step
        hook.remove()
