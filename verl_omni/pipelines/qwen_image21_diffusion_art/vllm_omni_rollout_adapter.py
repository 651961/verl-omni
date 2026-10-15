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

"""DiffusionART output hooks on the native vLLM-Omni Qwen-Image 2.1 pipeline."""

import torch
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image21.common import QwenImage21TokenIdPromptMixin
from verl_omni.pipelines.qwen_image21_diffusion_art.capture import capture_art_rollout
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.trainer.diffusion.art_capture import ARTTrajectoryRecorder, frozen_cpu

__all__ = ["QwenImage21DiffusionARTPipeline"]


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="diffusion_art")
class QwenImage21DiffusionARTPipeline(QwenImage21TokenIdPromptMixin, QwenImage21Pipeline):
    """Keep native batching, KV, CFG, decoding and weight loading."""

    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._published_policy_version = 0

    def set_published_policy_version(self, version):
        # Native decode graphs retain request-prefix buffers across execution.
        """Commit a serving version after its complete weight stream is applied."""
        self.release_captured_graphs()
        self._published_policy_version = int(version)

    @torch.no_grad()
    def forward(self, req):
        """Attach selected ODE states and conditions to native full-forward outputs."""
        if not any((p.extra_args or {}).get("art_capture_trajectory", True) for p in req.sampling_params_list):
            return [with_rollout_data(output, to_cpu=True) for output in super().forward(req)]
        with capture_art_rollout(self, req.sampling_params_list, self._published_policy_version) as caches:
            outputs = super().forward(req)
        return [
            with_rollout_data(output, rl={"art_cache": cache} if cache else {}, to_cpu=True)
            for output, cache in zip(outputs, caches, strict=True)
        ]

    def prepare_encode(self, state, **kwargs):
        """Initialize a request-local recorder and freeze its replay conditions."""
        if state.sampling.generator is None and state.sampling.seed is not None:
            state.sampling.generator = torch.Generator(device=self.device).manual_seed(state.sampling.seed)
        state = super().prepare_encode(state, **kwargs)
        extra = state.sampling.extra_args or {}
        if extra.get("art_capture_trajectory", True):
            if state.latents.shape[0] != 1:
                raise ValueError("Use rollout.n for ART groups; each request must generate one image.")
            state.extra["art_recorder"] = ARTTrajectoryRecorder(len(state.timesteps), extra.get("art_step_indices"))
            state.extra["art_policy_version"] = self._published_policy_version
            conditions = []
            for embeds, mask, image_mask in (
                (state.prompt_embeds, state.prompt_embeds_mask, state.extra["image_pad_mask"]),
                (
                    state.negative_prompt_embeds,
                    state.negative_prompt_embeds_mask,
                    state.extra["negative_image_pad_mask"],
                ),
            ):
                conditions.append(
                    None
                    if embeds is None
                    else frozen_cpu(
                        {
                            "encoder_hidden_states": embeds,
                            "encoder_hidden_states_mask": mask,
                            "img_shapes": state.img_shapes,
                            "img_mask": self._append_target_slots(image_mask, state.latents.shape[1]),
                        }
                    )
                )
            reference = state.extra["image_latents"]
            state.extra["art_conditions"] = {
                "inputs": conditions[0],
                "negative_inputs": conditions[1],
                "image_latents": frozen_cpu(reference if reference is not None else state.latents[:, :0]),
                "true_cfg_scale": float(state.sampling.true_cfg_scale or 1.0),
            }
        return state

    def step_scheduler(self, state, noise_pred, **kwargs):
        """Record the exact state and velocity consumed by the native scheduler."""
        recorder = state.extra.get("art_recorder")
        if recorder is not None:
            index = state.step_index
            sigma = state.scheduler.sigmas[index]
            recorder.record(
                index,
                state.latents[0],
                noise_pred[0],
                state.current_timestep,
                sigma,
                state.scheduler.sigmas[index + 1] - sigma,
            )
        super().step_scheduler(state, noise_pred, **kwargs)

    def post_decode(self, state, **kwargs):
        """Attach the completed request's ART replay cache to its decoded output."""
        output = super().post_decode(state, **kwargs)
        recorder = state.extra.pop("art_recorder", None)
        if recorder is None:
            return with_rollout_data(output, to_cpu=True)
        cache = {
            **recorder.finish(state.latents[0]),
            "conditions": state.extra.pop("art_conditions"),
            "height": state.extra["height"],
            "width": state.extra["width"],
            "policy_version": state.extra.pop("art_policy_version"),
        }
        return with_rollout_data(output, rl={"art_cache": cache}, to_cpu=True)
