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

"""Qwen-Image 2.1 tokenization and per-sample generation sizes."""

from verl.experimental.agent_loop.agent_loop import register

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.qwen_image21.media import load_rgba, messages_to_text
from verl_omni.pipelines.qwen_image21.prompt import prepare_prompt_inputs


@register("qwen_image21_agent")
class QwenImage21AgentLoop(DiffusionSingleTurnAgentLoop):
    """Send the processor's token IDs and vision inputs to the native pipeline."""

    async def run(self, sampling_params, **kwargs):
        """Resolve per-sample canvas sizes before invoking the single-turn loop."""
        sampling_params = dict(sampling_params)
        self._prompt_inputs = {}
        self._condition_media = None
        self._condition_media = await self.process_multi_modal_info(kwargs["raw_prompt"])
        scale = sampling_params.get("true_cfg_scale", 1.0)
        if scale is not None and scale > 1 and kwargs.get("raw_negative_prompt") is None:
            kwargs["raw_negative_prompt"] = [{"role": "user", "content": ""}]
        if scale is None or scale <= 1:
            kwargs.pop("raw_negative_prompt", None)
        info = kwargs.get("extra_info") or {}
        resolution = kwargs.get("output_resolution")
        if resolution is None:
            resolution = info.get("output_resolution")
        sample_resolution = resolution is not None
        if resolution is None:
            resolution = sampling_params.get("output_resolution")
        explicit_sizes = set()
        for key in ("height", "width"):
            value = kwargs.get(key)
            if value is None:
                value = info.get(key)
            if value is not None:
                if type(value) is not int and not hasattr(value, "__index__"):
                    raise ValueError(f"Qwen-Image 2.1 per-sample {key} must be an integer.")
                if int(value) < 32:
                    raise ValueError(f"Qwen-Image 2.1 per-sample {key} must be at least 32.")
                sampling_params[key] = int(value)
                explicit_sizes.add(key)
        if resolution is not None:
            if type(resolution) is not int and not hasattr(resolution, "__index__"):
                raise ValueError("Qwen-Image 2.1 output_resolution must be an integer.")
            if int(resolution) < 32:
                raise ValueError("Qwen-Image 2.1 output_resolution must be at least 32.")
            from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import calculate_dimensions

            images = self._condition_media
            aspect = images["images"][-1].width / images["images"][-1].height if images else 1.0
            width, height = calculate_dimensions(int(resolution) ** 2, aspect)
            if (sample_resolution and "height" not in explicit_sizes) or not sampling_params.get("height"):
                sampling_params["height"] = height
            if (sample_resolution and "width" not in explicit_sizes) or not sampling_params.get("width"):
                sampling_params["width"] = width
        return await super().run(sampling_params, **kwargs)

    def _assert_mm_supported(self, has_media):
        # The model-specific loop supports native RGBA condition images.
        return

    async def process_multi_modal_info(self, messages):
        """Load image conditions while preserving their alpha channel."""
        if self._condition_media is not None:
            return self._condition_media
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                continue
            for item in message["content"]:
                if item.get("type") in ("video", "audio"):
                    raise ValueError("Qwen-Image 2.1 accepts image conditions only.")
                if item.get("type") == "image":
                    images.append(await self.loop.run_in_executor(None, load_rgba, item.get("image", item)))
        return {"images": images} if images else {}

    async def ct_build_initial_tokens(self, messages, **kwargs):
        """Tokenize one condition and retain its processor tensors for rollout."""
        inputs = await self.loop.run_in_executor(
            None,
            lambda: prepare_prompt_inputs(
                self.processor,
                messages_to_text(messages),
                kwargs.get("images"),
                self.rollout_config.pipeline.max_sequence_length,
            ),
        )
        ids = inputs.pop("input_ids")[0].tolist()
        self._prompt_inputs["positive" if not self._prompt_inputs else "negative"] = inputs
        return ids

    def _get_mm_processor_kwargs(self, audios):
        return {"qwen21": self._prompt_inputs}
