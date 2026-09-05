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
"""MiniMax H3 agent loop for token-id-native raw-text prompts."""

from collections.abc import Mapping
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.utils.dataset.minimax_h3_video import resolve_video_spec

from .common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY, messages_to_text

__all__ = ["MiniMaxH3DiffusionSingleTurnAgentLoop"]


@register("minimax_h3_diffusion_single_turn_agent")
class MiniMaxH3DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Tokenize H3 prompts verbatim and preserve raw reference media."""

    @staticmethod
    def _apply_row_shape(sampling_params: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply per-row output dimensions to an H3 reference-conditioned request."""
        if sampling_params.get("task") not in {"fl2va", "ref2va"}:
            return sampling_params

        spec = resolve_video_spec(kwargs)
        # Upstream duration aliases take precedence over num_frames. Disallow
        # them so a global setting cannot silently override a row's frame count.
        for source in (sampling_params, sampling_params.get("extra_args") or {}):
            if not isinstance(source, Mapping):
                raise ValueError("MiniMax H3 sampling extra_args must be a mapping.")
            target = source.get("target") or {}
            if not isinstance(target, Mapping):
                raise ValueError("MiniMax H3 sampling target must be a mapping.")
            if any(source.get(key) is not None for key in ("duration", "duration_seconds")) or (
                target.get("duration_seconds") is not None
            ):
                raise ValueError("MiniMax H3 per-row num_frames cannot be combined with a pipeline duration override.")
        # H3 reads fps rather than OmniDiffusionSamplingParams.frame_rate.
        return {**sampling_params, **spec, "fps": int(spec["frame_rate"])}

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        """Mark IDs so the H3 rollout can reject generic chat-template tokens."""
        sampling_params = self._apply_row_shape(sampling_params, kwargs)
        sampling_params = {**sampling_params, MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True}
        tensor_dims = {}
        if sampling_params.get("task") in {"fl2va", "ref2va"}:
            model_config = self.config.actor_rollout_ref.model
            adapter = DiffusionModelBase.get_class_by_name(
                "MiniMaxH3Pipeline", model_config.algorithm, getattr(model_config, "external_lib", None)
            )
            tensor_dims = adapter.ragged_rollout_tensor_dims()
            if not tensor_dims:
                raise NotImplementedError(
                    f"MiniMax H3 {model_config.algorithm} does not declare variable-size tensors."
                )
        output = await super().run(sampling_params, **kwargs)
        if tensor_dims:
            # Declare variable-size tensors explicitly, even for homogeneous worker
            # chunks: another worker may return a different resolution or duration.
            output.extra_fields["ragged_tensor_keys"] = ("responses", "audio", *tensor_dims)
            output.extra_fields["ragged_tensor_dims"] = tensor_dims
        return output

    async def process_multi_modal_info(self, messages: list[dict]) -> dict[str, list[Any]]:
        """Keep H3 reference paths and waveforms in their upstream input format."""
        media: dict[str, list[Any]] = {"images": [], "videos": [], "audios": []}
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                media_type = item.get("type")
                if media_type == "image":
                    media["images"].append(item["image"])
                elif media_type == "video":
                    video = item["video"]
                    start = item.get("start_time_seconds")
                    media["videos"].append(video if start is None else {"path": video, "start_time_seconds": start})
                elif media_type == "audio":
                    media["audios"].append(item["audio"])
        return {key: values for key, values in media.items() if values}

    async def _tokenize_raw_text(self, messages: list[dict]) -> list[int]:
        """Return raw H3 text IDs without applying a chat template."""
        text = messages_to_text(messages)
        if not text:
            raise ValueError("MiniMax H3 requires a non-empty text prompt.")
        prompt_length = self.rollout_config.prompt_length
        tokenized = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=prompt_length,
                add_special_tokens=False,
            )["input_ids"],
        )
        return normalize_token_ids(tokenized)

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        """Override verl's Continuous Token entry point with H3 raw-text IDs."""
        del tools, images, videos, audios
        return await self._tokenize_raw_text(messages)

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Keep the legacy entry point aligned with Continuous Token behavior."""
        del tools, images, videos, audios, mm_processor_kwargs, remove_system_prompt
        return await self._tokenize_raw_text(messages)
