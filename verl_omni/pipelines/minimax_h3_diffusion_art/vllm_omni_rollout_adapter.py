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

"""Native MiniMax H3 Euler rollout with request-local DiffusionART capture."""

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native_pipeline

from verl_omni.pipelines.diffusion_rollout_output import (
    contains_meta_tensor,
    install_rollout_postprocessor,
    with_batched_media_artifacts,
    with_rollout_data,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    ref2va_reference_image_short_edge,
    validate_h3_parallel_config,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.weight_sync import MiniMaxH3WeightSyncMixin
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

from .capture import H3ARTCapture

__all__ = ["MiniMaxH3DiffusionARTPipeline"]

install_rollout_postprocessor(native_pipeline, "get_minimax_h3_post_process_func")


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_art")
class MiniMaxH3DiffusionARTPipeline(MiniMaxH3WeightSyncMixin, MiniMaxH3Pipeline):
    """Retain native conditioning, Euler updates and decoding for FL2VA and Ref2VA."""

    supports_request_batch = False
    supports_step_execution = False
    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "video_preview": MediaSpec("video", "decoded", "TCHW"),
            "audio": MediaSpec("audio", "decoded", "CT", sample_rate=32000),
        }
    )

    def __init__(self, *, od_config: Any, prefix: str = ""):
        validate_h3_parallel_config(od_config.parallel_config)
        if od_config.cache_backend not in (None, "none"):
            raise ValueError("H3 ART requires cache_backend=none for native velocity replay.")
        self._reference_image_short_edge = validate_ref2va_reference_image_short_edge()
        super().__init__(od_config=od_config, prefix=prefix)
        self._published_policy_version = 0
        self._art_capture = None
        self._art_hook = None
        self._art_cache = None
        self._art_extra = {}

    def set_published_policy_version(self, version: int) -> None:
        """Commit the serving version only after its complete weight stream is loaded."""
        self._published_policy_version = int(version)

    def _build_denoise_inputs(self, **kwargs):
        if kwargs.get("task") not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("H3 ART supports t2va, fl2va and ref2va.")
        if kwargs.get("sampler", "euler") != "euler":
            raise ValueError("H3 ART requires the Euler sampler for frozen ODE velocity targets.")
        unsupported = ("refine", "init_latents", "locked_audio_rows", "video_edit_clean_rows", "audio_edit_clean_rows")
        if any(kwargs.get(key) is not None for key in unsupported):
            raise ValueError("H3 ART does not support refinement, latent editing or locked audio.")
        inputs = super()._build_denoise_inputs(**kwargs)
        if self._art_extra.get("art_capture_trajectory", True):
            if self._art_capture is not None:
                raise ValueError("H3 ART requires one denoising pass per request; disable continuation/refinement.")
            self._art_capture = H3ARTCapture(
                inputs, self._art_extra.get("art_step_indices"), self._published_policy_version
            )
            self._art_hook = self._transformer_for_task(kwargs["task"]).register_forward_hook(
                self._art_capture.record_forward, with_kwargs=True
            )
        return inputs

    def _unpack_denoised_rows(self, branch, video_rows, audio_rows, **kwargs):
        if self._art_capture is not None:
            self._art_cache = self._art_capture.finish(video_rows, audio_rows)
        return super()._unpack_denoised_rows(branch, video_rows, audio_rows, **kwargs)

    @torch.no_grad()
    def forward(self, request):
        """Attach the frozen native trajectory, cleaning request state even on failure."""
        if len(request.requests) != 1:
            raise ValueError("H3 ART requires one request at a time.")
        sampling = request.requests[0].sampling_params
        if int(sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("Use rollout.n for ART groups; each request must generate one video.")
        if sampling.frame_rate is not None and float(sampling.frame_rate) != 24.0:
            raise ValueError("H3 ART uses 24 fps; configure num_frames to change duration.")
        self._art_extra = dict(sampling.extra_args or {})
        short_edge = self._art_extra.get("reference_image_short_edge", self._reference_image_short_edge)
        try:
            with ref2va_reference_image_short_edge(short_edge):
                self._ensure_prompt_text(request)
                output = super().forward(request)
            if self._art_extra.get("art_capture_trajectory", True) and self._art_cache is None:
                raise RuntimeError("H3 ART rollout did not capture a trajectory.")
            # CPU-offloaded non-reply TP ranks can return meta placeholders.
            has_meta = contains_meta_tensor(output.output)
            result = with_rollout_data(
                output, rl={"art_cache": self._art_cache} if self._art_cache is not None else {}, to_cpu=not has_meta
            )
            if has_meta:
                return result
            video, audio = output.output
            return with_batched_media_artifacts(
                result,
                data={"video_preview": video, "audio": audio},
                specs={
                    "video_preview": MediaSpec("video", "decoded", "THWC", fps=sampling.resolved_frame_rate or 24.0),
                    "audio": MediaSpec("audio", "decoded", "CT", sample_rate=32000),
                },
                primary="video_preview",
                preview="video_preview",
                audio="audio",
                context=f"pipeline={type(self).__name__}, request_id={request.request_id}",
                requested=self._art_extra.get("requested_outputs"),
            )
        finally:
            if self._art_hook is not None:
                self._art_hook.remove()
            self._art_hook = self._art_capture = self._art_cache = None
            self._art_extra = {}
            self._h3_prompt_ids = None
