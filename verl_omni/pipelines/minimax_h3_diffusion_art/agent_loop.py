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

"""Per-sample H3 video sizes for online DiffusionART."""

import operator

from verl.experimental.agent_loop.agent_loop import register

from verl_omni.pipelines.minimax_h3_diffusion_nft.agent_loop import MiniMaxH3DiffusionSingleTurnAgentLoop


def h3_sample_parameters(sampling_params: dict, extra_info: dict) -> dict:
    """Resolve required video geometry from one dataset row without mutation."""
    params = dict(sampling_params)
    for key in ("height", "width", "num_frames"):
        value = extra_info.get(key)
        if value is None:
            raise ValueError(f"H3 ART requires extra_info.{key} in every training and validation row.")
        try:
            if isinstance(value, bool):
                raise TypeError
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"H3 per-sample {key} must be a positive integer.") from exc
        if value <= 0 or (key in ("height", "width") and value % 32):
            raise ValueError(f"H3 {key} must be positive; height and width must be divisible by 32.")
        params[key] = value
    fps = extra_info.get("frame_rate", params.get("frame_rate", 24))
    if fps is not None and float(fps) != 24:
        raise ValueError("H3 ART uses 24 fps; set num_frames for variable duration.")
    params["frame_rate"] = 24.0
    return params


@register("minimax_h3_art_agent")
class MiniMaxH3ARTAgentLoop(MiniMaxH3DiffusionSingleTurnAgentLoop):
    """Preserve H3 raw media/token IDs and resolve sample-specific video dimensions."""

    async def run(self, sampling_params, **kwargs):
        """Keep each prompt group's generation geometry fixed across its samples."""
        params = h3_sample_parameters(sampling_params, kwargs.get("extra_info") or {})
        return await super().run(params, **kwargs)
