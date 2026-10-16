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

"""Capture native H3 Euler inputs without replacing its denoising loop."""

from typing import Any

import torch

from verl_omni.trainer.diffusion.art_capture import ARTTrajectoryRecorder, frozen_cpu


def flatten_targets(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Return [video_elements + audio_elements, 1] without channel padding."""
    return torch.cat((video.reshape(-1), audio.reshape(-1))).unsqueeze(-1)


class H3ARTCapture:
    """Own one request's dual schedule, frozen conditions and selected ODE states."""

    def __init__(self, inputs: dict[str, Any], step_indices: list[int] | None, policy_version: int):
        self.branch = inputs["branch"]
        self.sigmas = torch.stack(
            [torch.as_tensor(inputs[key], dtype=torch.float32) for key in ("sigmas_video", "sigmas_audio")], dim=1
        )
        if self.sigmas.ndim != 2 or self.sigmas.shape[0] < 2 or not (self.sigmas[:-1] > self.sigmas[1:]).all():
            raise ValueError("H3 ART requires strictly decreasing video/audio schedules.")
        if not torch.all(self.sigmas[0] == 1) or not torch.all(self.sigmas[-1] == 0):
            raise ValueError("H3 ART schedules must span 1 to 0.")
        self.recorder = ARTTrajectoryRecorder(len(self.sigmas) - 1, step_indices)
        self.policy_version = policy_version
        self.conditions = None
        self.step = 0

    def record_forward(self, module, args, kwargs, output) -> None:
        """Record target rows and the exact seed-dependent condition anchors."""
        branch = self.branch
        video = kwargs["x"][0, branch.img_pos_dev]
        audio = kwargs["audio_x"][0, branch.audio_pos_dev]
        video_mask, audio_mask = branch.update_mask_dev, branch.audio_update_mask_dev
        if self.conditions is None:
            self.conditions = frozen_cpu(
                {
                    "embeds": kwargs["prompt_embeds"],
                    "position_ids": branch.img_position_ids_dev[: branch.used_len],
                    "token_tags": branch.token_tags_dev[: branch.used_len],
                    "video_indices": branch.img_pos_dev,
                    "audio_indices": branch.audio_pos_dev,
                    "text_indices": branch.text_pos_dev,
                    "video_mask": video_mask,
                    "audio_mask": audio_mask,
                    "video_condition": video[~video_mask],
                    "audio_condition": audio[~audio_mask],
                }
            )
        if self.step >= len(self.sigmas) - 1:
            raise ValueError("H3 ART requires exactly one denoiser forward per Euler step; disable model caches.")
        if self.step not in self.recorder.selected:
            self.step += 1
            return
        velocity_video, velocity_audio = output
        sigma = self.sigmas[self.step]
        self.recorder.record(
            self.step,
            flatten_targets(video[video_mask], audio[audio_mask]),
            -flatten_targets(velocity_video.float()[video_mask], velocity_audio.float()[audio_mask]),
            sigma[0] * 1000,
            sigma,
            self.sigmas[self.step + 1] - sigma,
        )
        self.step += 1

    def finish(self, video: torch.Tensor, audio: torch.Tensor) -> dict[str, Any]:
        """Freeze final target rows after the native loop, excluding reference anchors."""
        if self.step != len(self.sigmas) - 1 or self.conditions is None:
            raise ValueError("H3 ART did not capture the complete native Euler loop.")
        return {
            **self.recorder.finish(
                flatten_targets(video[self.branch.update_mask_dev], audio[self.branch.audio_update_mask_dev])
            ),
            "conditions": self.conditions,
            "policy_version": self.policy_version,
        }
