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

"""Ragged H3 ART transport and reconstruction of native packed layouts."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion.art_data import pack_art_trajectories, pad_stack


@torch.no_grad()
def pack_h3_art_caches(
    caches: list[dict[str, Any]], rewards: torch.Tensor, uids: list[str], *, covariance_options: dict | None = None
) -> dict[str, torch.Tensor]:
    """Pack variable duration/resolution groups, retaining each sample's noisy anchors."""
    fields = pack_art_trajectories(caches, rewards, uids, covariance_options=covariance_options)
    conditions = [cache["conditions"] for cache in caches]
    layouts = {}
    for cache, cond, uid in zip(caches, conditions, uids, strict=True):
        if cache["sigmas"].shape != (cache["states"].shape[0], 2):
            raise ValueError("H3 ART requires both video and audio sigma schedules.")
        video_count = int(cond["video_mask"].sum())
        audio_count = int(cond["audio_mask"].sum())
        if cache["states"].shape[1:] != (video_count * 96 + audio_count * 32, 1):
            raise ValueError("H3 ART target dimensions do not match the captured layout.")
        if cond["video_condition"].shape != (int((~cond["video_mask"]).sum()), 96):
            raise ValueError("H3 ART video condition rows do not match the captured layout.")
        if cond["audio_condition"].shape != (int((~cond["audio_mask"]).sum()), 32):
            raise ValueError("H3 ART audio condition rows do not match the captured layout.")
        indices = torch.cat([cond[key] for key in ("video_indices", "audio_indices", "text_indices")])
        if not torch.equal(indices.sort().values, torch.arange(len(cond["position_ids"]))):
            raise ValueError("H3 ART packed indices must cover the unpadded sequence exactly once.")
        if len(cond["embeds"]) != len(cond["text_indices"]):
            raise ValueError("H3 ART embeddings must align with text indices.")
        # Equal element counts do not imply equal temporal/spatial layouts.
        previous = layouts.setdefault(str(uid), cond["position_ids"])
        if not torch.equal(previous, cond["position_ids"]):
            raise ValueError("Samples of one ART prompt must share the same video/audio layout.")
    for key in conditions[0]:
        fields["art_h3_" + key] = pad_stack([cond[key] for cond in conditions])
        fields["art_h3_" + key + "_count"] = torch.tensor([len(cond[key]) for cond in conditions])
    fields["art_h3_target_counts"] = torch.tensor(
        [[int(cond["video_mask"].sum()), int(cond["audio_mask"].sum())] for cond in conditions]
    )
    return fields


def trim_h3_art_transport(batch: TensorDict) -> TensorDict:
    """Remove transport padding on CPU before staging a single sample to the actor."""
    if batch["art_states"].shape[0] != 1:
        raise ValueError("H3 ART requires ppo_micro_batch_size_per_gpu=1 for variable video layouts.")
    batch = batch.clone(recurse=False)
    tokens = int(batch["art_token_count"][0])
    for key in ("art_states", "art_old_velocities", "art_ref_predictions"):
        if key in batch:
            batch[key] = batch[key][:, :, :tokens]
    batch["art_directions"] = batch["art_directions"][:, :tokens]
    for key in list(batch.keys()):
        if key.startswith("art_h3_") and key + "_count" in batch:
            batch[key] = batch[key][:, : int(batch[key + "_count"][0])]
    return batch


def h3_art_inputs(batch: TensorDict, step: int) -> tuple[dict[str, Any], None, torch.Tensor]:
    """Build Diffusers kwargs from captured native positions, tags and modality clocks."""
    counts = batch["art_h3_target_counts"][0].tolist()
    video_count, audio_count = counts
    state = batch["art_states"][:, step, : video_count * 96 + audio_count * 32]

    def field(key):
        name = "art_h3_" + key
        return batch[name][0, : int(batch[name + "_count"][0])]

    video_mask, audio_mask = field("video_mask").bool(), field("audio_mask").bool()
    video = state.new_empty((len(video_mask), 96))
    audio = state.new_empty((len(audio_mask), 32))
    video[video_mask] = state[0, : video_count * 96].reshape(video_count, 96)
    audio[audio_mask] = state[0, video_count * 96 :].reshape(audio_count, 32)
    video[~video_mask] = field("video_condition").to(video)
    audio[~audio_mask] = field("audio_condition").to(audio)
    video_indices, audio_indices = field("video_indices").long(), field("audio_indices").long()
    video_t, audio_t = 1 - batch["art_sigmas"][0, step]
    row_timesteps = video_t.expand(len(field("position_ids"))).clone()
    row_timesteps[video_indices[~video_mask]] = video_t.clamp_min(0.999)
    row_timesteps[audio_indices[audio_mask]] = audio_t
    row_timesteps[audio_indices[~audio_mask]] = 1.0
    unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    return (
        {
            "hidden_states": video.unsqueeze(0),
            "audio_hidden_states": audio.unsqueeze(0),
            "encoder_hidden_states": field("embeds").unsqueeze(0),
            "position_ids": field("position_ids"),
            "token_tags": field("token_tags"),
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": field("text_indices"),
            "timestep": unique,
            "timestep_indices": inverse,
            "return_dict": False,
        },
        None,
        state,
    )


def h3_art_coefficients(batch: TensorDict, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the two captured clocks over scalar target features, excluding padding."""
    video_count, audio_count = batch["art_h3_target_counts"][0].tolist()
    return tuple(
        torch.cat(
            (
                batch[key][:, step, 0:1].expand(-1, video_count * 96),
                batch[key][:, step, 1:2].expand(-1, audio_count * 32),
            ),
            dim=1,
        ).unsqueeze(-1)
        for key in ("art_sigmas", "art_dt")
    )
