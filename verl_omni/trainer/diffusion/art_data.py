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

"""Architecture-independent ART group statistics and ragged transport."""

from collections import defaultdict

import torch

from verl_omni.trainer.diffusion.art_math import art_scalar_covariance


def pad_stack(values: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable tensor dimensions for transport, preserving dtype."""
    values = [v.detach().cpu() for v in values]
    rank = values[0].ndim
    if any(v.ndim != rank or v.dtype != values[0].dtype for v in values):
        raise ValueError("ART transport fields must have consistent rank and dtype.")
    shape = tuple(max(v.shape[axis] for v in values) for axis in range(rank))
    out = values[0].new_zeros((len(values), *shape))
    for i, value in enumerate(values):
        out[(i, *(slice(0, size) for size in value.shape))] = value
    return out


@torch.no_grad()
def pack_art_trajectories(caches, rewards, uids, *, covariance_options=None):
    """Pack [steps, tokens, channels] states and compute complete-group covariance."""
    if not caches or len(caches) != len(uids):
        raise ValueError("ART requires one cache and prompt uid per sample.")
    rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if rewards.numel() != len(caches):
        raise ValueError("ART requires one scalar reward per sample.")
    groups = defaultdict(list)
    steps = None
    for row, (cache, uid) in enumerate(zip(caches, uids, strict=True)):
        groups[str(uid)].append(row)
        for key in ("states", "old_velocities", "endpoints", "timesteps", "sigmas", "dt", "conditions"):
            if key not in cache:
                raise ValueError(f"ART rollout cache is missing {key}.")
        state = cache["states"]
        if state.ndim != 3 or not state.numel() or state.shape != cache["old_velocities"].shape:
            raise ValueError("ART states/old velocities must have shape [steps, tokens, channels].")
        if cache["endpoints"].shape != state.shape[1:]:
            raise ValueError("ART endpoint shape does not match trajectory states.")
        if steps is None:
            steps = state.shape[0]
        if state.shape[0] != steps:
            raise ValueError("ART samples must select the same number of training steps on all ranks.")
        if cache["timesteps"].shape != (steps,):
            raise ValueError("Invalid ART timesteps schedule.")
        sigmas, dt = cache["sigmas"], cache["dt"]
        if sigmas.ndim not in (1, 2) or sigmas.shape[0] != steps or sigmas.shape != dt.shape:
            raise ValueError("ART sigma and dt schedules must have matching step dimensions.")
        if not ((sigmas > 0) & (sigmas <= 1)).all() or not (dt < 0).all():
            raise ValueError("ART requires positive sigmas and decreasing ODE steps.")
    directions = [None] * len(caches)
    magnitudes = torch.empty(len(caches), dtype=torch.float32)
    for rows in groups.values():
        if len(rows) < 2:
            raise ValueError("ART covariance requires at least two samples per prompt.")
        endpoints = [caches[row]["endpoints"].detach().float().cpu() for row in rows]
        if any(x.shape != endpoints[0].shape for x in endpoints):
            raise ValueError("Samples of the same ART prompt must use the same target resolution.")
        result = art_scalar_covariance(
            torch.stack(endpoints),
            rewards[rows].cpu(),
            torch.zeros(len(rows), dtype=torch.long),
            1,
            **{"zero_signal_mode": "exact", **(covariance_options or {})},
        )
        for row in rows:
            directions[row] = result["group_directions"][0]
            magnitudes[row] = result["group_guidance_magnitudes"][0]

    fields = {
        "art_states": pad_stack([c["states"].float() for c in caches]),
        "art_old_velocities": pad_stack([c["old_velocities"].float() for c in caches]),
        "art_directions": pad_stack(directions),
        "art_magnitudes": magnitudes,
        "art_token_count": torch.tensor([c["states"].shape[1] for c in caches]),
    }
    for key in ("timesteps", "sigmas", "dt"):
        fields[f"art_{key}"] = torch.stack([c[key].detach().float().cpu() for c in caches])
    return fields
