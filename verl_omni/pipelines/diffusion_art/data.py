# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 DiffusionART contributors
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

from verl_omni.pipelines.diffusion_art.objective import art_scalar_covariance


def prompt_window_indices(uids, *, generator: torch.Generator | None = None) -> tuple[list[int], list[int]]:
    """Split each prompt's trajectories into two windows, retaining each row's full replay.

    Adapted from DiffusionART's prompt_disjoint_windows (58e2052). Row order is
    always randomized; a supplied generator makes the permutation reproducible.
    Zero-guidance and masked rows remain part of the update denominator.
    """
    if not len(uids):
        raise ValueError("ART windows require nonempty prompt uids.")
    order = torch.randperm(len(uids), generator=generator).tolist()
    members = defaultdict(list)
    for row in order:
        members[str(uids[row])].append(row)
    if any(len(rows) % 2 for rows in members.values()):
        raise ValueError("Each ART prompt group must have even K for two update windows.")
    assignments = [0] * len(uids)
    for rows in members.values():
        size = len(rows) // 2
        for position, row in enumerate(rows):
            assignments[row] = position // size
    return [row for row in order if assignments[row] == 0], [row for row in order if assignments[row] == 1]


def pad_stack(values: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable tensor dimensions for transport, preserving dtype."""
    values = [v.detach().cpu() for v in values]
    rank = values[0].ndim
    if any(v.ndim != rank or v.dtype != values[0].dtype for v in values):
        raise ValueError("ART transport fields must have consistent rank and dtype.")
    shape = tuple(max(v.shape[axis] for v in values) for axis in range(rank))
    padded = values[0].new_zeros((len(values), *shape))
    for i, value in enumerate(values):
        padded[(i, *(slice(0, size) for size in value.shape))] = value
    return padded


@torch.no_grad()
def pack_art_trajectories(caches, rewards, uids, *, covariance_options=None, feature_group_sizes=None):
    """Pack [steps, tokens, channels] states and compute complete-group covariance."""
    if not caches or len(caches) != len(uids):
        raise ValueError("ART requires one cache and prompt uid per sample.")
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    grouped = feature_group_sizes is not None
    if grouped:
        feature_group_sizes = torch.as_tensor(feature_group_sizes, dtype=torch.long).cpu()
        if feature_group_sizes.ndim != 2 or feature_group_sizes.shape[0] != len(caches):
            raise ValueError("ART feature group sizes must have shape [samples, groups].")
        if rewards.shape != feature_group_sizes.shape or not (feature_group_sizes > 0).all():
            raise ValueError("ART requires one reward and positive feature count per sample and group.")
    else:
        rewards = rewards.reshape(-1)
        if rewards.numel() != len(caches):
            raise ValueError("ART requires one scalar reward per sample.")
    if not torch.isfinite(rewards).all():
        raise ValueError("ART rewards must be finite.")
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
    magnitudes = torch.empty(rewards.shape, dtype=torch.float32)
    for rows in groups.values():
        if len(rows) < 2:
            raise ValueError("ART covariance requires at least two samples per prompt.")
        endpoints = [caches[row]["endpoints"].detach().float().cpu() for row in rows]
        if any(x.shape != endpoints[0].shape for x in endpoints):
            raise ValueError("Samples of the same ART prompt must use the same target resolution.")
        sizes = feature_group_sizes[rows] if grouped else torch.tensor([[endpoints[0].shape[0]]] * len(rows))
        if not torch.equal(sizes, sizes[0].expand_as(sizes)) or int(sizes[0].sum()) != endpoints[0].shape[0]:
            raise ValueError("ART feature groups must cover each prompt's target layout consistently.")
        group_directions, group_magnitudes = [], []
        offset = 0
        for group, size in enumerate(sizes[0].tolist()):
            result = art_scalar_covariance(
                torch.stack([endpoint[offset : offset + size] for endpoint in endpoints]),
                rewards[rows, group].cpu() if grouped else rewards[rows].cpu(),
                torch.zeros(len(rows), dtype=torch.long),
                1,
                **{"zero_signal_mode": "exact", **(covariance_options or {})},
            )
            group_directions.append(result["group_directions"][0])
            group_magnitudes.append(result["group_guidance_magnitudes"][0])
            offset += size
        for row in rows:
            directions[row] = torch.cat(group_directions)
            magnitudes[row] = torch.stack(group_magnitudes) if grouped else group_magnitudes[0]

    fields = {
        "art_states": pad_stack([c["states"].float() for c in caches]),
        "art_old_velocities": pad_stack([c["old_velocities"].float() for c in caches]),
        "art_directions": pad_stack(directions),
        "art_magnitudes": magnitudes,
        "art_token_count": torch.tensor([c["states"].shape[1] for c in caches]),
    }
    if grouped:
        fields["art_feature_group_sizes"] = feature_group_sizes
    for key in ("timesteps", "sigmas", "dt"):
        fields[f"art_{key}"] = torch.stack([c[key].detach().float().cpu() for c in caches])
    return fields


def trim_art_trajectories(batch):
    """Remove target padding from one CPU microbatch before device staging."""
    if batch["art_states"].shape[0] != 1:
        raise ValueError("ART requires ppo_micro_batch_size_per_gpu=1 for variable replay layouts.")
    batch = batch.clone(recurse=False)
    tokens = int(batch["art_token_count"][0])
    for key in ("art_states", "art_old_velocities", "art_ref_predictions"):
        if key in batch:
            batch[key] = batch[key][:, :, :tokens]
    batch["art_directions"] = batch["art_directions"][:, :tokens]
    return batch


def restore_shared_conditions(caches, uids):
    """Reconstruct omitted per-prompt conditions after a complete group is read."""
    shared_keys = ("conditions", "timesteps", "sigmas", "dt")
    groups = {}
    for uid, cache in zip(uids, caches, strict=True):
        if isinstance(cache, dict) and all(k in cache for k in shared_keys):
            groups.setdefault(str(uid), cache)
    for row, (uid, cache) in enumerate(zip(uids, caches, strict=True)):
        if not isinstance(cache, dict):
            raise ValueError("ART trajectory was lost during TransferQueue transport.")
        missing = [k for k in shared_keys if k not in cache]
        if missing:
            if str(uid) not in groups:
                raise ValueError("ART transport requires a complete prompt group with shared conditions.")
            caches[row] = {**cache, **{k: groups[str(uid)][k] for k in missing}}
    return caches
