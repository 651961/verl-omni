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

"""Qwen-Image 2.1 replay layouts for DiffusionART."""

from collections import defaultdict

import torch

from verl_omni.trainer.diffusion.art_math import art_scalar_covariance


@torch.no_grad()
def pack_art_caches(caches, rewards, uids, *, covariance_options=None):
    """Pad only for transport; preserve exact lengths/layouts for model replay.

    Covariance is computed before DP dispatch, over complete prompt groups.
    Different prompts may have different resolutions and reference images.
    """
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
        if steps != state.shape[0]:
            raise ValueError("ART samples must select the same number of training steps on all ranks.")
        for key in ("timesteps", "sigmas", "dt"):
            value = cache[key]
            if value.shape != (steps,):
                raise ValueError(f"Invalid ART {key} schedule.")
        if not ((cache["sigmas"] > 0) & (cache["sigmas"] <= 1)).all() or not (cache["dt"] < 0).all():
            raise ValueError("ART requires positive sigmas and decreasing ODE steps.")
        conditions = cache["conditions"]
        reference = conditions["image_latents"]
        if reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[-1] != state.shape[-1]:
            raise ValueError("ART reference latents must have shape [1, reference_tokens, channels].")
        for key in ("inputs", "negative_inputs"):
            cond = conditions.get(key)
            if cond is None:
                if key == "inputs":
                    raise ValueError("ART requires positive condition inputs.")
                continue
            emb = cond["encoder_hidden_states"]
            mask = cond["img_mask"]
            shapes = cond["img_shapes"]
            if emb.ndim != 3 or emb.shape[0] != 1:
                raise ValueError("ART condition embeddings must have batch size one.")
            n = state.shape[1]
            if n % 4 or mask.shape != (1, emb.shape[1] + n // 4):
                raise ValueError("ART joint image mask must span VLM embeddings and target slots.")
            if len(shapes) != 1 or not shapes[0] or any(len(s) != 3 or min(s) < 1 for s in shapes[0]):
                raise ValueError("ART requires valid per-image Qwen21 latent shapes.")
            counts = [int(s[0] * s[1] * s[2]) for s in shapes[0]]
            if counts[-1] != n or sum(counts[:-1]) != reference.shape[1]:
                raise ValueError("ART image layout does not match target/reference latent counts.")
            if int(mask.bool().sum()) * 4 != sum(counts) or not mask[:, -n // 4 :].bool().all():
                raise ValueError("ART image slots do not match latent layout.")
            embed_mask = cond.get("encoder_hidden_states_mask")
            if embed_mask is not None and embed_mask.shape != emb.shape[:2]:
                raise ValueError("ART encoder mask must align with the VLM embedding sequence.")
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

    def pad_stack(values):
        values = [v.detach().cpu() for v in values]
        rank = values[0].ndim
        if any(v.ndim != rank or v.dtype != values[0].dtype for v in values):
            raise ValueError("ART transport fields must have consistent rank and dtype.")
        shape = tuple(max(v.shape[axis] for v in values) for axis in range(rank))
        out = values[0].new_zeros((len(values), *shape))
        for i, value in enumerate(values):
            out[(i, *(slice(0, size) for size in value.shape))] = value
        return out

    fields = {
        "art_states": pad_stack([c["states"].float() for c in caches]),
        "art_old_velocities": pad_stack([c["old_velocities"].float() for c in caches]),
        "art_directions": pad_stack(directions),
        "art_magnitudes": magnitudes,
        "art_token_count": torch.tensor([c["states"].shape[1] for c in caches]),
    }
    for key in ("timesteps", "sigmas", "dt"):
        fields[f"art_{key}"] = torch.stack([c[key].detach().float().cpu() for c in caches])
    for prefix in ("", "negative_"):
        conditions = [c["conditions"].get(prefix + "inputs") for c in caches]
        prototype = caches[0]["conditions"]["inputs"]["encoder_hidden_states"]
        width = prototype.shape[-1]
        embeds, masks, image_masks, lengths, mask_lengths = [], [], [], [], []
        for cond in conditions:
            emb = cond["encoder_hidden_states"].squeeze(0) if cond else prototype.new_zeros(0, width)
            mask = cond.get("encoder_hidden_states_mask") if cond else None
            mask = mask.squeeze(0).bool() if mask is not None else torch.ones(len(emb), dtype=torch.bool)
            image_mask = cond["img_mask"].reshape(-1).bool() if cond else torch.zeros(0, dtype=torch.bool)
            embeds.append(emb)
            masks.append(mask)
            image_masks.append(image_mask)
            lengths.append(len(emb))
            mask_lengths.append(len(image_mask))
        name = "art_" + prefix
        fields[name + "embeds"] = pad_stack(embeds)
        fields[name + "embed_mask"] = pad_stack(masks)
        fields[name + "img_mask"] = pad_stack(image_masks)
        fields[name + "embed_count"] = torch.tensor(lengths)
        fields[name + "img_mask_count"] = torch.tensor(mask_lengths)
    shapes = [torch.tensor(c["conditions"]["inputs"]["img_shapes"][0], dtype=torch.long) for c in caches]
    fields["art_img_shapes"] = pad_stack(shapes)
    fields["art_image_count"] = torch.tensor([len(s) for s in shapes])
    condition_latents = [c["conditions"]["image_latents"].squeeze(0) for c in caches]
    fields["art_image_latents"] = pad_stack(condition_latents)
    fields["art_image_token_count"] = torch.tensor([len(x) for x in condition_latents])
    scales = [float(c["conditions"].get("true_cfg_scale", 1.0)) for c in caches]
    cfg_modes = [scale > 1 for scale in scales]
    if len(set(cfg_modes)) != 1 or any(
        active != (c["conditions"].get("negative_inputs") is not None)
        for active, c in zip(cfg_modes, caches, strict=True)
    ):
        raise ValueError("All ART samples must use the same CFG forward count across FSDP ranks.")
    fields["art_cfg_scale"] = torch.tensor(scales)
    return fields


def trim_art_transport(batch):
    """Drop cross-sample transport padding before staging tensors on the GPU."""
    batch = batch.clone(recurse=False)
    target = int(batch["art_token_count"].max())
    for key in ("art_states", "art_old_velocities", "art_ref_predictions"):
        if key in batch:
            batch[key] = batch[key][:, :, :target]
    batch["art_directions"] = batch["art_directions"][:, :target]
    batch["art_image_latents"] = batch["art_image_latents"][:, : int(batch["art_image_token_count"].max())]
    batch["art_img_shapes"] = batch["art_img_shapes"][:, : int(batch["art_image_count"].max())]
    for prefix in ("", "negative_"):
        name = "art_" + prefix
        embeds = int(batch[name + "embed_count"].max())
        mask = int(batch[name + "img_mask_count"].max())
        for suffix in ("embeds", "embed_mask"):
            batch[name + suffix] = batch[name + suffix][:, :embeds]
        batch[name + "img_mask"] = batch[name + "img_mask"][:, :mask]
    return batch


def qwen21_art_inputs(batch, step):
    """Rebuild one original joint layout, without padded tokens or rollout KV cache."""
    if batch["art_states"].shape[0] != 1:
        raise ValueError("Qwen-Image 2.1 ART requires micro_batch_size_per_gpu=1 for ragged joint layouts.")
    n = int(batch["art_token_count"][0])
    state = batch["art_states"][:, step, :n]
    k = int(batch["art_image_token_count"][0])
    condition = batch["art_image_latents"][:, :k]
    shapes = batch["art_img_shapes"][0, : int(batch["art_image_count"][0])].tolist()
    hidden = torch.cat([condition.to(state), state], dim=1) if k else state
    # The official pipeline casts the timestep to the model input dtype before /1000.
    dtype = batch["art_embeds"].dtype
    common = {
        "hidden_states": hidden.to(dtype),
        "timestep": batch["art_timesteps"][:, step].to(dtype) / 1000,
        "img_shapes": [[tuple(x) for x in shapes]],
        "return_dict": False,
    }
    inputs = []
    for prefix in ("", "negative_"):
        name = "art_" + prefix
        length = int(batch[name + "embed_count"][0])
        if not length:
            inputs.append(None)
            continue
        mask_length = int(batch[name + "img_mask_count"][0])
        inputs.append(
            {
                **common,
                "encoder_hidden_states": batch[name + "embeds"][:, :length],
                "encoder_hidden_states_mask": batch[name + "embed_mask"][:, :length],
                "img_mask": batch[name + "img_mask"][:, :mask_length],
            }
        )
    return inputs[0], inputs[1], state
