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

import torch

from verl_omni.trainer.diffusion.art_data import pack_art_trajectories, pad_stack


@torch.no_grad()
def pack_art_caches(caches, rewards, uids, *, covariance_options=None):
    """Pad only for transport; preserve exact lengths/layouts for model replay.

    Covariance is computed before DP dispatch, over complete prompt groups.
    Different prompts may have different resolutions and reference images.
    """
    fields = pack_art_trajectories(caches, rewards, uids, covariance_options=covariance_options)
    for cache in caches:
        state = cache["states"]
        if cache["sigmas"].ndim != 1:
            raise ValueError("Qwen ART requires a scalar sigma per step.")
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
        batch[name + "embeds"] = batch[name + "embeds"][:, :embeds]
        valid = batch[name + "embed_mask"]
        if valid is not None:
            valid = valid[:, :embeds]
            if valid.device.type != "cpu":
                raise ValueError("ART transport masks must be checked on CPU before device staging.")
            # Do not let cross-sample transport padding disable FlashAttention.
            # Retain real rollout padding: its positions are part of the RoPE layout.
            batch[name + "embed_mask"] = None if valid.all() else valid
        batch[name + "img_mask"] = batch[name + "img_mask"][:, :mask]
    return batch


def qwen21_art_inputs(batch, step):
    """Rebuild the rollout layout, excluding transport padding and rollout KV cache."""
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
        valid = batch[name + "embed_mask"]
        inputs.append(
            {
                **common,
                "encoder_hidden_states": batch[name + "embeds"][:, :length],
                "encoder_hidden_states_mask": None if valid is None else valid[:, :length],
                "img_mask": batch[name + "img_mask"][:, :mask_length],
            }
        )
    return inputs[0], inputs[1], state
