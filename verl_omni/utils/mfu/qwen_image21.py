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

"""Training FLOPs for Qwen-Image-2.1 ART replay (without rollout KV reuse)."""

from __future__ import annotations

from verl_omni.utils.mfu.diffusion_flops_counter import DiffusionModelFlops, register_diffusion_architecture


@register_diffusion_architecture("QwenImage21Pipeline")
class QwenImage21Flops(DiffusionModelFlops):
    """Single-stream QKV/out + SwiGLU, with interleaved block-causal attention.

    Counts useful forward/backward matmul FLOPs. Excludes checkpoint recompute,
    normalization, reward, VAE and text encoder work, as other training counters do.
    """

    def collect_meta(self, data):
        """Collect actual joint-layout sizes for each conditional forward."""
        result = {
            key: []
            for key in (
                "latent_seqlens",
                "prompt_seqlens",
                "projection_seqlens",
                "attention_pairs",
                "negative_prompt_seqlens",
                "negative_projection_seqlens",
                "negative_attention_pairs",
            )
        }
        for row, target in enumerate(data["art_token_count"].tolist()):
            refs = int(data["art_image_token_count"][row])
            result["latent_seqlens"].append(int(target) + refs)
            shapes = data["art_img_shapes"][row, : int(data["art_image_count"][row])].tolist()
            image_lengths = [int(t * h * w) for t, h, w in shapes]
            for prefix in ("", "negative_"):
                length = int(data["art_" + prefix + "embed_count"][row])
                text = length - refs // 4 if length else 0
                pairs = 0
                if length:
                    mask_length = int(data["art_" + prefix + "img_mask_count"][row])
                    mask = data["art_" + prefix + "img_mask"][row, :mask_length].tolist()
                    # The prompt alternates causal text runs and full image
                    # blocks. Each block sees all preceding blocks plus itself.
                    segments = []
                    image = 0
                    remaining = image_lengths[0]
                    for is_image in mask:
                        role = image if is_image else -1
                        width = 4 if is_image else 1
                        if segments and segments[-1][0] == role:
                            segments[-1][1] += width
                        else:
                            segments.append([role, width])
                        if is_image:
                            remaining -= width
                            if remaining == 0:
                                image += 1
                                remaining = image_lengths[image] if image < len(image_lengths) else 0
                    start = 0
                    for role, width in segments:
                        pairs += width * start + (width * (width + 1) // 2 if role == -1 else width * width)
                        start += width
                result[prefix + "prompt_seqlens"].append(text)
                result[prefix + "projection_seqlens"].append(length)
                result[prefix + "attention_pairs"].append(pairs)
        return result

    def estimate_flops(
        self,
        latent_seqlens,
        prompt_seqlens,
        delta_time,
        *,
        num_timesteps,
        num_forward_passes,
        projection_seqlens,
        attention_pairs,
        negative_prompt_seqlens,
        negative_projection_seqlens,
        negative_attention_pairs,
    ):
        """Estimate useful ART forward/backward TFLOP/s from the captured layouts."""
        cfg, dim = self.config, self.dim
        layers = int(cfg["num_layers"])
        # SwiGLU has three linears; the default expansion ratio is 3.
        block_params = (4 + 3 * int(cfg.get("mlp_ratio", 3))) * dim * dim
        img_params = int(cfg["in_channels"]) * dim
        txt_params = int(cfg["context_in_dim"]) * dim + dim * dim
        output_params = int(cfg.get("patch_size", 1)) ** 2 * int(cfg.get("out_channels") or cfg["in_channels"]) * dim
        total = 0.0
        for row, image in enumerate(latent_seqlens):
            for texts, projections, pairs in (
                (prompt_seqlens, projection_seqlens, attention_pairs),
                (negative_prompt_seqlens, negative_projection_seqlens, negative_attention_pairs),
            ):
                if not projections[row]:
                    continue
                joint = int(image) + int(texts[row])
                dense = 6.0 * (
                    (layers * block_params + output_params) * joint + img_params * image + txt_params * projections[row]
                )
                # Shared timestep MLP, shared modulation and output AdaNorm.
                time_rows = 2 if cfg.get("causal_condition", True) else 1
                dense += 6.0 * time_rows * (256 * dim + 6 * dim * dim)
                total += dense + 12.0 * layers * dim * pairs[row]
        # Conditional and optional unconditional passes were counted per row.
        return total * num_timesteps / delta_time / 1e12
