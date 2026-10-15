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

"""Fused norms, RoPE and segmented attention for Qwen-Image 2.1 training."""

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21AttnProcessor,
    _qwenimage21_prepare_qkv,
)
from diffusers.models.transformers.transformer_qwenimage21 import (
    apply_rotary_emb_qwen as _native_apply_rotary_emb_qwen,
)
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from quack.rotary import apply_rotary_emb
from torch import nn
from torch.nn.attention.bias import causal_lower_right
from torch.utils._python_dispatch import _disable_current_modes


class QwenImage21FlashAttnProcessor(QwenImage21AttnProcessor):
    """Keep block-causal visibility while allowing SDPA's FlashAttention kernel."""

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        # ART removes all-valid masks once on CPU. Preserve native handling for
        # actual padding, explicit masks and context-parallel attention.
        if key_valid is not None or attention_mask is not None or self._parallel_config is not None:
            return super().__call__(
                attn,
                hidden_states,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb,
                layer_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                cache_write_slice=cache_write_slice,
                segments=segments,
                key_valid=key_valid,
            )

        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        if segments is None:
            hidden_states = F.scaled_dot_product_attention(query, key, value)
        else:
            outputs = []
            for start, end, is_text in segments:
                # Text after an image sees every preceding key plus its own
                # causal prefix. Construct CausalBias outside SAC dispatch
                # modes; its Tensor subclass cannot be created under them.
                if is_text and start:
                    with _disable_current_modes():
                        bias = causal_lower_right(end - start, end)
                else:
                    bias = None
                outputs.append(
                    F.scaled_dot_product_attention(
                        query[:, :, start:end],
                        key[:, :, :end],
                        value[:, :, :end],
                        attn_mask=bias,
                        is_causal=is_text and start == 0,
                    )
                )
            prefix_len = segments[-1][1] if segments else 0
            outputs.append(F.scaled_dot_product_attention(query[:, :, prefix_len:], key, value))
            hidden_states = torch.cat(outputs, dim=2)
        hidden_states = hidden_states.transpose(1, 2)[:, :seq_len_q].flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


def apply_flash_attention(module: nn.Module) -> None:
    """Install segmented SDPA on native Qwen21 attention modules before FSDP."""
    for child in module.modules():
        if isinstance(child, QwenImage21Attention) and type(child.processor) is QwenImage21AttnProcessor:
            processor = QwenImage21FlashAttnProcessor()
            processor._attention_backend = child.processor._attention_backend
            processor._parallel_config = child.processor._parallel_config
            child.set_processor(processor)


def apply_quack_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    """Apply Qwen21's complex frequencies without changing their FP32 precision."""
    if use_real:
        return _native_apply_rotary_emb_qwen(x, freqs_cis, use_real, use_real_unbind_dim)
    return apply_rotary_emb(
        x,
        freqs_cis.real.contiguous(),
        freqs_cis.imag.contiguous(),
        interleaved=True,
        inplace=False,
    )


def apply_quack_rope() -> None:
    """Replace the Qwen21-only helper in this actor/reference worker process."""
    from diffusers.models.transformers import transformer_qwenimage21

    # Both native attention processors resolve this helper at call time. Keep
    # their masking/cache logic and the original three-axis frequency generator.
    transformer_qwenimage21.apply_rotary_emb_qwen = apply_quack_rotary_emb


class QwenImage21LigerRMSNorm(nn.Module):
    """Keep each Qwen21 RMSNorm's casting order and existing weight parameter."""

    def __init__(self, weight: nn.Parameter, eps: float, *, zero_centered: bool):
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.offset = 1.0 if zero_centered else 0.0
        self.casting_mode = "gemma" if zero_centered else "llama"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run Liger without overwriting the incoming backward gradient."""
        # Q/K rounds normalized activations before scaling; text scales in FP32.
        return LigerRMSNormFunction.apply(
            hidden_states, self.weight, self.eps, self.offset, self.casting_mode, False, None
        )


def apply_liger_rms_norms(module: nn.Module) -> None:
    """Replace Qwen21 RMSNorms before FSDP, retaining original parameters."""
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21ZeroCenterRMSNorm

    for name, child in list(module.named_modules()):
        if isinstance(child, QwenImage21ZeroCenterRMSNorm):
            replacement = QwenImage21LigerRMSNorm(child.weight, child.eps, zero_centered=True)
        elif isinstance(child, RMSNorm):
            replacement = QwenImage21LigerRMSNorm(child.weight, child.eps, zero_centered=False)
        else:
            continue
        module.set_submodule(name, replacement.train(child.training))
