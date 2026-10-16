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

"""Fused RMSNorm and partial RoPE for MiniMax H3 ART training."""

from quack.rmsnorm import QuackRMSNorm
from quack.rotary import apply_rotary_emb
from torch import Tensor, nn


def apply_quack_rms_norms(module: nn.Module) -> None:
    """Install Quack before FSDP while retaining H3's parameters and epsilon."""
    for name, child in list(module.named_modules()):
        if type(child) is nn.RMSNorm:
            replacement = QuackRMSNorm(
                child.normalized_shape[0], eps=child.eps, device=child.weight.device, dtype=child.weight.dtype
            )
            replacement.weight = child.weight
            module.set_submodule(name, replacement.train(child.training))


def apply_quack_rotary_emb(hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate H3's leading channels with its native coefficient precision."""
    half_rotary_dim = cos.shape[-1] // 2
    return apply_rotary_emb(
        hidden_states,
        cos[:, :half_rotary_dim].to(hidden_states.dtype),
        sin[:, :half_rotary_dim].to(hidden_states.dtype),
        interleaved=False,
        inplace=False,
    )


def apply_quack_rope() -> None:
    """Replace the H3-only rotary helper in this actor/reference worker."""
    from diffusers.models.transformers import transformer_minimax_h3

    transformer_minimax_h3._apply_rotary_emb = apply_quack_rotary_emb
