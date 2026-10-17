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

"""CUDA numerical and checkpoint tests for H3 actor kernels."""

import copy

import pytest
import torch
from test_minimax_h3_quack_on_cpu import configure_adapter, h3, ops, tiny_model

from verl_omni.utils.diffusion_checkpoint import enable_gradient_checkpointing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Quack kernels")


def relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("dim", [128, 5376])
def test_real_h3_norm_dimensions_forward_backward(dim):
    torch.manual_seed(23)
    native = torch.nn.Sequential(torch.nn.RMSNorm(dim, eps=1e-5)).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        native[0].weight.copy_(torch.randn_like(native[0].weight) * 0.1 + 1)
    candidate = copy.deepcopy(native)
    ops.apply_quack_rms_norms(candidate)
    with torch.enable_grad():
        shape = (1, 16384, 56, dim) if dim == 128 else (1, 16384, dim)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)
        original_dy = dy.clone()
        expected = native(x)
        expected_grads = torch.autograd.grad(expected, (x, native[0].weight), dy)
        actual = candidate(x)
        grads = torch.autograd.grad(actual, (x, candidate[0].weight), dy)
    assert relative_error(actual, expected) < 5e-4
    assert relative_error(grads[0], expected_grads[0]) < 5e-4
    # Native BF16 weight-gradient reductions lose precision across 56 heads.
    assert relative_error(grads[1], expected_grads[1]) < (5e-3 if dim == 128 else 5e-4)
    torch.testing.assert_close(dy, original_dy, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True, "selective_linear"])
def test_h3_video_audio_forward_backward_with_checkpointing(checkpointing, monkeypatch):
    torch.manual_seed(654)
    native = tiny_model().to(device="cuda", dtype=torch.bfloat16)
    native.rope.inv_freq = native.rope.inv_freq.float()
    candidate = copy.deepcopy(native)
    if checkpointing:
        native.enable_gradient_checkpointing()
        enable_gradient_checkpointing(
            candidate,
            ["transformer_blocks.*.ff.net.2", "transformer_blocks.*.attn.to_out.0"]
            if checkpointing == "selective_linear"
            else [],
        )
    inputs = {
        "hidden_states": torch.randn(1, 17, 16, device="cuda", dtype=torch.bfloat16),
        "audio_hidden_states": torch.randn(1, 11, 8, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 7, 32, device="cuda", dtype=torch.bfloat16),
        "timestep": torch.tensor([0.999, 0.523, 0.0], device="cuda"),
        "timestep_indices": torch.tensor([2] * 7 + [0] * 17 + [1] * 11, device="cuda"),
        "token_tags": torch.tensor([1] * 7 + [0] * 17 + [2] * 11, device="cuda"),
        "position_ids": torch.rand(35, 3, device="cuda") * 533.667,
        "video_indices": torch.arange(7, 24, device="cuda"),
        "audio_indices": torch.arange(24, 35, device="cuda"),
        "text_indices": torch.arange(7, device="cuda"),
        "return_dict": False,
    }
    input_names = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    actual_inputs = dict(inputs)
    for name in input_names:
        inputs[name].requires_grad_()
        actual_inputs[name] = inputs[name].detach().clone().requires_grad_()
    with torch.enable_grad():
        expected = native(**inputs)
        dy = [torch.randn_like(output) for output in expected]
        torch.autograd.backward(expected, dy)
        # Finish native checkpoint recomputation before replacing its global helper.
        configure_adapter(candidate, monkeypatch)
        actual = candidate(**actual_inputs)
        torch.autograd.backward(actual, dy)
    for result, reference in zip(actual, expected, strict=True):
        assert relative_error(result, reference) < 4e-3
    for name in input_names:
        assert relative_error(actual_inputs[name].grad, inputs[name].grad) < 0.015
    expected_params = dict(native.named_parameters())
    for name, param in candidate.named_parameters():
        assert param.grad is not None, name
        # RoPE fusion changes BF16 rounding; indexed AdaLN reductions also vary across native repeats.
        assert relative_error(param.grad, expected_params[name].grad) < 0.02, name


def test_partial_rope_forward_backward_preserves_tail_and_inputs():
    torch.manual_seed(105)
    position_ids = torch.rand(16384, 3, device="cuda") * 533.667
    cos, sin = h3.MiniMaxH3RotaryPosEmbed().cuda()(position_ids)
    original_cos, original_sin = cos.clone(), sin.clone()
    with torch.enable_grad():
        x = torch.randn(1, 16384, 56, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn_like(x)
        original_x, original_dy = x.detach().clone(), dy.clone()
        expected = h3._apply_rotary_emb(x, cos, sin)
        expected_dx = torch.autograd.grad(expected, x, dy)[0]
        actual = ops.apply_quack_rotary_emb(x, cos, sin)
        actual_dx = torch.autograd.grad(actual, x, dy)[0]
    # Native BF16 products round separately; Quack fuses the FP32 multiply/add.
    assert relative_error(actual, expected) < 4e-3
    assert relative_error(actual_dx, expected_dx) < 4e-3
    for result, reference in (
        (actual[..., 96:], x[..., 96:]),
        (actual_dx[..., 96:], dy[..., 96:]),
        (x, original_x),
        (dy, original_dy),
        (cos, original_cos),
        (sin, original_sin),
    ):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
