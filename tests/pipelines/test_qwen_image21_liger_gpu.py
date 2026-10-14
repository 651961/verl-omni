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

"""CUDA numerical and checkpoint-recomputation tests for Qwen21 training kernels."""

import copy

import pytest
import torch
from test_qwen_image21_liger_on_cpu import configure_adapter, norms, qwen, tiny_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Liger kernels")


def relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("zero_centered,dim", [(False, 128), (True, 4096)])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_real_norm_shapes_forward_backward(zero_centered, dim, scale):
    torch.manual_seed(23)
    cls = qwen.QwenImage21ZeroCenterRMSNorm if zero_centered else qwen.RMSNorm
    native = cls(dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        native.weight.copy_(torch.randn_like(native.weight) * 0.1 + (0 if zero_centered else 1))
    candidate = norms.QwenImage21LigerRMSNorm(
        torch.nn.Parameter(native.weight.detach().clone()), 1e-6, zero_centered=zero_centered
    )
    x = (torch.randn(1, 257, dim, device="cuda", dtype=torch.bfloat16) * scale).requires_grad_()
    dy = torch.randn_like(x)
    original_dy = dy.clone()
    expected = native(x)
    expected_grads = torch.autograd.grad(expected, (x, native.weight), dy)
    actual = candidate(x)
    grads = torch.autograd.grad(actual, (x, candidate.weight), dy)
    # Fusion changes intermediate backward rounding; guard against errors beyond
    # the BF16 differences measured against the native model implementation.
    assert relative_error(actual, expected) < 5e-4
    assert relative_error(grads[0], expected_grads[0]) < (5e-4 if zero_centered else 5e-3)
    assert relative_error(grads[1], expected_grads[1]) < 5e-4
    torch.testing.assert_close(dy, original_dy, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_transformer_forward_backward_and_optimizer_step(checkpointing, padded, monkeypatch):
    torch.manual_seed(71)
    native = tiny_model().to(device="cuda", dtype=torch.bfloat16)
    candidate = copy.deepcopy(native)
    if checkpointing:
        native.enable_gradient_checkpointing()
        candidate.enable_gradient_checkpointing()
    inputs = {
        "hidden_states": torch.randn(1, 16, 4, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 8, 8, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, not padded, not padded]], device="cuda", dtype=torch.bool
        ),
        "timestep": torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
        "img_shapes": [[(1, 4, 4)]],
        "img_mask": torch.tensor([[False] * 8 + [True] * 4], device="cuda"),
        "return_dict": False,
    }
    expected = native(**inputs)[0]
    dy = torch.randn_like(expected)
    expected.backward(dy)
    # Finish native recomputation before installing the worker-local RoPE helper.
    configure_adapter(candidate, monkeypatch)
    if not padded:
        inputs["encoder_hidden_states_mask"] = None
    with pytest.MonkeyPatch.context() as spy:
        calls = []
        kernel = norms.apply_rotary_emb

        def tracked_kernel(*args, **kwargs):
            calls.append(args[0].shape)
            return kernel(*args, **kwargs)

        spy.setattr(norms, "apply_rotary_emb", tracked_kernel)
        actual = candidate(**inputs)[0]
        assert relative_error(actual, expected) < 0.01
        actual.backward(dy)
    assert len(calls) == len(candidate.transformer_blocks) * 2 * (2 if checkpointing else 1)
    expected_params = dict(native.named_parameters())
    for name, param in candidate.named_parameters():
        assert param.grad is not None, name
        assert relative_error(param.grad, expected_params[name].grad) < 0.03, name
    before = candidate.txt_in.text_norm.weight.detach().clone()
    optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(before, candidate.txt_in.text_norm.weight)
    native.load_state_dict(candidate.state_dict(), strict=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_quack_rope_forward_backward_with_text_and_multiple_images(dtype, strided):
    torch.manual_seed(104)
    # Text between image blocks exercises frame offsets and negative spatial coordinates.
    mask = torch.tensor([False] * 7 + [True] * 64 + [False] * 5 + [True] * 64 + [False] * 3, device="cuda")
    freqs = qwen.QwenImage21Rope(10000, [16, 56, 56])([(1, 8, 8), (1, 8, 8)], mask, mask.device)
    shape = (2, 32, len(mask), 128) if strided else (2, len(mask), 32, 128)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if strided:
        x = x.transpose(1, 2)
    x.requires_grad_()
    dy = torch.randn_like(x)
    original_x, original_dy = x.detach().clone(), dy.clone()
    expected = qwen.apply_rotary_emb_qwen(x, freqs, use_real=False)
    expected_dx = torch.autograd.grad(expected, x, dy)[0]
    actual = norms.apply_quack_rotary_emb(x, freqs, use_real=False)
    actual_dx = torch.autograd.grad(actual, x, dy)[0]
    for result, reference in ((actual, expected), (actual_dx, expected_dx), (x, original_x), (dy, original_dy)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_segmented_attention_uses_flash_forward_and_backward():
    torch.manual_seed(93)
    native = qwen.QwenImage21Attention(dim=256, heads=2, dim_head=128).to(device="cuda", dtype=torch.bfloat16)
    candidate = copy.deepcopy(native)
    norms.apply_flash_attention(candidate)
    segments = [(0, 64, True), (64, 192, False), (192, 208, True)]
    x = torch.randn(1, 464, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual_x = x.detach().clone().requires_grad_()
    valid = torch.ones(1, 464, device="cuda", dtype=torch.bool)
    expected = native(x, segments=segments, key_valid=valid)
    dy = torch.randn_like(expected)
    expected.backward(dy)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = candidate(actual_x, segments=segments)
        actual.backward(dy)
    events = {event.key for event in profile.key_averages()}
    assert "aten::_scaled_dot_product_flash_attention" in events
    assert "aten::_scaled_dot_product_flash_attention_backward" in events
    assert "aten::_scaled_dot_product_efficient_attention" not in events
    assert relative_error(actual, expected) < 0.005
    assert relative_error(actual_x.grad, x.grad) < 0.01
    for actual_param, expected_param in zip(candidate.parameters(), native.parameters(), strict=True):
        assert relative_error(actual_param.grad, expected_param.grad) < 0.01
