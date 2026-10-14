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

"""Check segmented visibility against an independent dense attention oracle."""

import copy

import pytest
import torch
from test_qwen_image21_liger_on_cpu import norms, qwen


def dense_attention(attn, x, segments, key_valid):
    query = attn.norm_q(attn.to_q(x).unflatten(-1, (attn.heads, -1))).transpose(1, 2)
    key = attn.norm_k(attn.to_k(x).unflatten(-1, (attn.heads, -1))).transpose(1, 2)
    value = attn.to_v(x).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
    length = x.shape[1]
    image_ids = torch.full((length,), -1)
    for index, (start, end, is_text) in enumerate(segments):
        if not is_text:
            image_ids[start:end] = index
    image_ids[segments[-1][1] if segments else 0 :] = len(segments)
    positions = torch.arange(length)
    allowed = (positions[:, None] >= positions[None, :]) | (
        (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    )
    if key_valid is not None:
        allowed = allowed & key_valid[:, None, :]
    scores = (query @ key.transpose(-1, -2)) / query.shape[-1] ** 0.5
    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    result = (probabilities @ value).transpose(1, 2).flatten(2)
    return attn.to_out[1](attn.to_out[0](result))


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(0, 3, True)],
        [(0, 3, False), (3, 6, True)],
        [(0, 2, True), (2, 5, False), (5, 7, False), (7, 10, True)],
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_segmented_forward_backward_matches_dense_block_causal_attention(segments, padded):
    torch.manual_seed(92)
    reference = qwen.QwenImage21Attention(dim=16, heads=2, dim_head=8)
    candidate = copy.deepcopy(reference)
    norms.apply_flash_attention(candidate)
    x = torch.randn(1, 14, 16, requires_grad=True)
    actual_x = x.detach().clone().requires_grad_()
    valid = torch.ones(1, 14, dtype=torch.bool) if padded else None
    if padded:
        valid[:, [1, 8]] = False
    expected = dense_attention(reference, x, segments, valid)
    actual = candidate(actual_x, segments=segments, key_valid=valid)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    dy = torch.randn_like(expected)
    expected.backward(dy)
    actual.backward(dy)
    torch.testing.assert_close(actual_x.grad, x.grad, rtol=3e-5, atol=3e-6)
    for actual_param, expected_param in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=3e-5, atol=3e-6)


def test_explicit_decode_mask_is_preserved():
    torch.manual_seed(28)
    native = qwen.QwenImage21Attention(dim=16, heads=2, dim_head=8)
    candidate = copy.deepcopy(native)
    norms.apply_flash_attention(candidate)
    x = torch.randn(1, 5, 16)
    mask = torch.tensor([[[[True, False, True, False, True]]]])
    torch.testing.assert_close(candidate(x, attention_mask=mask), native(x, attention_mask=mask))
