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
"""Selective checkpointing must preserve gradients and avoid selected recomputation."""

import copy
from unittest.mock import Mock

import pytest
import torch
from diffusers.models.modeling_utils import ModelMixin
from torch.utils._python_dispatch import TorchDispatchMode

from verl_omni.utils.diffusion_checkpoint import enable_gradient_checkpointing


class _CheckpointModel(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False
        self.block = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, inputs):
        if self.gradient_checkpointing:
            return self._gradient_checkpointing_func(self.block, inputs)
        return self.block(inputs)


class _CountMatmuls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mm.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))


def test_empty_selection_preserves_model_default():
    model = Mock()
    enable_gradient_checkpointing(model, [])
    model.enable_gradient_checkpointing.assert_called_once_with()


@pytest.mark.parametrize("name", ["aten.mm", "aten.mm.missing", "missing.op.default", "aten.mm.__name__"])
def test_invalid_operator_is_rejected_before_forward(name):
    with pytest.raises(ValueError, match="gradient_checkpointing_save_ops"):
        enable_gradient_checkpointing(_CheckpointModel(), [name])


@torch.enable_grad()
def test_selected_outputs_are_reused_with_matching_gradients_across_updates():
    torch.manual_seed(7)
    baseline = _CheckpointModel()
    selective = copy.deepcopy(baseline)
    enable_gradient_checkpointing(baseline, [])
    enable_gradient_checkpointing(selective, ["aten.mm.default"])
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in (baseline, selective)]

    for step in range(2):
        inputs = torch.randn(2, 4)
        results, counts = [], []
        for model, optimizer in zip((baseline, selective), optimizers, strict=True):
            optimizer.zero_grad()
            leaf = inputs.clone().requires_grad_()
            torch.manual_seed(100 + step)
            with _CountMatmuls() as counter:
                output = model(leaf)
                output.square().sum().backward()
            results.append((output.detach(), leaf.grad))
            counts.append(counter.calls)
            optimizer.step()

        # Two linear forwards are cached; backward matrix multiplications still run.
        assert counts[0] - counts[1] == 2
        for actual, expected in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(selective.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
