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
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default):
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


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("patterns", [["block.0"], ["block.[02]", "block.0"]])
@torch.enable_grad()
def test_named_linear_selection_preserves_gradients_and_only_skips_selected_gemms(bias, patterns):
    torch.manual_seed(19)
    baseline = _CheckpointModel()
    baseline.block[0] = torch.nn.Linear(4, 4, bias=bias)
    baseline.block[2] = torch.nn.Identity()
    baseline.block[3] = torch.nn.Linear(4, 4, bias=bias)
    # Both projections have identical shapes; only the first is selected.
    selective = copy.deepcopy(baseline)
    enable_gradient_checkpointing(baseline, [])
    enable_gradient_checkpointing(selective, patterns)

    for _ in range(2):
        inputs = torch.randn(2, 4)
        outputs, leaves, counts = [], [], []
        for model in (baseline, selective):
            model.zero_grad(set_to_none=True)
            leaf = inputs.clone().requires_grad_()
            with _CountMatmuls() as counter:
                output = model(leaf)
                output.square().sum().backward()
            outputs.append(output.detach())
            leaves.append(leaf.grad)
            counts.append(counter.calls)
        assert counts[0] - counts[1] == 1
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        torch.testing.assert_close(leaves[1], leaves[0], rtol=0, atol=0)
        for actual, expected in zip(selective.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("pattern", ["missing.*", "block", "block.1"])
def test_linear_selection_rejects_unmatched_or_non_linear_modules(pattern):
    with pytest.raises(ValueError, match="matched no Linear modules"):
        enable_gradient_checkpointing(_CheckpointModel(), [pattern])


def test_reconfiguring_checkpointing_removes_old_linear_hooks():
    model = _CheckpointModel()
    enable_gradient_checkpointing(model, ["block.0"])
    assert len(model.block[0]._forward_pre_hooks) == 1
    enable_gradient_checkpointing(model, ["block.3"])
    assert not model.block[0]._forward_pre_hooks
    assert len(model.block[3]._forward_pre_hooks) == 1
    enable_gradient_checkpointing(model, [])
    assert not model.block[3]._forward_pre_hooks
    assert not model.block[3]._forward_hooks


@torch.enable_grad()
def test_checkpoint_early_stop_does_not_leak_selected_linear_scope():
    model = _CheckpointModel()
    model.block = torch.nn.Linear(4, 4, bias=False)
    model.other = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Sigmoid())
    enable_gradient_checkpointing(model, ["block"])
    # Replaying this checkpoint stops inside Linear before its GEMM runs.
    model(torch.randn(2, 4, requires_grad=True)).square().sum().backward()

    ordinary = copy.deepcopy(model.other)
    baseline = _CheckpointModel()
    baseline.block = ordinary
    enable_gradient_checkpointing(baseline, [])
    inputs = torch.randn(2, 4)
    counts = []
    for root, block in ((baseline, ordinary), (model, model.other)):
        with _CountMatmuls() as counter:
            output = root._gradient_checkpointing_func(block, inputs.clone().requires_grad_())
            output.square().sum().backward()
        counts.append(counter.calls)
    assert counts[0] == counts[1]


def test_named_linear_selection_rejects_regional_compilation():
    from types import SimpleNamespace

    from verl_omni.utils.diffusion_compile import _maybe_compile_repeated_blocks

    model = _CheckpointModel()
    enable_gradient_checkpointing(model, ["block.0"])
    with pytest.raises(ValueError, match="requires model.use_regional_compile=False"):
        _maybe_compile_repeated_blocks(model, SimpleNamespace(use_regional_compile=True), None)


@pytest.mark.parametrize("architecture", ["qwenimage21", "minimax_h3"])
@torch.enable_grad()
def test_real_diffusers_blocks_reuse_only_selected_projection_outputs(architecture):
    if architecture == "qwenimage21":
        source = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
        block = source.QwenImage21TransformerBlock(16, 2, 8)
        args = (torch.randn(1, 64),)
        patterns = ["block.img_mlp.out", "block.attn.to_out.0"]
    else:
        source = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
        block = source.MiniMaxH3TransformerBlock(16, 2, 8, 48, 16, 1e-6, 1e-6)
        args = (torch.randn(1, 16), torch.zeros(6, dtype=torch.long), None)
        patterns = ["block.ff.net.2", "block.attn.to_out.0"]

    class BlockModel(_CheckpointModel):
        def forward(self, inputs):
            return self._gradient_checkpointing_func(self.block, inputs, *args)

    baseline = BlockModel()
    baseline.block = block
    selective = copy.deepcopy(baseline)
    enable_gradient_checkpointing(baseline, [])
    enable_gradient_checkpointing(selective, patterns)

    for _ in range(2):
        inputs = torch.randn(1, 6, 16)
        results, counts = [], []
        for model in (baseline, selective):
            model.zero_grad(set_to_none=True)
            leaf = inputs.clone().requires_grad_()
            with _CountMatmuls() as counter:
                output = model(leaf)
                output.square().sum().backward()
            results.append((output.detach(), leaf.grad))
            counts.append(counter.calls)
        assert counts[0] - counts[1] == 2
        for actual, expected in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(selective.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
