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

"""Qwen21 kernel installation and checkpoint contracts without executing CUDA."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("liger_kernel")
pytest.importorskip("quack.rotary")
qwen = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen21_liger_ops", ROOT / "verl_omni/pipelines/qwen_image21/ops.py")
norms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norms)


def tiny_model():
    return qwen.QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        context_in_dim=8,
        axes_dims_rope=(2, 2, 4),
    )


def configure_adapter(module, monkeypatch):
    # Exercise the actual pre-FSDP hook without importing the serving stack.
    # Record the process-local helper so pytest restores it after this test.
    monkeypatch.setattr(qwen, "apply_rotary_emb_qwen", qwen.apply_rotary_emb_qwen)
    path = ROOT / "verl_omni/pipelines/qwen_image21_diffusion_art/diffusers_training_adapter.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if getattr(node, "name", None) == "configure_trainable_params")
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    aliases = {"verl_omni.pipelines.qwen_image21.ops": norms}
    for name in ("verl_omni", "verl_omni.pipelines", "verl_omni.pipelines.qwen_image21"):
        package = types.ModuleType(name)
        package.__path__ = []
        aliases[name] = package
    with patch.dict(sys.modules, aliases):
        namespace["configure_trainable_params"](None, module, None)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_adapter_preserves_parameters_layernorm_and_checkpoint_keys(device, monkeypatch):
    with torch.device(device):
        model = tiny_model().eval()
    model.transformer_blocks[0].attn.norm_k.weight.requires_grad_(False)
    parameters = dict(model.named_parameters())
    state_keys = set(model.state_dict())
    original_modules = dict(model.named_modules())

    original_rope = model.pos_embed
    configure_adapter(model, monkeypatch)
    assert qwen.apply_rotary_emb_qwen is norms.apply_quack_rotary_emb
    assert model.pos_embed is original_rope
    processors = [block.attn.processor for block in model.transformer_blocks]
    assert all(isinstance(processor, norms.QwenImage21FlashAttnProcessor) for processor in processors)

    replacements = {
        name: child for name, child in model.named_modules() if isinstance(child, norms.QwenImage21LigerRMSNorm)
    }
    assert set(replacements) == {
        "txt_in.text_norm",
        "transformer_blocks.0.attn.norm_q",
        "transformer_blocks.0.attn.norm_k",
        "transformer_blocks.1.attn.norm_q",
        "transformer_blocks.1.attn.norm_k",
    }
    assert all(not child.training for child in replacements.values())
    assert set(model.state_dict()) == state_keys
    for name, param in model.named_parameters():
        assert param is parameters[name]
        assert param.device.type == device
    assert not model.transformer_blocks[0].attn.norm_k.weight.requires_grad
    for name, child in model.named_modules():
        if name not in replacements:
            assert child is original_modules[name]
    configure_adapter(model, monkeypatch)
    assert all(model.get_submodule(name) is child for name, child in replacements.items())
    assert all(
        block.attn.processor is processor for block, processor in zip(model.transformer_blocks, processors, strict=True)
    )
    if device == "cpu":
        restored = tiny_model()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.load_state_dict(restored.state_dict(), strict=True)


@pytest.mark.parametrize("zero_centered,offset,mode", [(False, 0.0, "llama"), (True, 1.0, "gemma")])
def test_rmsnorm_dispatch_preserves_casting_and_does_not_overwrite_upstream_gradient(zero_centered, offset, mode):
    weight = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    module = norms.QwenImage21LigerRMSNorm(weight, 1e-6, zero_centered=zero_centered)
    x = torch.randn(1, 5, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    with patch.object(norms.LigerRMSNormFunction, "apply", return_value=x) as kernel:
        assert module(x) is x
    kernel.assert_called_once_with(x, weight, 1e-6, offset, mode, False, None)


def test_rope_dispatch_preserves_fp32_frequencies_and_adjacent_pairs():
    x = torch.randn(1, 5, 2, 128, dtype=torch.bfloat16)
    angles = torch.randn(5, 64, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    with patch.object(norms, "apply_rotary_emb", return_value=x) as kernel:
        assert norms.apply_quack_rotary_emb(x, freqs, use_real=False) is x
    args, kwargs = kernel.call_args
    assert args[0] is x
    for actual, expected in zip(args[1:], (freqs.real, freqs.imag), strict=True):
        assert actual.dtype == torch.float32
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert kwargs == {"interleaved": True, "inplace": False}
