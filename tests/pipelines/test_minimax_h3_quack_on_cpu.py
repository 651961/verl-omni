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

"""H3 kernel installation preserves FSDP and checkpoint parameter identity."""

import ast
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

quack = pytest.importorskip("quack.rmsnorm")
h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "h3_quack_ops", ROOT / "verl_omni/pipelines/minimax_h3_diffusion_art/ops.py"
)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def tiny_model():
    return h3.MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=256,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=512,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        freq_dim=32,
        time_embed_hidden_dim=256,
        time_embed_dim=128,
        norm_eps=1e-5,
        qk_norm_eps=1e-6,
    )


def configure_adapter(module, monkeypatch):
    monkeypatch.setattr(h3, "_apply_rotary_emb", h3._apply_rotary_emb)
    # Exercise the actual pre-FSDP hook without importing rollout dependencies.
    path = ROOT / "verl_omni/pipelines/minimax_h3_diffusion_art/diffusers_training_adapter.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if getattr(node, "name", None) == "configure_trainable_params")
    method.decorator_list = []
    namespace = {"__name__": "h3_adapter_hook", "__package__": "verl_omni.pipelines.minimax_h3_diffusion_art"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    with patch.dict(sys.modules, {"verl_omni.pipelines.minimax_h3_diffusion_art.ops": ops}):
        namespace["configure_trainable_params"](None, module, None)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_adapter_preserves_parameters_and_checkpoint(device, monkeypatch):
    with torch.device(device):
        model = tiny_model().eval()
    model.transformer_blocks[0].attn.norm_k.weight.requires_grad_(False)
    model.transformer_blocks[0].attn.norm_q.train()
    parameters = dict(model.named_parameters())
    original = dict(model.named_modules())
    expected_norms = {name: child for name, child in original.items() if type(child) is torch.nn.RMSNorm}
    state_keys = set(model.state_dict())

    configure_adapter(model, monkeypatch)
    assert h3._apply_rotary_emb is ops.apply_quack_rotary_emb
    replacements = {name: child for name, child in model.named_modules() if type(child) is quack.QuackRMSNorm}
    assert set(replacements) == set(expected_norms)
    assert len(replacements) == 14
    assert set(model.state_dict()) == state_keys
    for name, param in model.named_parameters():
        assert param is parameters[name]
        assert param.device.type == device
    assert not model.transformer_blocks[0].attn.norm_k.weight.requires_grad
    for name, child in model.named_modules():
        if name in replacements:
            assert child.eps == expected_norms[name].eps
            assert child.training == expected_norms[name].training
        else:
            assert child is original[name]
    configure_adapter(model, monkeypatch)
    assert all(model.get_submodule(name) is child for name, child in replacements.items())
    if device == "cpu":
        restored = tiny_model()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.load_state_dict(restored.state_dict(), strict=True)


def test_rope_dispatch_preserves_partial_rotation_and_bf16_frequencies():
    x = torch.randn(1, 5, 2, 128, dtype=torch.bfloat16)
    angles = torch.randn(5, 48)
    cos, sin = (value.repeat(1, 2) for value in (angles.cos(), angles.sin()))
    with patch.object(ops, "apply_rotary_emb", return_value=x) as kernel:
        assert ops.apply_quack_rotary_emb(x, cos, sin) is x
    args, kwargs = kernel.call_args
    assert args[0] is x
    for actual, expected in zip(args[1:], (cos[:, :48], sin[:, :48]), strict=True):
        assert actual.dtype == x.dtype
        torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)
    assert kwargs == {"interleaved": False, "inplace": False}
