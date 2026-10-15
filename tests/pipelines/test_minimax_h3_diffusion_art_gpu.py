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

"""Real FSDP2 coverage for H3 ART clocks and fractional rotary positions."""

import copy

import pytest
import torch
import torch.distributed as dist
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers import transformer_minimax_h3
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from verl_omni.pipelines.minimax_h3_diffusion_art.diffusers_training_adapter import MiniMaxH3DiffusionART


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 requires CUDA")
@torch.enable_grad()
def test_fsdp2_preserves_h3_coordinates_and_fp32_master_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(transformer_minimax_h3, "_apply_rotary_emb", transformer_minimax_h3._apply_rotary_emb)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        torch.manual_seed(17)
        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=128,
            hidden_size=32,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=64,
            text_dim=8,
            freq_dim=16,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=16,
        ).cuda()
        MiniMaxH3DiffusionART.configure_trainable_params(model, None)
        expected_model = copy.deepcopy(model)
        for parameter in expected_model.parameters():
            parameter.data = parameter.data.bfloat16()
        inputs = {
            "hidden_states": torch.randn(1, 2, 96, device="cuda"),
            "audio_hidden_states": torch.randn(1, 2, 32, device="cuda"),
            "encoder_hidden_states": torch.randn(1, 2, 8, device="cuda"),
            # Media positions continue after the text prefix and contain fractional offsets.
            "position_ids": torch.tensor(
                [[0, 0, 0], [1, 0, 0], [512, 0.3, 0.7], [513.67, 0.3, 0.7], [512, 0, 0], [513, 0, 0]],
                dtype=torch.float64,
                device="cuda",
            ),
            "token_tags": torch.tensor([1, 1, 0, 0, 2, 2], device="cuda"),
            "video_indices": torch.tensor([2, 3], device="cuda"),
            "audio_indices": torch.tensor([4, 5], device="cuda"),
            "text_indices": torch.tensor([0, 1], device="cuda"),
            "timestep": torch.tensor([0.4, 0.7, 0.999], device="cuda"),
            "timestep_indices": torch.tensor([0, 0, 2, 0, 1, 1], device="cuda"),
            "return_dict": False,
        }
        with torch.no_grad():
            expected = expected_model(**inputs)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=MiniMaxH3DiffusionART.cast_forward_inputs(),
        )
        for block in (*model.token_refiner.refiner_blocks, *model.transformer_blocks):
            fully_shard(block, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
        observed = {}

        def capture_input(name):
            def hook(module, args):
                observed[name] = args[0].detach().clone()

            return hook

        model.rope.register_forward_pre_hook(capture_input("positions"))
        model.time_proj.register_forward_pre_hook(capture_input("timesteps"))
        actual = model(**inputs)
        torch.testing.assert_close(observed["positions"], inputs["position_ids"], rtol=0, atol=0)
        torch.testing.assert_close(observed["timesteps"], inputs["timestep"], rtol=0, atol=0)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        sum(value.float().square().mean() for value in actual).backward()
        for parameter in model.parameters():
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad.to_local()).all()
        assert model.proj_out.weight.grad.to_local().abs().sum() > 0
        assert model.audio_proj_out.weight.grad.to_local().abs().sum() > 0
    finally:
        dist.destroy_process_group()
