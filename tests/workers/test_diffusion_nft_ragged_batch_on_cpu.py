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
"""Ragged NFT samples accumulate equal sample weights before one optimizer step."""

import pickle
from contextlib import nullcontext
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import BaseEngine

import verl_omni.workers.engine.fsdp.diffusers_impl as engine_module
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import AUDIO_ROW_WIDTH, VIDEO_ROW_WIDTH, serialize_ref_blocks
from verl_omni.utils.ragged_tensors import unpack_ragged_tensors
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding


def test_ragged_batch_updates_once_with_equal_sample_weight(monkeypatch):
    monkeypatch.setattr(engine_module, "get_device_id", lambda: "cpu")
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    engine.is_mp_src_rank_with_outputs = lambda: True
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    updates, forwards = [], []
    engine.optimizer_zero_grad = optimizer.zero_grad

    def optimizer_step():
        updates.append(parameter.grad.item())
        optimizer.step()
        return parameter.grad.item()

    engine.optimizer_step = optimizer_step

    def forward_step(self, micro_batch, loss_function, forward_only, step):
        values = micro_batch["latents_clean"]
        assert not values.is_nested and values.shape[0] == 1
        forwards.append((values.shape[-1], parameter.item()))
        divisor = tu.get_non_tensor_data(micro_batch, "gradient_accumulation_steps", default=None)
        loss = (values * parameter).square().mean() / divisor
        return loss, {"model_output": {"prediction": values * parameter}, "loss": loss.item(), "metrics": {}}

    engine.forward_step = MethodType(forward_step, engine)
    data = tu.get_tensordict(
        tensor_dict={
            "latents_clean": torch.nested.as_nested_tensor([torch.ones(2), torch.full((6,), 3.0)], layout=torch.jagged),
            "train_timesteps": torch.tensor([[500.0, 250.0], [500.0, 250.0]]),
        },
        non_tensor_dict={"micro_batch_size_per_gpu": 2},
    )
    result = BaseEngine.train_batch(engine, data, loss_function=None)

    # mean(sample losses) = (p^2 + 9p^2)/2, so dL/dp = 10 at p=1.
    # Averaging padded tokens or updating after each sample gives a different result.
    assert updates == pytest.approx([10.0])
    assert parameter.item() == pytest.approx(0.9)
    assert forwards == [(2, 1.0), (2, 1.0), (6, 1.0), (6, 1.0)]
    assert result["model_output"] == {}


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_h3_mixed_shapes_reach_adapter_and_nft_loss_with_empty_audio_conditions(monkeypatch, task):
    monkeypatch.setattr(engine_module, "get_device_id", lambda: "cpu")
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.get_data_parallel_group = lambda: None
    engine.is_mp_src_rank_with_outputs = lambda: True
    engine.use_adapter = lambda _: nullcontext()
    engine.disable_adapter = nullcontext
    engine._set_adapter = lambda _: None
    engine.model_config = SimpleNamespace(
        architecture="MiniMaxH3Pipeline", algorithm="diffusion_nft", external_lib=None
    )
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    updates, shapes = [], []
    engine.optimizer_zero_grad = optimizer.zero_grad

    def optimizer_step():
        updates.append(parameter.grad.item())
        optimizer.step()
        return parameter.grad.item()

    engine.optimizer_step = optimizer_step

    def transformer(**kwargs):
        video, audio = kwargs["hidden_states"], kwargs["audio_hidden_states"]
        shapes.append((video.shape[1], audio.shape[1]))
        return video * parameter, audio * parameter

    engine.module = MagicMock(side_effect=transformer)
    engine.module.config = None
    lengths = [4 * VIDEO_ROW_WIDTH + 6 * AUDIO_ROW_WIDTH, 16 * VIDEO_ROW_WIDTH + 10 * AUDIO_ROW_WIDTH]
    condition_counts = [4, 8] if task == "fl2va" else [4, 4]
    data = tu.get_tensordict(
        tensor_dict={
            "latents_clean": torch.nested.as_nested_tensor([torch.ones(n) for n in lengths], layout=torch.jagged),
            "latent_meta": torch.tensor([[4, 6, 1, 4, 4, 3], [16, 10, 2, 4, 8, 5]]),
            "train_timesteps": torch.full((2, 1), 500.0),
            "reward_prob": torch.full((2, 1), 0.8),
            "prompt_embeds": torch.ones(2, 3, 8),
            "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.bool),
            "condition_video_rows": torch.zeros(2, 8, VIDEO_ROW_WIDTH),
            "condition_video_rows_mask": torch.arange(8).unsqueeze(0) < torch.tensor(condition_counts).unsqueeze(1),
            "condition_video_row_count": torch.tensor(condition_counts).unsqueeze(1),
            "condition_audio_rows": torch.zeros(2, 0, AUDIO_ROW_WIDTH),
            "condition_audio_rows_mask": torch.zeros(2, 0, dtype=torch.bool),
            "condition_audio_row_count": torch.zeros(2, 1, dtype=torch.long),
        },
        non_tensor_dict={"micro_batch_size_per_gpu": 2},
    )
    if task == "fl2va":
        data["keyframe_frame_indices"] = torch.zeros(2, 1, dtype=torch.long)
    else:
        block_meta, count = serialize_ref_blocks([{"kind": "image", "latent_h": 4, "latent_w": 4}])
        data["ref_block_meta"] = block_meta.unsqueeze(0).repeat(2, 1, 1)
        data["ref_block_count"] = torch.full((2, 1), count)
    embeds_padding_2_no_padding(data)
    config = SimpleNamespace(
        diffusion_loss=DiffusionLossConfig(loss_mode="diffusion_nft"),
        global_batch_info={},
        loss_scale_factor=None,
        use_kl_loss=False,
        use_distill_loss=False,
    )
    result = BaseEngine.train_batch(engine, data, loss_function=partial(diffusion_loss, config))
    assert len(updates) == 1 and torch.isfinite(torch.tensor(updates)).all()
    assert parameter.item() != pytest.approx(0.5)
    assert shapes == [(4 + condition_counts[0], 6)] * 3 + [(16 + condition_counts[1], 10)] * 3
    assert result["model_output"] == {}

    inference = engine.forward_backward_batch(data, loss_function=None, forward_only=True)
    predictions = inference["model_output"]["forward_prediction"]
    assert predictions.is_nested
    predictions = pickle.loads(pickle.dumps(predictions.cpu()))
    assert [tuple(item.shape) for item in unpack_ragged_tensors(predictions, 2)] == [(1, lengths[0]), (1, lengths[1])]
