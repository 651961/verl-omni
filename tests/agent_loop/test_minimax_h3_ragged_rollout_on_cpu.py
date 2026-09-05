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
"""Mixed resolution/duration transport across rollout workers and reward scoring."""

import asyncio
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.utils import tensordict_utils as tu

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker, _InternalDiffusionAgentLoopOutput
from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.minimax_h3_diffusion_nft.agent_loop import MiniMaxH3DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager
from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict
from verl_omni.utils.batch_fields import get_batch_field


def _result(index):
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2, 3]]),
        response_diffusion_output=torch.full((1, 2 + index, 3, 32, 32 * (index + 1)), index, dtype=torch.uint8),
        reward_score=float(index),
        metrics=AgentLoopMetrics(),
        extra_fields={
            "ragged_tensor_keys": ("responses", "latents_clean", "audio"),
            "ragged_tensor_dims": {"latents_clean": 1},
            "latents_clean": torch.arange(10 + index, dtype=torch.float32).unsqueeze(0),
            "audio": torch.ones(1, 2, 20 + index),
            "audio_sample_rate": 32000,
            "train_timesteps": torch.tensor([[500.0, 250.0]]),
            "latent_meta": torch.tensor([[4, 6, index + 1, 4, 4, index + 2]]),
        },
    )


def _worker_output(indices):
    extra_info = np.empty(len(indices), dtype=object)
    extra_info[:] = [{"resolution": "1024x1024", "num_frames": 124, "frame_rate": 24.0} for _ in indices]
    return DiffusionAgentLoopWorker._postprocess(
        SimpleNamespace(max_prompt_embed_length=8),
        [_result(index) for index in indices],
        input_non_tensor_batch={
            "uid": np.array([f"uid-{index}" for index in indices], dtype=object),
            "extra_info": extra_info,
        },
    )


def test_worker_merge_serialization_and_actor_split_preserve_variable_tensors():
    # Each worker may be homogeneous although their results differ globally.
    combined = DataProto.concat([_worker_output([0, 0]), _worker_output([1, 2])])
    combined = pickle.loads(pickle.dumps(combined))
    assert len(combined) == 4
    assert "responses" not in combined.batch
    assert [tuple(item.shape) for item in get_batch_field(combined, "responses")] == [
        (2, 3, 32, 32),
        (2, 3, 32, 32),
        (3, 3, 32, 64),
        (4, 3, 32, 96),
    ]
    assert [item.shape[-1] for item in get_batch_field(combined, "audio")] == [20, 20, 21, 22]
    actor = _to_diffusion_worker_tensordict(combined)
    assert "responses" not in actor and "audio" not in actor
    assert actor["latents_clean"].is_nested
    parts = tu.chunk_tensordict(actor, 2)
    selected = tu.index_select_tensor_dict(parts[1], torch.tensor([1, 0]))
    assert [len(item) for item in selected["latents_clean"].unbind()] == [12, 11]
    torch.testing.assert_close(selected["latents_clean"][0], torch.arange(12, dtype=torch.float32))


def test_one_worker_can_collect_mixed_shapes():
    batch = _worker_output([0, 1, 2])
    assert len(get_batch_field(batch, "responses")) == 3


def test_visual_reward_receives_original_ragged_sample():
    batch = _worker_output([0, 1])
    batch.non_tensor_batch["data_source"] = np.array(["minimax_h3_fl2va"] * 2, dtype=object)
    rewards = np.empty(2, dtype=object)
    rewards[:] = [{"ground_truth": "Animate"}, {"ground_truth": "Animate"}]
    batch.non_tensor_batch["reward_model"] = rewards
    received = []

    async def score(**kwargs):
        received.append(kwargs["solution_image"])
        return {"score": 0.75}

    manager = object.__new__(VisualRewardManager)
    manager.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(pipeline={"output_type": "pt"}))
    )
    manager.compute_score = score
    manager.is_async_reward_score = True
    manager.reward_router_address = None
    for i in range(2):
        result = asyncio.run(manager.run_single(batch[i : i + 1]))
        assert result["reward_score"] == 0.75
    assert [item.shape[0] for item in received] == [2, 3]


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_row_spec_overrides_pipeline_shapes_and_requires_all_fields(task):
    params = {"task": task, "height": 768, "width": 1344, "num_frames": 124, "frame_rate": 24.0}
    fields = {"resolution": "1024x1024", "num_frames": 141, "frame_rate": 24.0}
    actual = MiniMaxH3DiffusionSingleTurnAgentLoop._apply_row_shape(params, {"extra_info": fields})
    assert (actual["height"], actual["width"], actual["num_frames"], actual["fps"]) == (1024, 1024, 141, 24)
    assert params["width"] == 1344
    with pytest.raises(ValueError, match="Missing required"):
        MiniMaxH3DiffusionSingleTurnAgentLoop._apply_row_shape(params, {"extra_info": {}})


@pytest.mark.parametrize("override", [{"duration": 5}, {"duration_seconds": 5}, {"target": {"duration_seconds": 5}}])
@pytest.mark.parametrize("nested", [False, True])
def test_global_duration_cannot_override_row_frame_count(override, nested):
    params = {"task": "ref2va", **({"extra_args": override} if nested else override)}
    fields = {"resolution": "1024x1024", "num_frames": 141, "frame_rate": 24.0}
    with pytest.raises(ValueError, match="duration override"):
        MiniMaxH3DiffusionSingleTurnAgentLoop._apply_row_shape(params, {"extra_info": fields})


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_h3_agent_uses_registered_algorithm_tensor_contract(monkeypatch, algorithm, task):
    calls = []

    async def generate(self, sampling_params, **kwargs):
        calls.append(sampling_params)
        return SimpleNamespace(extra_fields={})

    monkeypatch.setattr(DiffusionSingleTurnAgentLoop, "run", generate)
    agent = object.__new__(MiniMaxH3DiffusionSingleTurnAgentLoop)
    agent.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(algorithm=algorithm)))
    fields = {"resolution": "1024x1024", "num_frames": 141, "frame_rate": 24.0}
    result = asyncio.run(agent.run({"task": task}, extra_info=fields))
    assert (calls[0]["width"], calls[0]["height"], calls[0]["num_frames"], calls[0]["fps"]) == (1024, 1024, 141, 24)
    expected = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", algorithm).ragged_rollout_tensor_dims()
    assert result.extra_fields["ragged_tensor_dims"] == expected
    assert set(result.extra_fields["ragged_tensor_keys"]) == {"responses", "audio", *expected}
