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
"""Mixed-shape H3 trajectory transport, log-prob replay, and accumulated PPO gradients."""

import pickle
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import BaseEngine

import verl_omni.workers.engine.fsdp.diffusers_impl as engine_module
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker, _InternalDiffusionAgentLoopOutput
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    build_layout_from_meta,
    serialize_ref_blocks,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    H3_AUDIO_WIDTH,
    H3_VIDEO_WIDTH,
    combine_log_probs,
    flatten_joint_latents,
    sample_h3_transition,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict
from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer
from verl_omni.trainer.diffusion.teacher_manager import DiffusionTeacherManager
from verl_omni.utils.ragged_tensors import unpack_ragged_tensors
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding


@pytest.fixture(autouse=True)
def _cpu_schedulers(monkeypatch):
    monkeypatch.setattr(
        "verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter.get_device_name", lambda: "cpu"
    )


def _model_config():
    return OmegaConf.create(
        {
            "architecture": "MiniMaxH3Pipeline",
            "algorithm": "flow_grpo",
            "external_lib": None,
            "pipeline": {"num_inference_steps": 4, "height": 768, "width": 1344},
            "algo": {"noise_level": 0.8, "sde_type": "cps"},
        }
    )


def _rollout(index, task):
    meta = [[4, 6, 1, 4, 4, 3], [16, 10, 2, 4, 8, 5]][index]
    nv, na = meta[:2]
    text_len = 2 + index
    generator = torch.Generator().manual_seed(42 + index)
    video = torch.randn(1, nv, H3_VIDEO_WIDTH, generator=generator)
    audio = torch.randn(1, na, H3_AUDIO_WIDTH, generator=generator)
    fields = {
        "prompt_embeds": torch.ones(1, 4, 8),
        "prompt_embeds_mask": (torch.arange(4) < text_len).unsqueeze(0),
    }
    if task == "fl2va":
        positions, tags, vi, ai, ti, ncv, _ = build_layout_from_meta(meta, text_len, keyframe_anchors=("first",))
        condition = torch.full((1, ncv, H3_VIDEO_WIDTH), 2.0)
        fields.update(
            {
                "h3_seq_len": torch.tensor([len(positions)]),
                "h3_video_rows": torch.tensor([len(vi)]),
                "h3_audio_rows": torch.tensor([len(ai)]),
                "h3_position_ids": positions.unsqueeze(0),
                "h3_token_tags": tags.unsqueeze(0),
                "h3_video_indices": vi.unsqueeze(0),
                "h3_audio_indices": ai.unsqueeze(0),
                "h3_text_indices": ti.unsqueeze(0),
                "h3_video_update_mask": (torch.arange(len(vi)) >= ncv).unsqueeze(0),
            }
        )
    else:
        blocks = [{"kind": "image", "latent_h": 4, "latent_w": 4}]
        if index:
            blocks.append({"kind": "audio", "ref_audio_t": 2})
        block_meta, count = serialize_ref_blocks(blocks)
        fields.update(
            {
                "latent_meta": torch.tensor([meta]),
                "prompt_token_tags": torch.ones(1, 4, dtype=torch.long),
                "condition_video_rows": torch.full((1, 4, H3_VIDEO_WIDTH), 2.0),
                "condition_audio_rows": torch.full((1, 4 * index, H3_AUDIO_WIDTH), 2.0),
                "condition_video_row_count": torch.tensor([[4]]),
                "condition_audio_row_count": torch.tensor([[4 * index]]),
                "ref_block_meta": block_meta.unsqueeze(0),
                "ref_block_count": torch.tensor([[count]]),
            }
        )
    schedulers = MiniMaxH3FlowGRPO.build_scheduler(_model_config())
    current, following, log_probs = [], [], []
    for step in range(2):
        # The rollout policy predicts zero velocity; replay must reproduce its densities.
        vout = sample_h3_transition(
            schedulers[0],
            video,
            torch.zeros_like(video),
            step,
            noise_level=0.8,
            sde_type="cps",
            generator=generator,
        )
        aout = sample_h3_transition(
            schedulers[1],
            audio,
            torch.zeros_like(audio),
            step,
            noise_level=0.8,
            sde_type="cps",
            generator=generator,
        )
        full_video = torch.cat([condition, video], dim=1) if task == "fl2va" else video
        next_video = torch.cat([condition, vout[0]], dim=1) if task == "fl2va" else vout[0]
        current.append(flatten_joint_latents(full_video, audio))
        following.append(flatten_joint_latents(next_video, aout[0]))
        log_probs.append(combine_log_probs(vout[1], aout[1]))
        video, audio = vout[0], aout[0]
    dims = MiniMaxH3FlowGRPO.ragged_rollout_tensor_dims()
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2, 3]]),
        response_diffusion_output=torch.zeros(1, 2 + index, 3, 32, 32 * (index + 1), dtype=torch.uint8),
        response_logprobs=torch.stack(log_probs, dim=1),
        metrics=AgentLoopMetrics(),
        extra_fields={
            **fields,
            "ragged_tensor_keys": ("responses", "audio", *dims),
            "ragged_tensor_dims": dims,
            "all_latents": torch.stack(current, dim=1),
            "all_next_latents": torch.stack(following, dim=1),
            "all_timesteps": (1 - schedulers[0].sigmas[:2]).unsqueeze(0),
            "h3_audio_timesteps": (1 - schedulers[1].sigmas[:2]).unsqueeze(0),
            "h3_step_indices": torch.tensor([[0, 1]]),
            "audio": torch.zeros(1, 2, 20 + index),
            "audio_sample_rate": 32000,
        },
    )


def _batch(indices, task):
    return DiffusionAgentLoopWorker._postprocess(
        SimpleNamespace(max_prompt_embed_length=8),
        [_rollout(i, task) for i in indices],
        input_non_tensor_batch={"uid": np.array([str(i) for i in indices], dtype=object)},
    )


def _engine(monkeypatch, parameter):
    monkeypatch.setattr(engine_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(engine_module, "device_name", "cpu")
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.model_config = _model_config()
    engine.scheduler = MiniMaxH3FlowGRPO.build_scheduler(engine.model_config)
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.get_data_parallel_group = lambda: None
    engine.is_mp_src_rank_with_outputs = lambda: True
    engine.module = MagicMock(
        side_effect=lambda **kw: (
            kw["hidden_states"] * parameter,
            kw["audio_hidden_states"] * parameter,
        )
    )
    return engine


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_mixed_shape_replay_and_update_preserve_per_sample_density(monkeypatch, task):
    # Homogeneous workers returning different shapes must still concatenate and serialize.
    batch = pickle.loads(pickle.dumps(DataProto.concat([_batch([0], task), _batch([1], task)])))
    actor = embeds_padding_2_no_padding(_to_diffusion_worker_tensordict(batch))
    actor = pickle.loads(pickle.dumps(actor.cpu()))
    tu.assign_non_tensor(actor, micro_batch_size_per_gpu=8)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    engine = _engine(monkeypatch, parameter)

    inference = engine.forward_backward_batch(actor, loss_function=None, forward_only=True)
    output = inference["model_output"]
    assert not output["log_probs"].is_nested
    assert output["log_probs"].shape == (2, 2)
    torch.testing.assert_close(output["log_probs"], batch.batch["rollout_log_probs"], rtol=1e-5, atol=1e-6)
    assert output["prev_sample_mean"].is_nested
    assert [item.shape[-1] for item in unpack_ragged_tensors(output["prev_sample_mean"], 3)] == [
        4 * H3_VIDEO_WIDTH + 6 * H3_AUDIO_WIDTH,
        16 * H3_VIDEO_WIDTH + 10 * H3_AUDIO_WIDTH,
    ]

    # Exercise the real driver old-policy method and its DataProto -> TensorDict roundtrip.
    def infer(td):
        outputs = []
        td = pickle.loads(pickle.dumps(td.cpu()))
        for part in tu.chunk_tensordict(td, 2):
            part = pickle.loads(pickle.dumps(part))
            result = engine.forward_backward_batch(part, loss_function=None, forward_only=True)
            worker_output = tu.get_tensordict(result["model_output"], non_tensor_dict={"metrics": {}})
            outputs.append(pickle.loads(pickle.dumps(worker_output.cpu())))
        return pickle.loads(pickle.dumps(tu.concat_tensordict(outputs)))

    trainer = object.__new__(PolicyGradientRayTrainer)
    trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=engine.model_config))
    trainer.actor_rollout_wg = SimpleNamespace(infer_actor_batch=infer)
    old, _ = trainer._compute_old_log_prob(batch)
    batch.union(old)
    trainer.ref_in_actor = True
    batch.union(trainer._compute_ref_log_prob(batch))
    teacher = object.__new__(DiffusionTeacherManager)
    teacher.model_config = engine.model_config
    batch.union(teacher._to_dataproto(infer(_to_diffusion_worker_tensordict(batch))))
    batch = pickle.loads(pickle.dumps(batch))
    for key in ("old_prev_sample_mean", "ref_prev_sample_mean", "teacher_prev_sample_mean"):
        assert key not in batch.batch
        assert [tuple(item.shape) for item in batch.non_tensor_batch[key]] == [(2, 1, 576), (2, 1, 1856)]
    batch.batch["advantages"] = torch.tensor([[1.0, 1.0], [-0.5, -0.5]])
    actor = embeds_padding_2_no_padding(_to_diffusion_worker_tensordict(batch))
    actor = pickle.loads(pickle.dumps(actor.cpu()))
    tu.assign_non_tensor(actor, micro_batch_size_per_gpu=8)
    config = DiffusionActorConfig(
        strategy="fsdp2",
        rollout_n=2,
        ppo_micro_batch_size_per_gpu=1,
        diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo"),
        use_kl_loss=True,
    )
    loss_fn = partial(diffusion_loss, config)

    # Compare combined accumulation with the arithmetic mean of standalone sample gradients.
    gradients = []
    for part in tu.chunk_tensordict(actor, 2):
        parameter.grad = None
        engine.forward_backward_batch(part, loss_function=loss_fn)
        gradients.append(parameter.grad.detach().clone())
    expected_gradient = torch.stack(gradients).mean()
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    updates = []
    engine.optimizer_zero_grad = optimizer.zero_grad

    def optimizer_step():
        updates.append(parameter.grad.detach().clone())
        optimizer.step()
        return parameter.grad.item()

    engine.optimizer_step = optimizer_step
    result = BaseEngine.train_batch(engine, actor, loss_function=loss_fn)
    assert len(updates) == 1
    torch.testing.assert_close(updates[0], expected_gradient)
    torch.testing.assert_close(parameter.detach(), -0.01 * expected_gradient)
    assert result["model_output"] == {}


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_mixed_worker_chunk_and_reordering_preserve_trajectory_pairs(task):
    batch = _batch([0, 1], task)
    actor = pickle.loads(pickle.dumps(_to_diffusion_worker_tensordict(batch)))
    reordered = tu.index_select_tensor_dict(actor, torch.tensor([1, 0]))
    parts = [pickle.loads(pickle.dumps(part)) for part in tu.chunk_tensordict(reordered, 2)]
    restored = pickle.loads(pickle.dumps(tu.concat_tensordict(parts)))
    for key in ("all_latents", "all_next_latents"):
        for actual, original in zip(
            unpack_ragged_tensors(restored[key], 3), reversed(batch.non_tensor_batch[key]), strict=True
        ):
            torch.testing.assert_close(actual, original)
