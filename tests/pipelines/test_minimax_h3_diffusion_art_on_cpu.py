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

"""H3 ART replay, dual-clock gradients and ragged transport without model weights."""

import __future__

import ast
import asyncio
import copy
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/trainer/diffusion"))
from test_art_core_on_cpu import math, recorder, transport  # noqa: E402


def load_module(name, bindings):
    """Load pure adapter boundaries without importing the serving runtime."""
    path = ROOT / "verl_omni/pipelines/minimax_h3_diffusion_art" / (name + ".py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom)]
    namespace = {"torch": torch, "Any": object, **bindings}
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return types.SimpleNamespace(**namespace)


capture = load_module(
    "capture", {"ARTTrajectoryRecorder": recorder.ARTTrajectoryRecorder, "frozen_cpu": recorder.frozen_cpu}
)
data = load_module("data", {"pack_art_trajectories": transport.pack_art_trajectories, "pad_stack": transport.pad_stack})


class Batch(dict):
    """Minimal CPU batch supporting the transport clone contract."""

    def clone(self, recurse=False):
        return Batch(self)


class NativeModel(torch.nn.Module):
    """Deterministic native packed forward with observable condition dependence."""

    def forward(self, x, audio_x, img_pos, audio_pos, **kwargs):
        return x[0, img_pos] * 0.25 + 0.1, audio_x[0, audio_pos] * 0.5 - 0.2


def rollout(video_count=2, audio_count=3, *, reference_audio=False, seed=1, step_indices=None):
    gen = torch.Generator().manual_seed(seed)
    nv, na = video_count + 1, audio_count + int(reference_audio)
    length = nv + na + 2
    video_indices = torch.arange(2, 2 + nv)
    audio_indices = torch.arange(2 + nv, length)
    video_mask = torch.arange(nv) > 0
    audio_mask = torch.arange(na) >= int(reference_audio)
    branch = types.SimpleNamespace(
        img_pos_dev=video_indices,
        audio_pos_dev=audio_indices,
        update_mask_dev=video_mask,
        audio_update_mask_dev=audio_mask,
        img_position_ids_dev=torch.arange(length * 3).reshape(length, 3).double() / 7,
        token_tags_dev=torch.tensor([1] * 2 + [0] * nv + [2] * na),
        text_pos_dev=torch.arange(2),
        used_len=length,
    )
    video = torch.randn(nv, 96, generator=gen)
    audio = torch.randn(na, 32, generator=gen)
    embeds = torch.randn(2, 8, generator=gen).bfloat16()
    sigmas = torch.tensor([[1.0, 1.0], [0.6, 0.3], [0.0, 0.0]])
    saved = capture.H3ARTCapture(
        {"branch": branch, "sigmas_video": sigmas[:, 0], "sigmas_audio": sigmas[:, 1]}, step_indices, 7
    )
    model = NativeModel()
    hook = model.register_forward_hook(saved.record_forward, with_kwargs=True)
    for step in range(2):
        x, audio_x = torch.zeros(1, length, 96), torch.zeros(1, length, 32)
        x[0, video_indices], audio_x[0, audio_indices] = video, audio
        vv, va = model(x=x, audio_x=audio_x, img_pos=video_indices, audio_pos=audio_indices, prompt_embeds=embeds)
        video = video.clone()
        audio = audio.clone()
        video[video_mask] -= (sigmas[step + 1, 0] - sigmas[step, 0]) * vv[video_mask]
        audio[audio_mask] -= (sigmas[step + 1, 1] - sigmas[step, 1]) * va[audio_mask]
    hook.remove()
    return saved.finish(video, audio)


def pack(caches, uids=None):
    return Batch(data.pack_h3_art_caches(caches, torch.arange(len(caches)).float(), uids or ["a"] * len(caches)))


def row(batch, index):
    return Batch({key: value[index : index + 1] for key, value in batch.items()})


class H3ARTTests(unittest.TestCase):
    def test_native_states_replay_both_clocks_and_condition_anchors(self):
        for reference_audio in (False, True):
            caches = [rollout(reference_audio=reference_audio, seed=i) for i in (1, 2)]
            batch = data.trim_h3_art_transport(row(pack(caches), 0))
            for step in range(2):
                inputs, negative, state = data.h3_art_inputs(batch, step)
                self.assertIsNone(negative)
                cond = caches[0]["conditions"]
                video_mask, audio_mask = cond["video_mask"], cond["audio_mask"]
                torch.testing.assert_close(inputs["hidden_states"][0, ~video_mask], cond["video_condition"])
                torch.testing.assert_close(inputs["audio_hidden_states"][0, ~audio_mask], cond["audio_condition"])
                vv = inputs["hidden_states"][0] * 0.25 + 0.1
                va = inputs["audio_hidden_states"][0] * 0.5 - 0.2
                predicted = -capture.flatten_targets(vv[video_mask], va[audio_mask])
                torch.testing.assert_close(predicted, caches[0]["old_velocities"][step])
                sigma, dt = data.h3_art_coefficients(batch, step)
                expected = caches[0]["states"][step + 1] if step == 0 else caches[0]["endpoints"]
                torch.testing.assert_close((state + dt * predicted)[0], expected)
                times = inputs["timestep"][inputs["timestep_indices"]]
                torch.testing.assert_close(times[inputs["video_indices"][video_mask]], (1 - sigma[0, 0, 0]).expand(2))
                torch.testing.assert_close(times[inputs["audio_indices"][audio_mask]], (1 - sigma[0, -1, 0]).expand(3))
                if reference_audio:
                    torch.testing.assert_close(times[inputs["audio_indices"][~audio_mask]], torch.ones(1))

    def test_mixed_resolution_duration_trim_and_covariance(self):
        caches = [rollout(seed=1), rollout(seed=2), rollout(5, 7, seed=3), rollout(5, 7, seed=4)]
        batch = pack(caches, ["short", "short", "long", "long"])
        batch["art_states"][0, :, caches[0]["states"].shape[1] :] = float("nan")
        small = data.trim_h3_art_transport(row(batch, 0))
        self.assertEqual(small["art_states"].shape, (1, 2, 2 * 96 + 3 * 32, 1))
        self.assertTrue(torch.isfinite(small["art_states"]).all())
        for index in (0, 2):
            sample = data.trim_h3_art_transport(row(batch, index))
            expected = math.art_scalar_covariance(
                torch.stack([caches[index]["endpoints"], caches[index + 1]["endpoints"]]),
                torch.tensor([float(index), float(index + 1)]),
                torch.zeros(2, dtype=torch.long),
                1,
                zero_signal_mode="exact",
            )
            torch.testing.assert_close(sample["art_directions"], expected["group_directions"])

    def test_seed_dependent_conditions_remain_per_sample(self):
        caches = [rollout(seed=1), rollout(seed=2)]
        batch = pack(caches)
        self.assertFalse(torch.equal(batch["art_h3_video_condition"][0], batch["art_h3_video_condition"][1]))
        for i in range(2):
            torch.testing.assert_close(batch["art_h3_video_condition"][i], caches[i]["conditions"]["video_condition"])

    def test_selected_steps_and_frozen_storage(self):
        cache = rollout(step_indices=[1])
        self.assertEqual(cache["states"].shape[0], 1)
        torch.testing.assert_close(cache["sigmas"], torch.tensor([[0.6, 0.3]]))
        self.assertEqual(cache["policy_version"], 7)
        self.assertFalse(cache["states"].requires_grad)

    def test_layout_mismatch_rejected_even_with_equal_element_counts(self):
        caches = [rollout(seed=1), rollout(seed=2)]
        caches[1]["conditions"]["position_ids"] += 1
        with self.assertRaisesRegex(ValueError, "same video/audio layout"):
            pack(caches)

    def test_incomplete_or_inconsistent_groups_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            pack([rollout()])
        with self.assertRaisesRegex(ValueError, "same target resolution"):
            pack([rollout(), rollout(3, 4)])
        with self.assertRaisesRegex(ValueError, "same number of training steps"):
            pack([rollout(), rollout(step_indices=[0])])

    def test_dual_clock_loss_and_gradient_match_coordinate_formula(self):
        current = torch.tensor([[[0.2], [0.4], [0.8], [1.2]]], requires_grad=True)
        old = torch.zeros_like(current)
        sigma = torch.tensor([[[0.6], [0.6], [0.3], [0.3]]])
        dt = torch.tensor([[[-0.6], [-0.6], [-0.3], [-0.3]]])
        direction = torch.tensor([[[1.0], [-1.0], [2.0], [-2.0]]])
        magnitude = torch.tensor([0.5])
        for loss_space in ("x0", "velocity_linear_sigma", "velocity_uniform"):
            result = math.art_loss(
                current,
                old,
                sigma,
                direction,
                magnitude,
                state=torch.ones_like(current),
                ode_velocity_coefficient=dt,
                corrective_mask_enabled=False,
                loss_space=loss_space,
            )
            displacement = 0.5 * direction / direction.square().mean().sqrt()
            residual = displacement + sigma * current if loss_space == "x0" else -displacement - current
            weight = sigma if loss_space == "velocity_linear_sigma" else torch.ones_like(sigma)
            expected = (weight * residual.square()).mean()
            torch.testing.assert_close(result["per_item_loss"][0], expected)
            actual_grad = torch.autograd.grad(result["per_item_loss"].sum(), current, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_scalar_schedule_and_expanded_schedule_are_equivalent(self):
        gen = torch.Generator().manual_seed(5)
        current, old, direction = [torch.randn(2, 4, 3, generator=gen) for _ in range(3)]
        sigma, dt = torch.tensor([0.5, 1.0]), torch.tensor([-0.2, -0.4])
        for metric in ("fixed_mse", "noise_normalized_v", "q2_sde_kl", "scheduler_reference_metric"):
            a = math.art_loss(
                current, old, sigma, direction, torch.ones(2), ode_velocity_coefficient=dt, velocity_mask_metric=metric
            )
            b = math.art_loss(
                current,
                old,
                sigma[:, None, None].expand_as(current),
                direction,
                torch.ones(2),
                ode_velocity_coefficient=dt[:, None, None].expand_as(current),
                velocity_mask_metric=metric,
            )
            for key in a:
                torch.testing.assert_close(a[key], b[key])

    def test_invalid_schedule_broadcast_does_not_mix_batch_rows(self):
        with self.assertRaises(ValueError):
            math.art_loss(
                torch.ones(2, 4, 1), torch.zeros(2, 4, 1), torch.ones(1, 2, 1), torch.ones(2, 4, 1), torch.ones(2)
            )

    def test_real_tiny_h3_actor_backward_with_reference_audio(self):
        from diffusers import MiniMaxH3Transformer3DModel
        from tensordict import TensorDict

        class Registry:
            @staticmethod
            def register(*args, **kwargs):
                return lambda cls: cls

        adapter = load_module(
            "diffusers_training_adapter",
            {
                "DiffusionModelBase": Registry,
                "MiniMaxH3DiffusionNFT": object,
                "flatten_targets": capture.flatten_targets,
                "h3_art_inputs": data.h3_art_inputs,
            },
        ).MiniMaxH3DiffusionART
        torch.manual_seed(3)
        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=64,
            text_dim=8,
            freq_dim=16,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=2,
        )
        batch = data.trim_h3_art_transport(
            row(pack([rollout(reference_audio=True, seed=1), rollout(reference_audio=True, seed=2)]), 0)
        )
        batch = data.trim_h3_art_transport(TensorDict(batch, batch_size=[1]))
        prediction, state = adapter.forward_art_velocity(model, batch, 1)
        sigma, dt = data.h3_art_coefficients(batch, 1)
        result = math.art_loss(
            prediction,
            batch["art_old_velocities"][:, 1],
            sigma,
            batch["art_directions"],
            batch["art_magnitudes"],
            state=state,
            ode_velocity_coefficient=dt,
            corrective_mask_enabled=False,
        )
        result["per_item_loss"].sum().backward()
        for head in (model.proj_out, model.audio_proj_out):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        self.assertIsNone(batch["art_states"].grad)

    def test_rollout_cleanup_on_failure_and_validation_without_cache(self):
        from contextlib import nullcontext
        from unittest.mock import Mock

        class Registry:
            @staticmethod
            def register(*args, **kwargs):
                return lambda cls: cls

        class Native:
            fail = True

            def forward(self, request):
                self._art_hook = Mock()
                self.last_hook = self._art_hook
                if self.fail:
                    raise RuntimeError("decode failed")
                return types.SimpleNamespace(output=(torch.zeros(1), torch.zeros(1)))

        class Sync:
            def _ensure_prompt_text(self, request):
                self._h3_prompt_ids = [1]

        pipeline = load_module(
            "vllm_omni_rollout_adapter",
            {
                "VllmOmniPipelineBase": Registry,
                "MiniMaxH3WeightSyncMixin": Sync,
                "MiniMaxH3Pipeline": Native,
                "DiffusionIOSpec": lambda **kwargs: None,
                "MediaSpec": lambda *args, **kwargs: None,
                "ref2va_reference_image_short_edge": lambda value: nullcontext(),
                "with_rollout_data": lambda output, **kwargs: kwargs,
            },
        ).MiniMaxH3DiffusionARTPipeline
        pipe = object.__new__(pipeline)
        pipe._art_cache = pipe._art_capture = pipe._art_hook = None
        pipe._reference_image_short_edge = 256
        sampling = types.SimpleNamespace(
            num_outputs_per_prompt=1, frame_rate=24, extra_args={"art_capture_trajectory": False}
        )
        request = types.SimpleNamespace(requests=[types.SimpleNamespace(sampling_params=sampling)])
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            pipe.forward(request)
        pipe.last_hook.remove.assert_called_once()
        self.assertIsNone(pipe._h3_prompt_ids)
        self.assertIsNone(pipe._art_capture)
        pipe.fail = False
        result = pipe.forward(request)
        self.assertEqual(result["rl"], {})
        self.assertIsNone(pipe._art_cache)
        pipe.set_published_policy_version(9)
        self.assertEqual(pipe._published_policy_version, 9)

    def test_per_sample_sizes_keep_group_geometry_and_input_immutable(self):
        class Registry:
            @staticmethod
            def register(name):
                return lambda cls: cls

        import operator

        agent = load_module(
            "agent_loop",
            {"operator": operator, "register": Registry.register, "MiniMaxH3DiffusionSingleTurnAgentLoop": object},
        )
        original = {"height": 288, "width": 448, "num_frames": 120}
        before = copy.deepcopy(original)
        for geometry in (original, {"height": 576, "width": 928, "num_frames": 240}):
            for defaults in (original, {}):
                result = agent.h3_sample_parameters(defaults, geometry)
                self.assertEqual(result, {**geometry, "frame_rate": 24.0})
        self.assertEqual(original, before)
        for key in original:
            for value in (None, "missing"):
                geometry = dict(original)
                if value is None:
                    geometry[key] = None
                else:
                    del geometry[key]
                with self.assertRaisesRegex(ValueError, f"requires extra_info.{key}"):
                    agent.h3_sample_parameters(original, geometry)
        for invalid in (
            {"height": 513},
            {"num_frames": 0},
            {"num_frames": 2.5},
            {"num_frames": True},
            {"frame_rate": 30},
        ):
            with self.assertRaises(ValueError):
                agent.h3_sample_parameters(original, {**original, **invalid})

    def test_agent_forwards_row_geometry_for_training_and_validation(self):
        import operator
        from unittest.mock import AsyncMock

        class NativeAgent:
            run = AsyncMock()

        agent = load_module(
            "agent_loop",
            {
                "operator": operator,
                "register": lambda name: lambda cls: cls,
                "MiniMaxH3DiffusionSingleTurnAgentLoop": NativeAgent,
            },
        ).MiniMaxH3ARTAgentLoop()
        rows = (
            {"height": 288, "width": 448, "num_frames": 120},
            {"height": 576, "width": 928, "num_frames": 240},
        )
        for task in ("fl2va", "ref2va"):
            for validate in (False, True):
                sampling = {"task": task, "height": 512, "width": 512, "num_frames": 1}
                before = dict(sampling)
                for row in rows:
                    asyncio.run(agent.run(sampling, extra_info=row, validate=validate))
                    NativeAgent.run.assert_awaited_with(
                        {**sampling, **row, "frame_rate": 24.0}, extra_info=row, validate=validate
                    )
                self.assertEqual(sampling, before)
        NativeAgent.run.reset_mock()
        with self.assertRaisesRegex(ValueError, "requires extra_info.height"):
            asyncio.run(agent.run(sampling))
        NativeAgent.run.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
