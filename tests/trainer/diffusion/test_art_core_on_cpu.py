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

"""Run directly without verl, Ray, vLLM, TransferQueue, or pytest installed."""

import copy
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_art_cpu_kernels"
package = types.ModuleType(PACKAGE)
package.__path__ = []
sys.modules[PACKAGE] = package


def load(name):
    paths = {
        "art_transport": "verl_omni/pipelines/diffusion_art/data.py",
        "art_data": "verl_omni/pipelines/qwen_image21_diffusion_art/data.py",
        "art_media": "verl_omni/pipelines/qwen_image21/media.py",
        "art_capture": "verl_omni/pipelines/qwen_image21_diffusion_art/capture.py",
        "art_recorder": "verl_omni/pipelines/diffusion_art/capture.py",
        "art_math": "verl_omni/pipelines/diffusion_art/objective.py",
        "art_precision": "verl_omni/pipelines/diffusion_art/precision.py",
    }
    path = ROOT / paths[name]
    module = types.ModuleType(PACKAGE + "." + name)
    module.__file__ = str(path)
    module.__package__ = PACKAGE
    sys.modules[module.__name__] = module
    source = path.read_text().replace(
        "from verl_omni.pipelines.diffusion_art.objective", "from " + PACKAGE + ".art_math"
    )
    source = source.replace("from verl_omni.pipelines.diffusion_art.data", "from " + PACKAGE + ".art_transport")
    source = source.replace("from verl_omni.pipelines.diffusion_art.capture", "from " + PACKAGE + ".art_recorder")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


math = load("art_math")
transport = load("art_transport")
data = load("art_data")
recorder = load("art_recorder")
capture = load("art_capture")
precision = load("art_precision")
media = load("art_media")


def cache(tokens=4, references=0, negative=False, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def condition(length):
        return {
            "encoder_hidden_states": torch.randn(1, length + references // 4, 8, generator=gen).bfloat16(),
            "encoder_hidden_states_mask": torch.ones(1, length + references // 4, dtype=torch.bool),
            "img_mask": torch.tensor([[False] * length + [True] * ((tokens + references) // 4)]),
            "img_shapes": [[(1, 2, 2)] * (references // 4) + [(1, 2, tokens // 2)]],
        }

    return {
        "states": torch.randn(2, tokens, 4, generator=gen),
        "old_velocities": torch.randn(2, tokens, 4, generator=gen),
        "endpoints": torch.randn(tokens, 4, generator=gen),
        "sigmas": torch.tensor([1.0, 0.5]),
        "dt": torch.tensor([-0.5, -0.5]),
        "timesteps": torch.tensor([1000.0, 500.0]),
        "policy_version": 7,
        "conditions": {
            "inputs": condition(3),
            "negative_inputs": condition(2) if negative else None,
            "image_latents": torch.randn(1, references, 4, generator=gen).bfloat16(),
            "true_cfg_scale": 2.0 if negative else 1.0,
        },
    }


class ARTCoreTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.enable_grad())

    def test_partitioned_loss_matches_equal_modality_formula_with_ragged_padding(self):
        sizes = torch.tensor([[2, 3], [4, 1], [1, 2]])
        magnitudes = torch.tensor([[1.0, 0.5], [0.2, 0.0], [0.0, 0.3]])
        current = torch.full((3, 5, 1), 0.2, requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        direction, sigma = torch.zeros_like(current), torch.zeros_like(current)
        expected_grad = torch.zeros_like(current)
        expected_loss = 0.0
        for row, counts in enumerate(sizes.tolist()):
            offset = 0
            for group, count in enumerate(counts):
                region = (row, slice(offset, offset + count))
                sign = 1 if group == 0 else -1
                weight = 0.6 if group == 0 else 0.3
                direction[region] = sign * (8 if group == 0 else 2)
                sigma[region] = weight
                if magnitudes[row, group] > 0:
                    residual = 0.2 + sign * magnitudes[row, group]
                    expected_loss += weight * residual.square() / 6
                    expected_grad[region] = 2 * weight * residual / (6 * count)
                offset += count
            direction[row, offset:] = float("nan")
            sigma[row, offset:] = float("nan")
        result = math.art_feature_group_loss(
            current,
            old,
            sigma,
            direction,
            magnitudes,
            feature_group_sizes=sizes,
            ode_velocity_coefficient=-sigma,
            corrective_mask_enabled=False,
        )
        loss = result["per_item_loss"].mean()
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        torch.testing.assert_close(current.grad, expected_grad)
        self.assertIsNone(old.grad)
        self.assertEqual(result["keep_mask"].tolist(), [[True, True], [True, False], [False, True]])

    def test_covariance_matches_manual_unbiased_estimator(self):
        endpoints = torch.tensor([[1.0, 3.0], [4.0, 2.0], [2.0, 8.0]])
        rewards = torch.tensor([1.0, -2.0, 4.0])
        expected = ((rewards - rewards.mean())[:, None] * endpoints).sum(0) / 2
        result = math.art_scalar_covariance(endpoints, rewards, torch.zeros(3, dtype=torch.long), 1)
        torch.testing.assert_close(result["group_covariance"][0], expected)
        torch.testing.assert_close(result["group_directions"].square().mean(), torch.tensor(1.0))

    def test_constant_rewards_have_exact_zero_signal_and_gradient(self):
        endpoints = torch.randn(7, 5, 4)
        result = math.art_scalar_covariance(
            endpoints, torch.full((7,), 0.7), torch.zeros(7, dtype=torch.long), 1, zero_signal_mode="exact"
        )
        self.assertEqual(result["group_directions"].count_nonzero().item(), 0)
        self.assertEqual(result["group_guidance_magnitudes"].item(), 0.0)
        v = torch.randn(1, 5, 4, requires_grad=True)
        loss = math.art_loss(
            v,
            torch.zeros_like(v),
            torch.tensor([0.5]),
            result["group_directions"],
            result["group_guidance_magnitudes"],
            ode_velocity_coefficient=torch.tensor([-0.2]),
        )
        loss["per_item_loss"].sum().backward()
        self.assertEqual(v.grad.count_nonzero().item(), 0)

    def test_frozen_target_and_velocity_loss_gradient(self):
        v = torch.tensor([[[0.1, -0.2]]], requires_grad=True)
        old = torch.tensor([[[0.2, 0.4]]], requires_grad=True)
        direction = torch.tensor([[[1.0, -1.0]]], requires_grad=True)
        magnitude = torch.tensor([0.3], requires_grad=True)
        result = math.art_loss(
            v,
            old,
            torch.tensor([0.5]),
            direction,
            magnitude,
            corrective_mask_enabled=False,
            ode_velocity_coefficient=torch.tensor([-0.1]),
        )
        expected_target = old.detach() - 0.3 * direction.detach()
        torch.testing.assert_close(result["target"], expected_target)
        result["per_item_loss"].sum().backward()
        torch.testing.assert_close(v.grad, (v.detach() - expected_target) * 0.5)
        self.assertIsNone(old.grad)
        self.assertIsNone(direction.grad)
        self.assertIsNone(magnitude.grad)
        self.assertFalse(result["target"].requires_grad)

    def test_directional_mask_and_reference_regularization(self):
        old = torch.zeros(2, 1, 2)
        v = torch.tensor([[[0.5, 0.5]], [[-1.0, -1.0]]], requires_grad=True)
        reference = torch.zeros_like(v, requires_grad=True)
        result = math.art_loss(
            v,
            old,
            torch.ones(2),
            -torch.ones_like(v),
            torch.ones(2),
            velocity_mask_metric="fixed_mse",
            movement_mask_threshold=0.1,
            reference_prediction=reference,
            reference_mse_beta=0.25,
        )
        self.assertEqual(result["masked"].tolist(), [True, False])
        result["per_item_loss"].sum().backward()
        self.assertTrue(v.grad[0].abs().sum() > 0)  # reference still restores masked rows
        self.assertIsNone(reference.grad)

    def test_ragged_conditions_trim_without_changing_group_statistics(self):
        caches = [cache(4, 0, True, i) for i in range(2)] + [cache(8, 4, True, i + 2) for i in range(2)]
        rewards, uids = [0.0, 1.0, 2.0, -1.0], ["a", "a", "b", "b"]
        packed = data.pack_art_caches(caches, rewards, uids)
        perm = [3, 0, 2, 1]
        shuffled = data.pack_art_caches([caches[i] for i in perm], [rewards[i] for i in perm], [uids[i] for i in perm])
        for row, source in enumerate(perm):
            torch.testing.assert_close(shuffled["art_directions"][row], packed["art_directions"][source])
        for row, original in enumerate(caches):
            micro = {k: v[row : row + 1] for k, v in packed.items()}
            positive, negative, state = data.qwen21_art_inputs(micro, 1)
            n = original["states"].shape[1]
            self.assertEqual(state.shape, (1, n, 4))
            torch.testing.assert_close(positive["hidden_states"][:, -n:], original["states"][1:2].bfloat16())
            for prepared, key in [(positive, "inputs"), (negative, "negative_inputs")]:
                cond = original["conditions"][key]
                torch.testing.assert_close(prepared["img_mask"], cond["img_mask"])
                torch.testing.assert_close(prepared["encoder_hidden_states"], cond["encoder_hidden_states"])
                self.assertEqual(prepared["img_shapes"], cond["img_shapes"])
                self.assertNotIn("kv_cache", prepared)
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            data.trim_art_transport(packed)

    def test_reject_incomplete_groups_and_bad_schedules(self):
        for update, pattern in [
            (lambda c: c.update(dt=torch.ones(2)), "decreasing"),
        ]:
            caches = [cache(), cache(seed=1)]
            update(caches[1])
            with self.assertRaisesRegex(ValueError, pattern):
                data.pack_art_caches(caches, [0.0, 1.0], ["a", "a"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            data.pack_art_caches([cache()], [1.0], ["a"])

    def test_async_versions_preserve_sample_velocities_and_covariance(self):
        caches = [cache(seed=i) for i in range(2)]
        expected = data.pack_art_caches(caches, [0.0, 1.0], ["a", "a"])
        caches[1]["policy_version"] = 8
        actual = data.pack_art_caches(caches, [0.0, 1.0], ["a", "a"])
        self.assertNotIn("art_policy_version", actual)
        self.assertNotIn("art_canvas", actual)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_fp32_adam_preserves_updates_smaller_than_bf16_spacing(self):
        master = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        adam = torch.optim.AdamW([master], lr=1e-4, weight_decay=0)
        published = master.detach().bfloat16().clone()
        for _ in range(4):
            master.grad = torch.ones_like(master)
            adam.step()
        self.assertFalse(torch.equal(master.detach(), torch.ones_like(master)))
        self.assertTrue(torch.equal(master.detach().bfloat16(), published))
        for _ in range(40):
            master.grad = torch.ones_like(master)
            adam.step()
        newer = master.detach().bfloat16().clone()
        indices = (newer.view(torch.int16) != published.view(torch.int16)).nonzero().flatten()
        replayed = published.clone()
        replayed[indices] = newer[indices]
        self.assertTrue(torch.equal(replayed, newer))
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(adam.state[master][key].dtype, torch.float32)
        self.assertEqual(master.grad.dtype, torch.float32)

    def test_precision_contract(self):
        model = {"lora_rank": 0}
        actor = {"strategy": "fsdp2", "ppo_micro_batch_size_per_gpu": 1, "fsdp_config": {"model_dtype": "fp32"}}
        precision.validate_art_precision(model, actor)
        for key, value in [
            ("model_dtype", "bf16"),
            ("offload_policy", True),
            ("mixed_precision", {"reduce_dtype": "bf16"}),
            ("mixed_precision", {"param_dtype": "fp32"}),
        ]:
            modified = copy.deepcopy(actor)
            modified["fsdp_config"][key] = value
            with self.assertRaises(ValueError):
                precision.validate_art_precision(model, modified)
        for micro_batch_size in (None, 0, 2, 4):
            with (
                self.subTest(micro_batch_size=micro_batch_size),
                self.assertRaisesRegex(ValueError, "ppo_micro_batch_size_per_gpu=1"),
            ):
                precision.validate_art_precision(model, {**actor, "ppo_micro_batch_size_per_gpu": micro_batch_size})
        with self.assertRaisesRegex(ValueError, "old/EMA"):
            precision.validate_art_precision({"policy_state_adapters": ["default", "old"]}, actor)

    def test_rgba_bytes_path_and_text(self):
        rgba = Image.new("RGBA", (64, 32), (17, 23, 19, 31))
        buffer = BytesIO()
        rgba.save(buffer, format="PNG")
        decoded = media.load_rgba({"bytes": buffer.getvalue()})
        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.getpixel((0, 0)), rgba.getpixel((0, 0)))
        text = media.messages_to_text(
            [{"role": "user", "content": [{"type": "image", "image": rgba}, {"type": "text", "text": "edit alpha"}]}]
        )
        self.assertEqual(text, "edit alpha")


class CaptureTests(unittest.TestCase):
    def pipe(self):
        class Transformer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                self.calls = getattr(self, "calls", 0) + 1
                return (hidden_states * 0.25,)

        class Scheduler:
            timesteps = torch.tensor([1000.0, 500.0])
            sigmas = torch.tensor([1.0, 0.5, 0.0])
            step_index = None
            begin_index = 0

            def step(self, velocity, t, sample, return_dict=False):
                i = self.step_index or 0
                out = sample.float() + (self.sigmas[i + 1] - self.sigmas[i]) * velocity.float()
                self.step_index = i + 1
                return (out.to(sample.dtype),)

        return types.SimpleNamespace(transformer=Transformer(), scheduler=Scheduler(), vae_scale_factor=16)

    def generate(self, pipe, cfg=1.0):
        state = torch.ones(1, 4, 4, dtype=torch.bfloat16)
        cond = cache(4, 0, cfg > 1)["conditions"]
        for t in (torch.tensor(1000.0), torch.tensor(500.0)):
            inputs = {**cond["inputs"], "hidden_states": state, "timestep": t[None] / 1000}
            pred = pipe.transformer(**inputs)[0]
            if cfg > 1:
                neg = pipe.transformer(**{**inputs, **cond["negative_inputs"]})[0]
                pred = neg + cfg * (pred - neg)
            state = pipe.scheduler.step(pred, t, state)[0]
        return state

    def test_selected_guided_predictions_are_frozen_without_extra_forwards(self):
        for cfg, calls in ((1.0, 2), (2.0, 4)):
            pipe = self.pipe()
            original = pipe.scheduler.step
            with capture.capture_art_rollout(
                pipe, types.SimpleNamespace(extra_args={"art_step_indices": [1]}, true_cfg_scale=cfg), 9
            ) as saved:
                endpoint = self.generate(pipe, cfg)
            self.assertEqual(pipe.transformer.calls, calls)
            self.assertEqual(saved["states"].shape, (1, 4, 4))
            torch.testing.assert_close(saved["old_velocities"][0], (saved["states"][0].bfloat16() * 0.25).float())
            torch.testing.assert_close(saved["endpoints"], endpoint[0].float())
            self.assertEqual(saved["policy_version"], 9)
            self.assertEqual(saved["conditions"]["inputs"]["img_shapes"], [[(1, 2, 2)]])
            self.assertEqual(saved["conditions"]["image_latents"].shape[1], 0)
            self.assertEqual(pipe.scheduler.step, original)
            endpoint.zero_()
            self.assertTrue(saved["endpoints"].abs().sum() > 0)

    def test_hooks_restore_on_failure_and_invalid_selection(self):
        pipe = self.pipe()
        original = pipe.scheduler.step
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with capture.capture_art_rollout(pipe, types.SimpleNamespace(extra_args={}, true_cfg_scale=1.0), 0):
                raise RuntimeError("forced")
        self.assertEqual(pipe.scheduler.step, original)
        self.assertEqual(len(pipe.transformer._forward_pre_hooks), 0)
        for indices in ([1, 0], [0, 0], [-1], [], [3]):
            with self.assertRaises(ValueError):
                with capture.capture_art_rollout(
                    pipe, types.SimpleNamespace(extra_args={"art_step_indices": indices}, true_cfg_scale=1.0), 0
                ):
                    self.generate(pipe)
            pipe.scheduler.step_index = None
            self.assertEqual(len(pipe.transformer._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
