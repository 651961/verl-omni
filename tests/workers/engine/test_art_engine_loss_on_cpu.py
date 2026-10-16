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

"""Exercise ART loss assembly with the installed verl TensorDict contract."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.qwen_image21_diffusion_art.diffusers_training_adapter import QwenImage21DiffusionART
from verl_omni.trainer.diffusion.art_math import art_loss, art_scalar_covariance
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss


@pytest.mark.parametrize("restore", [True, False])
@pytest.mark.parametrize("magnitude, drift", [(0.25, 0.0), (0.0, 0.2), (0.0, 0.0)])
def test_art_engine_uses_real_tensordict_metadata_and_normalizes_gradients(restore, magnitude, drift):
    engine = object.__new__(ARTDiffusersFSDPEngine)
    engine.get_data_parallel_group = lambda: None
    config = DiffusionActorConfig(
        strategy="fsdp2",
        rollout_n=2,
        diffusion_loss=DiffusionLossConfig(loss_mode="diffusion_art", art_restore_zero_signal=restore),
    )
    state = torch.zeros(1, 4, 2)
    prediction = torch.full_like(state, drift, requires_grad=True)
    batch = tu.get_tensordict(
        {
            "art_old_velocities": torch.zeros(1, 1, 4, 2),
            "art_sigmas": torch.tensor([[0.5]]),
            "art_dt": torch.tensor([[-0.5]]),
            "art_directions": torch.ones_like(state),
            "art_magnitudes": torch.tensor([magnitude]),
        }
    )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=8)

    def run_steps(data, loss_function, *args, **kwargs):
        return engine.forward_step(data, loss_function, False, 0)

    engine._run_forward_backward_batch = run_steps
    engine.model_config = None
    engine.module = None
    adapter = SimpleNamespace(
        forward_art_velocity=lambda *args: (prediction, state),
        art_loss_coefficients=QwenImage21DiffusionART.art_loss_coefficients,
    )
    with patch("verl_omni.pipelines.model_base.DiffusionModelBase.get_class", return_value=adapter):
        with torch.no_grad():
            loss, _ = engine.forward_backward_batch(batch, partial(diffusion_loss, config))
            assert not torch.is_grad_enabled()
    with torch.enable_grad():
        expected = (
            art_loss(
                prediction,
                torch.zeros_like(state),
                batch["art_sigmas"][:, 0],
                batch["art_directions"],
                batch["art_magnitudes"],
                ode_velocity_coefficient=batch["art_dt"][:, 0],
                restore_anchor_mask=torch.tensor([restore]),
            )["per_item_loss"].mean()
            / 8
        )
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    expected_count = prediction.numel() if magnitude > 0 or (restore and drift > 0) else 0
    assert actual_grad.count_nonzero() == expected_count


@pytest.mark.parametrize("joint_clocks", [False, True], ids=["image", "video_audio"])
@pytest.mark.parametrize("restore", [True, False])
@torch.enable_grad()
def test_art_zero_signal_restoration_preserves_other_prompt_gradients(joint_clocks, restore):
    endpoints = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    stats = art_scalar_covariance(
        endpoints,
        torch.tensor([0.7, 0.7, 0.0, 1.0]),
        torch.tensor([0, 0, 1, 1]),
        2,
        zero_signal_mode="exact",
    )
    current = torch.full_like(endpoints, 0.2, requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    sigma = torch.tensor([0.5, 0.25]).reshape(1, 2, 1).expand_as(current) if joint_clocks else torch.full((4,), 0.5)
    loss_config = DiffusionLossConfig(loss_mode="diffusion_art")
    if not restore:
        loss_config = DiffusionLossConfig(loss_mode="diffusion_art", art_restore_zero_signal=False)
    config = DiffusionActorConfig(strategy="fsdp2", rollout_n=2, diffusion_loss=loss_config)
    loss, _ = DiffusionARTLoss.compute_loss(
        config=config,
        current_velocity=current,
        old_velocity=old,
        state=torch.zeros_like(current),
        sigma=sigma,
        dt=-sigma,
        direction=stats["group_directions"].repeat_interleave(2, dim=0),
        magnitude=stats["group_guidance_magnitudes"].repeat_interleave(2),
    )
    # Constant-reward rows restore to zero; the other prompt keeps its target of -1.
    target = torch.tensor([0.0, 0.0, -1.0, -1.0]).reshape(4, 1, 1)
    weights = sigma if joint_clocks else sigma.reshape(4, 1, 1)
    keep = torch.tensor([restore, restore, True, True]).reshape(4, 1, 1)
    expected_loss = (weights * (current.detach() - target).square() * keep).mean()
    expected_grad = 2 * weights * (current.detach() - target) * keep / current.numel()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    torch.testing.assert_close(current.grad, expected_grad)
    assert old.grad is None
