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

import torch
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.art_math import art_loss
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss


def test_art_engine_uses_real_tensordict_metadata_and_normalizes_gradients():
    engine = object.__new__(ARTDiffusersFSDPEngine)
    engine.get_data_parallel_group = lambda: None
    config = DiffusionActorConfig(
        strategy="fsdp2", rollout_n=2, diffusion_loss=DiffusionLossConfig(loss_mode="diffusion_art")
    )
    state = torch.zeros(1, 4, 2)
    prediction = torch.zeros_like(state, requires_grad=True)
    batch = tu.get_tensordict(
        {
            "art_old_velocities": torch.zeros(1, 1, 4, 2),
            "art_sigmas": torch.tensor([[0.5]]),
            "art_dt": torch.tensor([[-0.5]]),
            "art_directions": torch.ones_like(state),
            "art_magnitudes": torch.tensor([0.25]),
        }
    )
    tu.assign_non_tensor(batch, gradient_accumulation_steps=8)

    def run_steps(data, loss_function, *args, **kwargs):
        return engine.forward_step(data, loss_function, False, 0)

    engine._run_forward_backward_batch = run_steps
    engine.model_config = None
    engine.module = None
    adapter = SimpleNamespace(forward_art_velocity=lambda *args: (prediction, state))
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
            )["per_item_loss"].mean()
            / 8
        )
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.count_nonzero() == prediction.numel()
