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

"""Request-local storage of the ODE states consumed by DiffusionART."""

import copy

import torch


def frozen_cpu(value):
    """Detach transport data from device storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: frozen_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(frozen_cpu(v) for v in value)
    return copy.deepcopy(value)


class ARTTrajectoryRecorder:
    """Capture selected scheduler inputs directly into their final CPU tensors."""

    def __init__(self, num_steps, step_indices=None):
        selected = list(range(num_steps)) if step_indices is None else list(step_indices)
        if (
            not selected
            or any(type(i) is not int or i < 0 or i >= num_steps for i in selected)
            or selected != sorted(set(selected))
        ):
            raise ValueError("art_step_indices must be sorted, unique indices within the rollout schedule.")
        self.selected = set(selected)
        self.count = len(selected)
        self.records = {}
        self.record_count = 0

    def record(self, index, state, velocity, timestep, sigma, dt):
        """Save actual scheduler inputs without an additional model forward."""
        if index not in self.selected:
            return
        values = {
            "states": state,
            "old_velocities": velocity,
            "timesteps": torch.as_tensor(timestep),
            "sigmas": sigma,
            "dt": dt,
        }
        if not self.records:
            self.records = {
                key: torch.empty((self.count, *value.shape), dtype=torch.float32) for key, value in values.items()
            }
        for key, value in values.items():
            self.records[key][self.record_count].copy_(value.detach())
        self.record_count += 1

    def finish(self, endpoint):
        """Transfer the final endpoint once and return the recorded training fields."""
        if self.record_count != self.count:
            raise ValueError("ART rollout did not execute every requested training step.")
        return {**self.records, "endpoints": frozen_cpu(endpoint.float())}
