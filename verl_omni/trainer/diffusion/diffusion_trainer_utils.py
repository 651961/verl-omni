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
"""Shared helpers for diffusion Ray trainers."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.distillation import is_distillation_enabled
from verl.utils import tensordict_utils as tu

from verl_omni.utils.ragged_tensors import pack_ragged_tensors, unpack_ragged_tensors


def _to_diffusion_worker_tensordict(batch: DataProto):
    """Project a driver batch for workers, preserving dense tensor storage."""
    worker_batch = batch.to_tensordict()
    worker_batch.pop("responses", None)
    worker_batch.pop("audio", None)
    ragged_dims = dict(batch.meta_info.get("ragged_tensor_dims", {}))
    for key in batch.non_tensor_batch:
        if f"{key}_ragged_dim" in batch.meta_info:
            ragged_dims[key] = batch.meta_info[f"{key}_ragged_dim"]
    for key, ragged_dim in ragged_dims.items():
        if key not in batch.non_tensor_batch:
            continue
        values = list(batch.non_tensor_batch[key])
        if not all(isinstance(value, torch.Tensor) and 1 <= ragged_dim <= value.ndim for value in values):
            raise ValueError(f"Ragged {key} requires one tensor per sample with axis {ragged_dim}.")
        worker_batch[key] = pack_ragged_tensors(values, ragged_dim)
    if ragged_dims:
        tu.assign_non_tensor(worker_batch, ragged_tensor_dims=ragged_dims)
    return worker_batch


def _diffusion_outputs_to_dataproto(output, field_map: dict[str, str], model_config) -> DataProto:
    """Rename policy outputs and restore variable tensors to driver-side objects.

    Each output carries its own axis metadata so old/ref/teacher results can be
    unioned independently without conflicting with rollout metadata.
    """
    from verl_omni.pipelines.model_base import DiffusionModelBase

    tensors, non_tensors, metadata = {}, {}, {}
    for source, destination in field_map.items():
        value = tu.get(output, source)
        if value is None:
            continue
        value = value.float()
        if value.is_nested:
            ragged_dim = DiffusionModelBase.get_class(model_config).ragged_model_output_dims()[source]
            samples = unpack_ragged_tensors(value, ragged_dim)
            objects = np.empty(len(samples), dtype=object)
            objects[:] = [sample.cpu() for sample in samples]
            non_tensors[destination] = objects
            metadata[f"{destination}_ragged_dim"] = ragged_dim
        else:
            tensors[destination] = value
    result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=metadata)
    if result.batch is None:
        # DataProto.union needs a TensorDict even for object-only teacher outputs.
        result.batch = TensorDict({}, batch_size=[len(result)])
    return result


OLD_POLICY_DECAY_SCHEDULES = {
    "copy": (0, 0.0, 0.0),
    "linear_to_0_5": (0, 0.001, 0.5),
    "delayed_linear_to_0_999": (75, 0.0075, 0.999),
}


def old_policy_decay(step: int, schedule: str) -> float:
    """Return the old-policy LoRA EMA decay for a named DiffusionNFT schedule.

    The decay is used as ``old <- decay * old + (1 - decay) * current`` when refreshing
    the rollout adapter. The schedules mirror the reference DiffusionNFT ``return_decay``
    helper: ``copy`` hard-copies the current adapter, ``linear_to_0_5`` ramps from 0 to
    0.5, and ``delayed_linear_to_0_999`` waits 75 steps before ramping to 0.999.
    """
    if schedule in OLD_POLICY_DECAY_SCHEDULES:
        warmup_steps, ramp_rate, max_decay = OLD_POLICY_DECAY_SCHEDULES[schedule]
    else:
        raise ValueError(f"Unsupported old_policy_decay_schedule: {schedule}")
    return 0.0 if step < warmup_steps else min((step - warmup_steps) * ramp_rate, max_decay)


def worker_group_port_ranges(master_port_range: Optional[Sequence[int]], num_groups: int) -> list[Optional[list[int]]]:
    """Slice a rendezvous port range into one disjoint sub-range per worker group.

    Ports are only bound at ``init_model``, after every group has been spawned, so groups
    sharing a range would all pick its first free port.
    """
    if master_port_range is None:
        return [None] * num_groups
    lo, hi = (int(port) for port in master_port_range)
    stride = (hi - lo) // num_groups
    if stride < 1:
        raise ValueError(
            f"trainer.ray_master_port_range={master_port_range} has fewer ports than worker groups ({num_groups})."
        )
    return [[lo + i * stride, hi if i == num_groups - 1 else lo + (i + 1) * stride] for i in range(num_groups)]


def validate_distillation_config(config) -> None:
    """Cross-check the distillation switch against the losses that consume teacher outputs."""
    actor = config.actor_rollout_ref.actor
    distill_active = actor.diffusion_loss.get("loss_mode", "flow_grpo") == "distill_kl" or actor.use_distill_loss
    enabled = is_distillation_enabled(config.get("distillation"))
    if enabled and not distill_active:
        raise ValueError(
            "distillation.enabled=true but no distillation loss is active; set "
            "actor.diffusion_loss.loss_mode=distill_kl or actor.use_distill_loss=true."
        )
    if distill_active and not enabled:
        raise ValueError(
            "A distillation loss is active but no teacher is configured; set distillation.enabled=true "
            "and distillation.teacher_models.teacher_model.model_path."
        )
    if enabled and actor.use_distill_loss and actor.distill_loss_mode != "distill_kl":
        raise NotImplementedError(
            f"The teacher runtime produces teacher_prev_sample_mean, which only distill_kl consumes, "
            f"but got distill_loss_mode={actor.distill_loss_mode!r} (distill_fm_mse has no producer here)."
        )
    if enabled and config.algorithm.trainer_type != "policy_gradient":
        raise NotImplementedError("Diffusion distillation requires algorithm.trainer_type=policy_gradient.")


class NoOpCheckpointManager:
    """Checkpoint-engine facade used when training does not start rollout replicas."""

    def update_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def sleep_replicas(self) -> None:
        return None
