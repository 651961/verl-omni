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

"""Transport and weight-publication policy shared by ART adapters."""

import torch


class DiffusionARTMixin:
    """Retain frozen replay targets while avoiding duplicate reward and media payloads."""

    share_rollout_conditions = False

    @classmethod
    def preserve_fp32_modules(cls) -> bool:
        """Use BF16 FSDP compute over FP32 master parameters."""
        return False

    @classmethod
    def weight_sync_dtype(cls) -> torch.dtype:
        """Match the native BF16 serving weights for both full and delta publication."""
        return torch.bfloat16

    @classmethod
    def reward_extra_fields(cls, extra: dict) -> dict:
        """Keep the trajectory cache out of reward RPCs."""
        return {key: value for key, value in extra.items() if key != "art_cache"}

    @classmethod
    def prepare_art_rewards(cls, batch, reward_tensor):
        """Use one scalar score per trajectory for single-objective ART."""
        return reward_tensor.detach().float().reshape(len(reward_tensor), -1).sum(-1)

    @classmethod
    def prepare_rollout_row(cls, field: dict, extra: dict, *, row_index: int, validate: bool, config) -> dict:
        """Store selected ODE states and publish the version that generated them."""
        metadata = field["extra_fields"]
        metadata["response_shape"] = tuple(field["responses"].shape)
        cache = extra.get("art_cache")
        if cache is not None:
            if not validate:
                field["art_cache"] = dict(cache)
                if cls.share_rollout_conditions and row_index:
                    for key in ("conditions", "timesteps", "sigmas", "dt"):
                        field["art_cache"].pop(key)
            metadata["min_global_steps"] = metadata["max_global_steps"] = int(cache["policy_version"])
        if not validate and "rm_scores" in field and not config.trainer.get("rollout_data_dir"):
            field = {
                key: value for key, value in field.items() if key in {"uid", "rm_scores", "art_cache", "extra_fields"}
            }
            metadata.pop("img_shapes", None)
        return field
