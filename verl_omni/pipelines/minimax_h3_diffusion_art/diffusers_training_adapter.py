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

"""MiniMax H3 replay adapter for the shared DiffusionART engine."""

from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.model_base import DiffusionModelBase

from .capture import flatten_targets
from .data import h3_art_coefficients, h3_art_inputs, pack_h3_art_caches, trim_h3_art_transport

__all__ = ["MiniMaxH3DiffusionART"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_art")
class MiniMaxH3DiffusionART(MiniMaxH3DiffusionNFT):
    """Replay both native Euler clocks and target-only video/audio velocities."""

    @classmethod
    def preserve_fp32_modules(cls):
        """Use BF16 FSDP compute over FP32 master parameters."""
        return False

    @classmethod
    def validate_art_config(cls, config):
        """Reject runtime paths that bypass request-local native trajectory capture."""
        rollout = config.actor_rollout_ref.rollout
        if rollout.step_execution or rollout.max_num_seqs != 1:
            raise ValueError("H3 ART requires rollout.step_execution=false and max_num_seqs=1.")
        if rollout.agent.default_agent_loop != "minimax_h3_art_agent":
            raise ValueError("H3 ART requires rollout.agent.default_agent_loop=minimax_h3_art_agent.")

    @classmethod
    def prepare_art_batch(cls, caches, rewards, uids, covariance_options):
        """Compute group covariance before dispatching padded samples to actor ranks."""
        return pack_h3_art_caches(caches, rewards, uids, covariance_options=covariance_options)

    @classmethod
    def trim_art_transport(cls, micro_batch):
        """Trim a single sample's transport padding before device transfer."""
        return trim_h3_art_transport(micro_batch)

    @classmethod
    def prepare_art_inputs(cls, micro_batch, step):
        """Reconstruct native packed model inputs at a captured ODE state."""
        return h3_art_inputs(micro_batch, step)

    @classmethod
    def art_loss_coefficients(cls, micro_batch, step):
        """Use the video or audio clock for each target feature."""
        return h3_art_coefficients(micro_batch, step)

    @classmethod
    def forward_art_velocity(cls, module, micro_batch, step):
        """Negate native data-minus-noise predictions into ART's flow convention."""
        inputs, _, state = cls.prepare_art_inputs(micro_batch, step)
        video, audio = module(**inputs)
        video_mask = micro_batch["art_h3_video_mask"][0, : video.shape[1]].bool()
        audio_mask = micro_batch["art_h3_audio_mask"][0, : audio.shape[1]].bool()
        prediction = -flatten_targets(video[0, video_mask].float(), audio[0, audio_mask].float())
        return prediction.unsqueeze(0), state
