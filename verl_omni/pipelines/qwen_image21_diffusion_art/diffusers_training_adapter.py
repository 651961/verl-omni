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

"""Native Qwen-Image 2.1 replay of ART's frozen rollout states."""

import json
from pathlib import Path

from diffusers import FlowMatchEulerDiscreteScheduler

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image21_diffusion_art.data import qwen21_art_inputs

__all__ = ["QwenImage21DiffusionART"]


@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="diffusion_art")
class QwenImage21DiffusionART(DiffusionModelBase):
    """Replay native Qwen-Image 2.1 conditions at frozen ART states."""

    @classmethod
    def validate_art_config(cls, config):
        """Check native Qwen21 dependencies before allocating workers."""
        from importlib.util import find_spec

        if (
            find_spec("vllm_omni.diffusion.models.qwen_image_21") is None
            or find_spec("diffusers.models.transformers.transformer_qwenimage21") is None
        ):
            raise ImportError("Qwen-Image 2.1 ART requires native QwenImage21 support in diffusers and vllm-omni.")

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Add the VLM config expected by ``AutoProcessor``.

        Qwen-Image checkpoints ship the processor metadata but omit the
        ``processor/config.json`` model-type marker that verl uses when it
        attaches the Qwen3-VL rotary-position helper.
        """
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            raise FileNotFoundError(f"Qwen-Image 2.1 processor directory not found: {processor_dir}")
        config_path = processor_dir / "config.json"
        if not config_path.is_file():
            config_path.write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
        return str(processor_dir)

    @classmethod
    def preserve_fp32_modules(cls):
        # FP32 master storage with BF16 FSDP compute, including norm/projection
        # islands inside each wrapping unit. Model ops may upcast internally.
        """Use BF16 compute uniformly over FP32 master parameters."""
        return False

    @classmethod
    def configure_trainable_params(cls, module, model_config):
        """Install Qwen21 kernels before FSDP and optimizer construction."""
        from verl_omni.pipelines.qwen_image21.ops import apply_flash_attention, apply_liger_rms_norms, apply_quack_rope

        apply_liger_rms_norms(module)
        apply_quack_rope()
        apply_flash_attention(module)

    @classmethod
    def build_scheduler(cls, model_config):
        """Load the checkpoint scheduler for engine initialization."""
        return FlowMatchEulerDiscreteScheduler.from_pretrained(model_config.local_path, subfolder="scheduler")

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device):
        # Training uses the captured, resolution-dependent schedule exclusively.
        """Initialize the engine scheduler; replay uses the captured schedule."""
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device, mu=1.0)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    ):
        """Reconstruct the selected step from the ART actor batch."""
        positive, negative, _ = qwen21_art_inputs(micro_batch, step)
        return positive, negative

    @classmethod
    def forward_and_sample_previous_step(cls, *args, **kwargs):
        """Reject the policy-gradient log-probability interface."""
        raise RuntimeError("ART replays ODE velocities; it never recomputes SDE log probabilities.")

    @classmethod
    def prepare_art_inputs(cls, micro_batch, step):
        """Return positive/negative model inputs and the target state."""
        return qwen21_art_inputs(micro_batch, step)

    @classmethod
    def trim_art_transport(cls, micro_batch):
        """Remove cross-sample padding before transferring a microbatch."""
        from .data import trim_art_transport

        return trim_art_transport(micro_batch)

    @classmethod
    def prepare_art_batch(cls, caches, rewards, uids, covariance_options):
        """Compute complete-group statistics and pad replay fields for transport."""
        from .data import pack_art_caches

        return pack_art_caches(caches, rewards, uids, covariance_options=covariance_options)

    @classmethod
    def art_loss_coefficients(cls, micro_batch, step):
        """Return the scalar schedule shared by all image tokens."""
        return micro_batch["art_sigmas"][:, step], micro_batch["art_dt"][:, step]

    @classmethod
    def forward_art_velocity(cls, module, micro_batch, step):
        """Predict the target velocity, including optional classifier-free guidance."""
        positive, negative, state = cls.prepare_art_inputs(micro_batch, step)
        n = state.shape[1]
        prediction = module(**positive)[0][:, -n:]
        if negative is not None:
            uncond = module(**negative)[0][:, -n:]
            scale = float(micro_batch["art_cfg_scale"][0])
            prediction = uncond + scale * (prediction - uncond)
        return prediction.float(), state
