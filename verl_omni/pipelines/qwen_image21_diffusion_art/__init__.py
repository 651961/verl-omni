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

from importlib.util import find_spec

from .diffusers_training_adapter import QwenImage21DiffusionART

__all__ = ["QwenImage21DiffusionART"]

# The pinned serving dependency predates Qwen-Image 2.1. Other models can
# still run; selecting ART validates its native dependencies before startup.
if find_spec("vllm_omni.diffusion.models.qwen_image_21") is not None:
    from .vllm_omni_rollout_adapter import QwenImage21DiffusionARTPipeline as QwenImage21DiffusionARTPipeline

    __all__.append("QwenImage21DiffusionARTPipeline")
