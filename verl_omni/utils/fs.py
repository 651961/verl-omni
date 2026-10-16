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

import os

from verl.utils.fs import copy_to_local

__all__ = ["resolve_model_local_dir", "resolve_diffusion_tokenizer_path"]


def resolve_model_local_dir(path: str, use_shm: bool = False) -> str:
    """Resolve ``path`` to an on-disk directory."""
    local_path = copy_to_local(path, use_shm=use_shm)
    if not os.path.isdir(local_path):
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(path)
    return local_path


def resolve_diffusion_tokenizer_path(local_path: str) -> str:
    """Find the tokenizer in native diffusers checkpoint layouts."""
    tokenizer_path = os.path.join(local_path, "tokenizer")
    if os.path.isdir(tokenizer_path):
        return tokenizer_path
    # Qwen-Image-2.1 bundles its tokenizer with the multimodal processor.
    processor_path = os.path.join(local_path, "processor")
    if os.path.isfile(os.path.join(processor_path, "tokenizer_config.json")):
        return processor_path
    return local_path
