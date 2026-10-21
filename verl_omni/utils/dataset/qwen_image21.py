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

"""Raw-image dataset: avoid RGB conversion and Qwen2.5-VL preprocessing."""

import copy
import re

from verl_omni.utils.dataset.rl_dataset import RLHFDataset


class QwenImage21Dataset(RLHFDataset):
    """Keep raw prompts and RGBA conditions for the model-specific agent loop."""

    def _build_messages(self, example, key):
        value = copy.deepcopy(example[key])
        messages = [{"role": "user", "content": value}] if isinstance(value, str) else list(value)
        images = list(example.get(self.image_key) or []) if key != self.negative_prompt_key else []
        offset = 0
        for message in messages:
            content = message["content"]
            if not isinstance(content, str) or "<image>" not in content:
                continue
            parts = []
            for part in re.split("(<image>)", content):
                if part == "<image>":
                    if offset >= len(images):
                        raise ValueError("Qwen-Image 2.1 dataset has more image placeholders than condition images.")
                    parts.append({"type": "image", "image": images[offset]})
                    offset += 1
                elif part:
                    parts.append({"type": "text", "text": part})
            message["content"] = parts
        if images and offset != len(images):
            raise ValueError("Qwen-Image 2.1 dataset must reference each images entry with one <image> placeholder.")
        return messages

    def __getitem__(self, item):
        row = dict(self.dataframe[item])
        row["raw_prompt"] = self._build_messages(row, self.prompt_key)
        if row.get(self.negative_prompt_key) is not None:
            row["raw_negative_prompt"] = self._build_messages(row, self.negative_prompt_key)
        info = row.get("extra_info") or {}
        row["index"] = info.get("index", item)
        row["extra_info"] = info
        row.pop(self.prompt_key, None)
        row.pop(self.image_key, None)
        return row
