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

"""Raw prompt and RGBA condition handling for native Qwen-Image 2.1."""

import base64
import os
from io import BytesIO
from urllib.request import urlopen

from PIL import Image


def messages_to_text(messages):
    """Join text content from dataset messages."""
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(item["text"] for item in content if item.get("type") == "text")
    return "\n".join(parts)


def load_rgba(value):
    """Load a condition image without dropping its alpha channel."""
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = BytesIO(value["bytes"])
        else:
            value = value.get("image", value.get("path", value.get("url")))
            return load_rgba(value)
    if isinstance(value, str | os.PathLike):
        value = os.fspath(value)
        if value.startswith("data:image/"):
            value = BytesIO(base64.b64decode(value.split(",", 1)[1]))
        elif value.startswith(("https://", "http://")):
            with urlopen(value, timeout=30) as response:
                value = BytesIO(response.read())
        elif value.startswith("file://"):
            value = value[7:]
    with Image.open(value) as image:
        return image.convert("RGBA")
