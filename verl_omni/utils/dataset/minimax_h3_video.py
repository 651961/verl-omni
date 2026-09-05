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
"""Required per-row MiniMax H3 output specifications.

The serving pipeline uses 32-aligned canvases, 17n+5 frames, 24 fps and
4--15 second outputs. Reject unaligned requests instead of letting the engine
silently change the dataset's target shape.
"""

from collections.abc import Mapping
from numbers import Integral
from typing import Any


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def resolve_video_spec(fields: Mapping[str, Any]) -> dict[str, int | float]:
    """Resolve mandatory resolution, num_frames and frame_rate without defaults."""
    if not isinstance(fields, Mapping):
        raise TypeError("Video specification must be a mapping.")
    extra = fields.get("extra_info", {})
    if not isinstance(extra, Mapping):
        raise TypeError("extra_info must be a mapping.")
    values = {}
    for key in ("resolution", "num_frames", "frame_rate"):
        value = extra.get(key)
        if key in fields:
            if value is not None and value != fields[key]:
                raise ValueError(f"{key} conflicts with extra_info.{key}.")
            value = fields[key]
        if value is None:
            raise ValueError(f"Missing required video field: extra_info.{key}.")
        values[key] = value

    resolution = values["resolution"]
    if not isinstance(resolution, str):
        raise ValueError("resolution must be a '<width>x<height>' string.")
    parts = resolution.lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(part.isascii() and part.isdecimal() for part in parts):
        raise ValueError(f"resolution must be a '<width>x<height>' string, got {resolution!r}.")
    width, height = (int(part) for part in parts)
    if min(width, height) <= 0 or width % 32 or height % 32:
        raise ValueError(f"MiniMax H3 width and height must be positive multiples of 32, got {resolution!r}.")
    if not 0.25 <= width / height <= 4.0:
        raise ValueError(f"MiniMax H3 aspect ratio must be between 1:4 and 4:1, got {resolution!r}.")
    for key, expected in (("width", width), ("height", height)):
        for source in (extra, fields):
            if key in source and source[key] != expected:
                raise ValueError(f"{key} conflicts with resolution.")

    num_frames = _positive_integer(values["num_frames"], "num_frames")
    frame_rate = values["frame_rate"]
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, int | float):
        raise ValueError(f"frame_rate must be numeric, got {frame_rate!r}.")
    if frame_rate != 24:
        raise ValueError(f"The MiniMax H3 rollout engine supports only frame_rate=24, got {frame_rate!r}.")
    if num_frames % 17 != 5:
        raise ValueError(f"MiniMax H3 num_frames must satisfy 17*n+5 (e.g. 107, 124, 141), got {num_frames}.")
    if not 4 <= num_frames / frame_rate <= 15:
        raise ValueError(f"MiniMax H3 output duration must be between 4 and 15 seconds, got {num_frames / frame_rate}.")
    return {"height": height, "width": width, "num_frames": num_frames, "frame_rate": float(frame_rate)}


def video_spec_metadata(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical dataset metadata for an explicitly specified target."""
    spec = resolve_video_spec(fields)
    return {
        "resolution": f"{spec['width']}x{spec['height']}",
        "num_frames": spec["num_frames"],
        "frame_rate": spec["frame_rate"],
    }
