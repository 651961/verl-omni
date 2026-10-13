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

"""Compact shared ART transport fields after per-sample scoring."""

import torch


def compact_art_caches(caches):
    """Store identical group conditions/schedules once without mutating captures."""

    def equal(left, right):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, tuple | list) and isinstance(right, type(left)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    result = []
    for row, cache in enumerate(caches):
        compact = dict(cache)
        if row:
            for key in ("conditions", "timesteps", "sigmas", "dt"):
                if key in cache and equal(cache[key], caches[0][key]):
                    compact.pop(key)
        result.append(compact)
    return result
