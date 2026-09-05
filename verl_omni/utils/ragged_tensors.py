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
"""Transport variable tensors with a canonical jagged axis across worker RPCs."""

import torch


def pack_ragged_tensors(tensors: list[torch.Tensor], ragged_dim: int) -> torch.Tensor:
    """Move the variable axis after batch so chunk/concat/serialization preserve it.

    ``ragged_dim`` describes the original layout including its batch dimension.
    The pinned verl helpers reconstruct jagged tensors along axis one; changing
    only their private ``_ragged_idx`` breaks serialization for higher axes.
    Keep the transport layout canonical and restore model axes at the boundary.
    """
    return torch.nested.as_nested_tensor(
        [tensor.movedim(ragged_dim - 1, 0).contiguous() for tensor in tensors], layout=torch.jagged
    )


def unpack_ragged_tensors(tensor: torch.Tensor, ragged_dim: int) -> list[torch.Tensor]:
    """Restore each sample's model layout from canonical jagged transport."""
    return [item.movedim(0, ragged_dim - 1).contiguous() for item in tensor.unbind()]
