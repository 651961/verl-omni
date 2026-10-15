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
"""Activation checkpointing for diffusion models."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


def enable_gradient_checkpointing(module: torch.nn.Module, save_ops: list[str]) -> None:
    """Enable block checkpointing, optionally saving selected torch operator outputs."""
    if not save_ops:
        module.enable_gradient_checkpointing()
        return

    ops = []
    for name in save_ops:
        try:
            namespace, operator, overload = name.split(".")
            op = getattr(getattr(getattr(torch.ops, namespace), operator), overload)
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"Invalid gradient_checkpointing_save_ops entry {name!r}; expected a registered "
                "torch operator such as 'aten.mm.default'."
            ) from exc
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError(f"gradient_checkpointing_save_ops entry {name!r} is not a torch operator overload.")
        ops.append(op)

    module.enable_gradient_checkpointing(
        gradient_checkpointing_func=partial(
            checkpoint,
            use_reentrant=False,
            context_fn=partial(create_selective_checkpoint_contexts, ops),
        )
    )
