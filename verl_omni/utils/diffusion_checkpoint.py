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

import logging
from contextvars import ContextVar
from fnmatch import fnmatchcase
from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

logger = logging.getLogger(__name__)


def enable_gradient_checkpointing(module: torch.nn.Module, save_ops: list[str]) -> None:
    """Save selected operator outputs and GEMMs inside named Linear modules.

    Entries can be registered ``namespace.operator.overload`` names or
    case-sensitive glob patterns over named ``nn.Linear`` module paths. Linear
    selectors save the module's GEMMs, including biased projections, without
    selecting equal-shaped projections elsewhere. Hooks track execution scope
    rather than parameter identity, so FSDP can replace parameter views between
    forward and checkpoint recomputation.
    """
    ops = []
    linears = None
    selected = {}
    for name in save_ops:
        try:
            namespace, operator, overload = name.split(".")
            op = getattr(getattr(getattr(torch.ops, namespace), operator), overload)
        except (AttributeError, ValueError):
            op = None
        if isinstance(op, torch._ops.OpOverload):
            ops.append(op)
            continue
        if linears is None:
            linears = {path: child for path, child in module.named_modules() if isinstance(child, torch.nn.Linear)}
        matches = {path: child for path, child in linears.items() if fnmatchcase(path, name)}
        if not matches:
            raise ValueError(
                f"Invalid gradient_checkpointing_save_ops entry {name!r}: it is not a registered operator "
                "overload and matched no Linear modules. Use namespace.operator.overload or a full Linear path/glob."
            )
        selected.update(matches)

    for handle in module.__dict__.get("_sac_linear_hooks", []):
        handle.remove()
    module._sac_linear_hooks = []

    if not ops and not selected:
        module.enable_gradient_checkpointing()
        return

    context_fn = partial(create_selective_checkpoint_contexts, ops)
    if selected:
        depth = ContextVar("sac_linear_depth", default=0)
        linear_ops = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default}

        def enter_linear(_module, _args):
            depth.set(depth.get() + 1)

        def leave_linear(_module, _args, _output):
            depth.set(depth.get() - 1)

        def policy_fn(_ctx, op, *args, **kwargs):
            if op in ops or (depth.get() > 0 and op in linear_ops):
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
        for child in selected.values():
            module._sac_linear_hooks.append(child.register_forward_pre_hook(enter_linear))
            # Non-reentrant checkpointing can stop inside Linear.forward().
            # Always unwind scope, including on early stop and exceptions.
            module._sac_linear_hooks.append(child.register_forward_hook(leave_linear, always_call=True))
        logger.info("SAC saves GEMM outputs in %d Linear modules selected by %s", len(selected), save_ops)

    module.enable_gradient_checkpointing(
        gradient_checkpointing_func=partial(
            checkpoint,
            use_reentrant=False,
            context_fn=context_fn,
        )
    )
