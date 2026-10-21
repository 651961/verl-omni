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
"""Create serialized CPU FSDP or HSDP DTensors in an isolated Gloo process group."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def save_rank(rank: int, roots: list[str], hsdp: bool) -> None:
    """Save genuine rank shards for tiny state dictionaries, sharing one Gloo startup."""
    world_size = 4 if hsdp else 2
    dist.init_process_group(
        "gloo", init_method=f"file://{Path(roots[0]) / 'rendezvous'}", rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2) if hsdp else (2,), mesh_dim_names=("ddp", "fsdp") if hsdp else ("fsdp",))
        for directory in roots:
            root = Path(directory)
            state = torch.load(root / "full.pt", weights_only=True)
            result = {}
            for key, tensor in sorted(state.items()):
                placement = Replicate() if tensor.ndim == 0 else Shard(1 if key == "columns" else 0)
                result[key] = distribute_tensor(tensor, mesh, [Replicate(), placement] if hsdp else [placement])
            torch.save(result, root / f"model_world_size_{world_size}_rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--hsdp", action="store_true")
    args = parser.parse_args()
    mp.spawn(save_rank, args=(args.roots, args.hsdp), nprocs=4 if args.hsdp else 2, join=True)
