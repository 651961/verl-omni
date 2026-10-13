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
"""Use the checked-in FLOPs estimators without optional distributed frameworks."""

import __future__

import ast
import os
import sys
import unittest
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/trainer/diffusion"))
from test_art_core_on_cpu import cache, data  # noqa: E402


def kernels():
    namespace = {
        "os": os,
        "torch": torch,
        "warnings": warnings,
        "is_device_available": lambda: False,
        "get_device_flops": lambda *a, **kw: 1000.0,
    }
    for filename in ("diffusion_flops_counter.py", "qwen_image21.py"):
        tree = ast.parse((ROOT / "verl_omni/utils/mfu" / filename).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom)]
        exec(compile(tree, filename, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class Qwen21MFUTests(unittest.TestCase):
    def test_single_stream_swiglu_block_causal_lengths_and_selected_steps(self):
        namespace = kernels()
        cfg = {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "in_channels": 4,
            "out_channels": 4,
            "context_in_dim": 8,
            "mlp_ratio": 3,
            "patch_size": 1,
        }
        counter = namespace["DiffusionFlopsCounter"]("QwenImage21Pipeline", cfg)
        packed = data.pack_art_caches([cache(seed=i) for i in range(2)], [0, 1], ["a", "a"])
        meta = namespace["collect_diffusion_flops_meta"](counter, packed)
        self.assertEqual(meta["num_timesteps"], 2)
        self.assertEqual(meta["latent_seqlens"], [4, 4])
        self.assertEqual(meta["prompt_seqlens"], [3, 3])
        self.assertEqual(meta["attention_pairs"], [6 + 4 * 7] * 2)
        self.assertEqual(meta["negative_projection_seqlens"], [0, 0])
        # d=8; block QKV/out=4d², three ratio-3 SwiGLU linears=9d².
        per_sample = 6 * (((4 + 9) * 64 + 4 * 8) * 7 + 4 * 8 * 4 + (8 * 8 + 64) * 3)
        per_sample += 6 * 2 * (256 * 8 + 6 * 64)
        per_sample += 12 * 1 * 8 * (6 + 4 * 7)
        estimated, peak = counter.estimate_flops(delta_time=1.0, **meta)
        self.assertAlmostEqual(estimated, per_sample * 2 * 2 / 1e12)
        self.assertEqual(peak, 1000.0)

    def test_reference_interleaving_and_cfg_are_counted_per_sample(self):
        namespace = kernels()
        cfg = {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "in_channels": 4,
            "context_in_dim": 8,
        }
        caches = [cache(8, references=4, negative=True, seed=i) for i in range(2)]
        for c in caches:
            c["conditions"]["inputs"]["img_mask"] = torch.tensor([[False, True, False, False, True, True]])
        packed = data.pack_art_caches(caches, [0, 1], ["a", "a"])
        counter = namespace["DiffusionFlopsCounter"]("QwenImage21Pipeline", cfg)
        meta = namespace["collect_diffusion_flops_meta"](counter, packed)
        self.assertEqual(meta["latent_seqlens"], [12, 12])
        self.assertEqual(meta["projection_seqlens"], [4, 4])
        self.assertEqual(meta["prompt_seqlens"], [3, 3])
        # text(1), image(4), text(2), target(8)
        self.assertEqual(meta["attention_pairs"], [1 + 4 * 5 + 2 * 5 + 3 + 8 * 15] * 2)
        cfg_flops, _ = counter.estimate_flops(delta_time=1.0, **meta)
        for key in ("negative_prompt_seqlens", "negative_projection_seqlens", "negative_attention_pairs"):
            meta[key] = [0, 0]
        single, _ = counter.estimate_flops(delta_time=1.0, **meta)
        self.assertGreater(cfg_flops, single)
        meta["num_timesteps"] = 1
        one_step, _ = counter.estimate_flops(delta_time=1.0, **meta)
        self.assertAlmostEqual(single, 2 * one_step)


if __name__ == "__main__":
    unittest.main()
