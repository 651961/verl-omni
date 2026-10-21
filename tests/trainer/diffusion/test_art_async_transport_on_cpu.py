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
"""ART transport, topology and observability regressions without Ray or GPUs."""

import __future__

import ast
import asyncio
import copy
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from tensordict import TensorDict as ReplayBatch
from test_art_core_on_cpu import ROOT, cache, data, transport
from test_art_wiring_on_cpu import TU, DataProto, TensorDict, engine_class, extract, isolated_art_imports, tiny_qwen

from verl_omni.pipelines.diffusion_art.common import DiffusionARTMixin
from verl_omni.pipelines.rollout_media import (
    ARTIFACT_CONTEXT,
    ARTIFACT_PREFIX,
    ARTIFACT_SPECS,
    PREVIEW_ARTIFACT,
    PRIMARY_ARTIFACT,
)


def tq_reader(tq):
    return extract(
        "verl_omni/trainer/diffusion/v1/tq_utils.py",
        [
            "_unwrap_non_tensor_item",
            "_to_object_array",
            "_stack_field",
            "diffusion_tq_batch_to_dataproto",
            "sort_diffusion_tq_keys",
            "_parse_tq_key",
            "diffusion_metric_tq_fields",
        ],
        {
            "np": np,
            "torch": torch,
            "tq": tq,
            "TensorDict": TensorDict,
            "DataProto": DataProto,
            "MEDIA_ARTIFACT_FIELD_PREFIX": ARTIFACT_PREFIX,
        },
    )


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_keeps_only_replay_inputs_and_restores_shared_group_fields(self):
        with isolated_art_imports():
            tq = SimpleNamespace(async_kv_batch_put=AsyncMock())
            cls = extract(
                "verl_omni/agent_loop/diffusion_agent_loop_tq.py",
                ["DiffusionAgentLoopWorkerTQ"],
                {
                    "torch": torch,
                    "tq": tq,
                    "list_of_dict_to_tensordict": lambda rows: rows,
                    "ARTIFACT_CONTEXT": ARTIFACT_CONTEXT,
                    "ARTIFACT_SPECS": ARTIFACT_SPECS,
                    "PRIMARY_ARTIFACT": PRIMARY_ARTIFACT,
                    "PREVIEW_ARTIFACT": PREVIEW_ARTIFACT,
                },
                class_methods={"_write_trajectories_to_tq"},
            )["DiffusionAgentLoopWorkerTQ"]
            worker = cls()
            worker.model_adapter = type("QwenARTAdapter", (DiffusionARTMixin,), {"share_rollout_conditions": True})
            worker.model_config = SimpleNamespace(algorithm="diffusion_art", architecture="QwenImage21Pipeline")
            worker.config = SimpleNamespace(trainer={})
            worker.rollout_config = SimpleNamespace(max_num_seqs=1, engine_kwargs={})
            caches = [cache(seed=i) for i in range(2)]
            caches[1]["conditions"] = copy.deepcopy(caches[0]["conditions"])
            outputs = [
                (
                    i,
                    SimpleNamespace(
                        prompt_ids=torch.ones(1, 4096, dtype=torch.long),
                        response_diffusion_output=torch.ones(1, 4, 32, 32, dtype=torch.uint8),
                        response_logprobs=None,
                        reward_score=float(i),
                        num_turns=1,
                        extra_fields={"art_cache": c, "reward_extra_info": {"quality": float(i)}},
                    ),
                )
                for i, c in enumerate(caches)
            ]
            await worker._write_trajectories_to_tq(
                outputs,
                uid="a",
                trajectory={"step": 2},
                validate=False,
                raw_prompt="draw",
                reward_model={"large": "metadata"},
                data_source="test",
            )
            rows = tq.async_kv_batch_put.await_args.kwargs["fields"]
            self.assertEqual(set(rows[0]), {"uid", "rm_scores", "art_cache", "extra_fields"})
            self.assertNotIn("art_cache", rows[0]["extra_fields"])
            for key in ("conditions", "timesteps", "sigmas", "dt"):
                self.assertIn(key, rows[0]["art_cache"])
                self.assertNotIn(key, rows[1]["art_cache"])
                self.assertIn(key, caches[1])  # captures were not mutated
            stored = {k: [row[k] for row in rows] for k in rows[0]}
            namespace = tq_reader(SimpleNamespace(kv_batch_get=Mock(return_value=stored)))
            meta = SimpleNamespace(keys=["a_0_0", "a_1_0"], partition_id="train")
            proto = namespace["diffusion_tq_batch_to_dataproto"](meta)
            restored = transport.restore_shared_conditions(list(proto.non_tensor_batch["art_cache"]), ["a", "a"])
            self.assertIs(restored[0]["conditions"], restored[1]["conditions"])
            actual = data.pack_art_caches(list(restored), [0, 1], ["a", "a"])
            expected = data.pack_art_caches(caches, [0, 1], ["a", "a"])
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            projection = namespace["diffusion_metric_tq_fields"]("direct_preference")
            self.assertNotIn("art_cache", projection)

            # Validation keeps real images for scoring/logging, but no trajectory.
            tq.async_kv_batch_put.reset_mock()
            await worker._write_trajectories_to_tq(outputs, uid="v", trajectory={"step": 2}, validate=True)
            val = tq.async_kv_batch_put.await_args.kwargs["fields"][0]
            self.assertIn("responses", val)
            self.assertNotIn("art_cache", val)

            # Explicit training image dumps retain their required fields.
            worker.config.trainer["rollout_data_dir"] = "/tmp/art-dump"
            tq.async_kv_batch_put.reset_mock()
            await worker._write_trajectories_to_tq(outputs, uid="d", trajectory={"step": 2}, validate=False)
            self.assertIn("responses", tq.async_kv_batch_put.await_args.kwargs["fields"][0])

    def test_shared_conditions_require_complete_group_reads(self):
        partial = cache()
        for key in ("conditions", "timesteps", "sigmas", "dt"):
            partial.pop(key)
        tq = SimpleNamespace(kv_batch_get=Mock(return_value={"uid": ["a"], "art_cache": [partial]}))
        with self.assertRaisesRegex(ValueError, "complete prompt group"):
            proto = tq_reader(tq)["diffusion_tq_batch_to_dataproto"](
                SimpleNamespace(keys=["a_1_0"], partition_id="train")
            )
            transport.restore_shared_conditions(list(proto.non_tensor_batch["art_cache"]), ["a"])

    def test_default_microbatch_trims_padding_but_reference_keeps_output_shape(self):
        from verl.utils import tensordict_utils as tu

        with isolated_art_imports():
            engine = engine_class()()
            batch = ReplayBatch(
                data.pack_art_caches(
                    [cache(4, seed=i) for i in range(2)] + [cache(8, seed=i) for i in range(2)],
                    [0, 1, 0, 1],
                    ["a", "a", "b", "b"],
                ),
                batch_size=[4],
            )
            with patch.dict(engine.forward_step.__globals__, tu=tu):
                micro = engine._prepare_microbatch_transport(batch[:1])
                self.assertEqual(micro["art_states"].shape[2], 4)
                self.assertEqual(tu.get_non_tensor_data(micro, "art_output_token_count", default=None), 8)
                engine.module = tiny_qwen()
                with torch.no_grad():
                    _, output = engine.forward_step(micro, None, True, 0)
                self.assertEqual(output["model_output"]["noise_pred"].shape, (1, 8, 4))
                self.assertEqual(output["model_output"]["noise_pred"][:, 4:].count_nonzero(), 0)


class ConfigurationTests(unittest.TestCase):
    def test_native_tokenizer_layouts(self):
        resolve = extract("verl_omni/utils/fs.py", ["resolve_diffusion_tokenizer_path"], {"os": os})[
            "resolve_diffusion_tokenizer_path"
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve(directory), directory)
            (root / "processor").mkdir()
            self.assertEqual(resolve(directory), directory)
            (root / "processor/tokenizer_config.json").write_text("{}")
            self.assertEqual(resolve(directory), str(root / "processor"))
            (root / "tokenizer").mkdir()
            self.assertEqual(resolve(directory), str(root / "tokenizer"))

    def test_generation_indices_are_unique_across_exact_refill_chunks(self):
        cls = extract(
            "verl_omni/trainer/diffusion/v1/trainer_base.py",
            ["PolicyGradientDiffusionTrainerV1"],
            {"np": np, "uuid": uuid, "tu": TU},
            class_methods={"_fetch_one_gen_batch"},
        )["PolicyGradientDiffusionTrainerV1"]
        worker = cls()
        worker._gen_batch_seq = 0
        worker._rollout_seed_offset = 0
        worker.train_dataloader_it = None
        worker.train_dataloader = [{"raw_prompt": ["a"]}, {"raw_prompt": ["b"]}]
        rows = [worker._fetch_one_gen_batch() for _ in range(4)]
        self.assertEqual([int(r["index"][0]) for r in rows], [0] * 4)
        self.assertEqual([int(r["_rollout_seed_global_idx"][0]) for r in rows], [0, 1, 2, 3])

    def test_recipe_node_topology_and_hsdp(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python3"
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            base = {
                "PATH": directory + ":" + os.environ["PATH"],
                "ART_MODEL_PATH": "/model path/Qwen-Image-2.1",
                "TRAIN_FILE": "train.parquet",
                "VAL_FILE": "val.parquet",
                "REWARD_FUNCTION_PATH": "reward.py",
                "REWARD_FUNCTION_NAME": "score",
            }
            for total in (8, 16, 32):
                env = {**base, "ACTOR_GPUS": str(total), "ROLLOUT_GPUS": "16"}
                result = subprocess.run(
                    ["bash", str(ROOT / "examples/diffusionart_trainer/qwen_image21/run_qwen_image21.sh")],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                arguments = dict(arg.split("=", 1) for arg in result.stdout.splitlines() if "=" in arg)
                self.assertEqual(arguments["trainer.nnodes"], str(total // 8))
                self.assertEqual(arguments["trainer.n_gpus_per_node"], "8")
                self.assertEqual(arguments["actor_rollout_ref.actor.fsdp_config.fsdp_size"], "8")
                self.assertEqual(arguments["actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size"], "1")
                self.assertEqual(arguments["actor_rollout_ref.rollout.nnodes"], "2")
                self.assertEqual(arguments["actor_rollout_ref.rollout.n_gpus_per_node"], "8")
                self.assertEqual(arguments["data.val_batch_size"], "8")

    def test_weight_sync_returns_parent_metrics(self):
        class Parent:
            async def update_weights(self, global_steps=None):
                return {"sync/bytes": 1024, "sync/step": global_steps}

        tree = ast.parse((ROOT / "verl_omni/workers/checkpoint_engine.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniCheckpointEngineManager")
        node.body = [n for n in node.body if getattr(n, "name", None) == "update_weights"]
        namespace = {"CheckpointEngineManager": Parent, "auto_await": lambda f: f}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "sync", "exec", flags=__future__.annotations.compiler_flag
            ),
            namespace,
        )
        manager = namespace["OmniCheckpointEngineManager"]()
        manager.backend = "naive"
        self.assertEqual(asyncio.run(manager.update_weights(7)), {"sync/bytes": 1024, "sync/step": 7})


if __name__ == "__main__":
    unittest.main()
