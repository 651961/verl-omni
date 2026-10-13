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

"""Isolated framework-method tests, plus replay through a real tiny Qwen21 DiT.

AST loading removes optional framework imports, but executes the checked-in
method bodies. These tests do not replace Ray/TransferQueue/FSDP integration.
"""

import __future__

import ast
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_art_core_on_cpu import ROOT, cache, data, math


class TensorDict(dict):
    def __init__(self, value=None, batch_size=None):
        super().__init__(value or {})
        self.batch_size = batch_size


class DataProto:
    def __init__(self, batch, non_tensor_batch=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}


class TU:
    @staticmethod
    def get_tensordict(value):
        return TensorDict(value)

    @staticmethod
    def assign_non_tensor(value, **kwargs):
        value.update(kwargs)

    @staticmethod
    def get_non_tensor_data(value, key, default):
        return value.get(key, default)


@contextmanager
def isolated_art_imports(**modules):
    # Do not replace real framework packages globally during test collection.
    aliases = {
        "verl_omni.trainer.diffusion.art_data": data,
        "verl_omni.pipelines.model_base": types.SimpleNamespace(
            DiffusionModelBase=types.SimpleNamespace(
                get_class=lambda config: types.SimpleNamespace(prepare_art_batch=data.pack_art_caches)
            )
        ),
    }
    aliases.update({"verl_omni.trainer.diffusion." + name: value for name, value in modules.items()})
    for name in ("verl_omni", "verl_omni.trainer", "verl_omni.trainer.diffusion"):
        package = types.ModuleType(name)
        package.__path__ = []
        aliases[name] = package
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        # Restore only our aliases. Removing unrelated newly imported torch
        # modules would cause duplicate native-op registration on the next test.
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def extract(relative, names, namespace, class_methods=None):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = []
    for node in tree.body:
        if getattr(node, "name", None) not in names:
            continue
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
            node.bases = [ast.Name(id="object", ctx=ast.Load())]
            if class_methods is not None:
                node.body = [child for child in node.body if getattr(child, "name", None) in class_methods]
        nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, relative, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def adapter_class():
    return extract(
        "verl_omni/pipelines/qwen_image21_diffusion_art/diffusers_training_adapter.py",
        ["QwenImage21DiffusionART"],
        {"torch": torch, "tu": TU, "qwen21_art_inputs": data.qwen21_art_inputs, "get_device_id": lambda: "cpu"},
    )["QwenImage21DiffusionART"]


def engine_class():
    adapter = adapter_class()
    adapter.trim_art_transport = data.trim_art_transport
    cls = extract(
        "verl_omni/workers/engine/fsdp/diffusers_impl.py",
        ["ARTDiffusersFSDPEngine"],
        {"torch": torch, "tu": TU, "DiffusionModelBase": types.SimpleNamespace(get_class=lambda config: adapter)},
    )["ARTDiffusersFSDPEngine"]
    cls.model_config = types.SimpleNamespace()
    return cls


def tiny_qwen(num_layers=1):
    from diffusers import QwenImage21Transformer2DModel

    torch.manual_seed(71)
    return QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=8,
        context_in_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).to(torch.bfloat16)


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_art_imports())

    def test_real_qwen21_replay_t2i_edit_cfg_and_reference_padding(self):
        engine = engine_class()()
        engine.module = tiny_qwen()
        engine.get_data_parallel_group = lambda: None
        for n, refs, cfg in ((4, 0, False), (8, 4, True)):
            caches = [cache(n, refs, cfg, seed) for seed in range(2)]
            packed = data.pack_art_caches(caches, [0.0, 1.0], ["x", "x"])
            micro = {k: v[:1].clone() for k, v in packed.items()}
            micro["gradient_accumulation_steps"] = 2
            positive, negative, state = data.qwen21_art_inputs(micro, 0)
            with torch.no_grad():
                expected = engine.module(**positive)[0][:, -n:]
                if cfg:
                    uncond = engine.module(**negative)[0][:, -n:]
                    expected = uncond + 2.0 * (expected - uncond)
            # Use the expected guided BF16 prediction as the frozen rollout velocity.
            micro["art_old_velocities"][:, 0, :n] = expected.float()

            def loss_function(model_output, data, dp_group):
                result = math.art_loss(
                    model_output["current_velocity"],
                    data["old_velocity"],
                    data["sigma"],
                    data["direction"],
                    data["magnitude"],
                    ode_velocity_coefficient=data["dt"],
                    corrective_mask_enabled=False,
                )
                return result["per_item_loss"].mean() / data["gradient_accumulation_steps"], {}

            engine.module.zero_grad()
            loss, output = engine.forward_step(micro, loss_function, False, 0)
            torch.testing.assert_close(output["model_output"]["current_velocity"], expected.float(), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in engine.module.parameters()))
            with torch.no_grad():
                _, output = engine.forward_step(micro, None, True, 0)
            self.assertEqual(output["model_output"]["noise_pred"].shape, (1, n, 4))

    def test_staging_moves_only_one_step_and_keeps_loss_metadata(self):
        engine = engine_class()()
        packed = data.pack_art_caches([cache(seed=i) for i in range(2)], [0.0, 1.0], ["x", "x"])
        fields, shared = engine._prepare_timestep_staging(packed, "art_timesteps")
        self.assertEqual(set(fields), {"art_states", "art_old_velocities", "art_timesteps", "art_sigmas", "art_dt"})
        self.assertIn("art_directions", shared)
        self.assertIn("gradient_accumulation_steps", shared)
        self.assertNotIn("art_states", shared)

    def test_tq_restores_opaque_cache_and_pads_both_canvas_axes(self):
        caches = [cache(), cache(8, seed=1)]
        raw = {
            "uid": ["a", "b"],
            "responses": [torch.ones(4, 32, 64), torch.ones(4, 64, 32)],
            "extra_fields": [{"art_cache": c, "min_global_steps": 7} for c in caches],
        }
        tq = types.SimpleNamespace(kv_batch_get=Mock(return_value=raw))
        names = [
            "_unwrap_non_tensor_item",
            "_to_object_array",
            "_stack_field",
            "diffusion_tq_batch_to_dataproto",
            "sort_diffusion_tq_keys",
            "_parse_tq_key",
        ]
        namespace = extract(
            "verl_omni/trainer/diffusion/v1/tq_utils.py",
            names,
            {"np": np, "torch": torch, "tq": tq, "TensorDict": TensorDict, "DataProto": DataProto},
        )
        proto = namespace["diffusion_tq_batch_to_dataproto"](
            types.SimpleNamespace(keys=["a_0_0", "b_0_0"], partition_id="train")
        )
        self.assertEqual(proto.batch["responses"].shape, (2, 4, 64, 64))
        self.assertIs(proto.non_tensor_batch["art_cache"][0], caches[0])
        self.assertEqual(proto.non_tensor_batch["min_global_steps"].tolist(), [7, 7])
        self.assertEqual(proto.batch["responses"][0, :, 32:].count_nonzero(), 0)

    def test_serving_version_commits_after_successful_weight_stream(self):
        namespace = extract(
            "verl_omni/workers/rollout/vllm_rollout/utils.py",
            ["vLLMOmniColocateWorkerExtension"],
            {},
            class_methods={"set_pending_lora_peft_config", "_commit_art_policy_version"},
        )
        extension = namespace["vLLMOmniColocateWorkerExtension"]()
        calls = []
        clear = Mock()
        extension.model_runner = types.SimpleNamespace(
            pipeline=types.SimpleNamespace(set_published_policy_version=calls.append, clear_prefix_cache=clear)
        )
        extension.set_pending_lora_peft_config(global_steps=12)
        clear.assert_not_called()
        self.assertEqual(calls, [])
        extension._commit_art_policy_version()
        self.assertEqual(calls, [12])
        extension._commit_art_policy_version()
        self.assertEqual(calls, [12])

    def test_async_group_preserves_actual_versions_without_pinning(self):
        import asyncio
        import logging
        from unittest.mock import AsyncMock

        tq = types.SimpleNamespace(async_kv_put=AsyncMock())
        namespace = extract(
            "verl_omni/agent_loop/diffusion_agent_loop_tq.py",
            ["DiffusionAgentLoopWorkerTQ"],
            {"asyncio": asyncio, "tq": tq, "logger": logging.getLogger(__name__)},
            class_methods={"_run_prompt"},
        )
        worker = namespace["DiffusionAgentLoopWorkerTQ"]()
        worker.model_config = types.SimpleNamespace(algorithm="diffusion_art")
        worker.rollout_config = types.SimpleNamespace(n=2)
        worker._run_agent_loop = AsyncMock(
            side_effect=[
                types.SimpleNamespace(extra_fields={"art_cache": {"policy_version": version}}) for version in (1, 2)
            ]
        )
        worker._write_trajectories_to_tq = AsyncMock()
        asyncio.run(worker._run_prompt({"uid": "a", "agent_name": "art"}, {}, {"validate": False}, 0))
        self.assertEqual(worker._run_agent_loop.await_count, 2)
        for call in worker._run_agent_loop.await_args_list:
            self.assertNotIn("art_condition_group_id", call.args[0])
            self.assertNotIn("_art_group_client", call.kwargs)
        rows = worker._write_trajectories_to_tq.await_args.args[0]
        self.assertEqual([output.extra_fields["art_cache"]["policy_version"] for _, output in rows], [1, 2])
        self.assertEqual(tq.async_kv_put.await_args.kwargs["tag"]["status"], "finished")

    def test_reward_starts_before_other_samples_finish_generating(self):
        import asyncio
        import logging
        from unittest.mock import AsyncMock

        async def check():
            namespace = extract(
                "verl_omni/agent_loop/diffusion_agent_loop_tq.py",
                ["DiffusionAgentLoopWorkerTQ"],
                {
                    "asyncio": asyncio,
                    "tq": types.SimpleNamespace(async_kv_put=AsyncMock()),
                    "logger": logging.getLogger(__name__),
                },
                class_methods={"_run_prompt"},
            )
            worker = namespace["DiffusionAgentLoopWorkerTQ"]()
            worker.rollout_config = types.SimpleNamespace(n=2)
            reward_started, second_generation, finish_rollout, finish_reward = [asyncio.Event() for _ in range(4)]

            async def sample(sampling_params, session_id, **kwargs):
                if session_id == 1:
                    await finish_rollout.wait()
                    second_generation.set()
                else:
                    reward_started.set()
                await finish_reward.wait()
                return types.SimpleNamespace(extra_fields={})

            worker._run_agent_loop = sample
            worker._write_trajectories_to_tq = AsyncMock()
            task = asyncio.create_task(
                worker._run_prompt({"uid": "a", "agent_name": "art"}, {}, {"validate": False}, 0)
            )
            await asyncio.wait_for(reward_started.wait(), 2)
            self.assertFalse(second_generation.is_set())
            self.assertFalse(task.done())
            finish_rollout.set()
            await asyncio.wait_for(second_generation.wait(), 2)
            finish_reward.set()
            await asyncio.wait_for(task, 2)
            worker._write_trajectories_to_tq.assert_awaited_once()

        asyncio.run(check())

    def test_trainer_retains_inferred_architecture_for_adapter_dispatch(self):
        import json
        import os
        import tempfile
        from unittest.mock import patch

        class Config(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        namespace = extract(
            "verl_omni/trainer/diffusion/v1/trainer_base.py",
            ["PolicyGradientDiffusionTrainerV1"],
            {"os": os},
            class_methods={"_init_tokenizer"},
        )
        worker = namespace["PolicyGradientDiffusionTrainerV1"]()
        with tempfile.TemporaryDirectory() as path:
            Path(path, "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
            model = Config(path=path, tokenizer_path=None, architecture=None, algorithm="diffusion_art")
            worker.config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(model=model), data={})
            registry = Mock(return_value=types.SimpleNamespace(prepare_processor_files=lambda local: None))
            aliases = {
                "verl": types.ModuleType("verl"),
                "verl.utils": types.SimpleNamespace(hf_tokenizer=Mock(), hf_processor=Mock()),
                "verl_omni.utils.fs": types.SimpleNamespace(
                    resolve_model_local_dir=lambda path, **kwargs: path,
                    resolve_diffusion_tokenizer_path=lambda path: path,
                ),
                "verl_omni.pipelines.model_base": types.SimpleNamespace(
                    DiffusionModelBase=types.SimpleNamespace(get_class_by_name=registry)
                ),
            }
            with patch.dict(sys.modules, aliases):
                worker._init_tokenizer()
            self.assertEqual(model.architecture, "QwenImage21Pipeline")
            registry.assert_called_once_with("QwenImage21Pipeline", "diffusion_art", None)

    def test_registered_loss_uses_complete_groups_and_optional_reference(self):
        from collections import defaultdict
        from types import SimpleNamespace

        namespace = extract(
            "verl_omni/trainer/diffusion/diffusion_algos.py",
            ["DiffusionARTLoss"],
            {"torch": torch, "defaultdict": defaultdict, "__package__": "_art_cpu_kernels"},
        )
        loss_class = namespace["DiffusionARTLoss"]
        caches = [cache(seed=i) for i in range(4)]
        batch = DataProto(TensorDict(), {"art_cache": np.array(caches, dtype=object), "uid": np.array(["a"] * 4)})
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(n=4), actor=SimpleNamespace(diffusion_loss={}), model=SimpleNamespace()
            )
        )
        result = loss_class.prepare_actor_batch(batch, torch.tensor([[0.0], [1.0], [2.0], [3.0]]), config)
        self.assertNotIn("art_cache", result.non_tensor_batch)
        self.assertEqual(result.batch["art_states"].shape[0], 4)
        v = torch.ones(1, 4, 4, requires_grad=True)
        actor_config = SimpleNamespace(
            diffusion_loss={"art_loss_options": {"corrective_mask_enabled": False}}, use_kl_loss=True, kl_loss_coef=0.3
        )
        loss, metrics = loss_class.compute_loss(
            config=actor_config,
            current_velocity=v,
            old_velocity=torch.zeros_like(v),
            state=torch.zeros_like(v),
            sigma=torch.tensor([0.5]),
            dt=torch.tensor([-0.5]),
            direction=-torch.ones_like(v),
            magnitude=torch.ones(1),
            reference_prediction=torch.zeros_like(v),
        )
        torch.testing.assert_close(loss, torch.tensor(0.3))  # zero policy residual, reference only
        loss.backward()
        self.assertTrue(v.grad.abs().sum() > 0)
        self.assertIn("art_reference_mse", metrics)

    def test_bf16_shard_export_matches_publication_without_touching_fp32(self):
        from types import SimpleNamespace

        class FakeDTensor:
            pass

        namespace = extract(
            "verl_omni/workers/engine/fsdp/diffusers_impl.py",
            ["DiffusersFSDPEngine"],
            {
                "torch": torch,
                "DTensor": FakeDTensor,
                "fsdp_version": lambda module: 2,
                "convert_weight_keys": lambda params, module: params,
                "get_device_id": lambda: "cpu",
                "ShardSpec": SimpleNamespace(from_param=lambda value: SimpleNamespace(shape=value.shape)),
            },
            class_methods={"get_per_tensor_param_shard"},
        )
        engine = namespace["DiffusersFSDPEngine"]()
        engine.module = torch.nn.Linear(3, 2)
        engine._is_offload_param = False
        engine.model_config = SimpleNamespace(algorithm="diffusion_art")
        originals = {key: value.clone() for key, value in engine.module.state_dict().items()}
        stream, peft = engine.get_per_tensor_param_shard()
        exported = list(stream)
        self.assertIsNone(peft)
        for name, value, spec in exported:
            original = originals[name.removeprefix("transformer.")]
            torch.testing.assert_close(value, original.bfloat16().reshape(-1), rtol=0, atol=0)
        for name, value in engine.module.state_dict().items():
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, originals[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
