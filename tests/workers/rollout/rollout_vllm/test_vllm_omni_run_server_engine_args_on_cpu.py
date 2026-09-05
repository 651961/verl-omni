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
"""``run_server`` must not forward the default fault-tolerance config to AsyncOmni.

``asdict(OmniEngineArgs)`` serializes the default ``FaultToleranceConfig``
instance into a plain dict, and vLLM's ``EngineArgs.__post_init__`` treats any
dict as an explicit ``--fault-tolerance-config``, auto-setting
``enable_fault_tolerance=True`` — engine creation then fails with
"Fault tolerance requires external load balancer mode" under verl's internal
load balancer. The strip was dropped as dead code during the 0.28 pin bump and
resurrected after the failure fired on GPU.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")


@pytest.fixture(autouse=True)
def _skip_model_resolution(monkeypatch):
    monkeypatch.setattr("vllm.engine.arg_utils.get_model_path", lambda model, revision: model)


async def _run_server_capture(monkeypatch, engine_args, *, cli_args=None, diffusion=False):
    """Drive ``run_server`` with stubbed plumbing; return the AsyncOmni kwargs."""
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server._generate_strategy = MagicMock()
    server.config = SimpleNamespace(
        external_lib=None,
        enable_prompt_embed_cache=False,
        prompt_embed_cache_size=32,
    )
    if diffusion:
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

        server._generate_strategy = DiffusionStrategy(server)
        server.model_config = SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm="flow_grpo")
    server._server_address = ("127.0.0.1", 0)

    monkeypatch.setattr(server_module.OmniEngineArgs, "from_cli_args", classmethod(lambda cls, _: engine_args))
    monkeypatch.setattr(server_module, "get_free_port", lambda *a, **k: (12345, SimpleNamespace(close=lambda: None)))
    # run_server sets MASTER_ADDR/MASTER_PORT; write them to a scratch copy.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    captured: dict = {}

    def _capture_async_omni(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    async def _noop_init_app_state(*args, **kwargs):
        return None

    async def _stub_run_uvicorn(*args, **kwargs):
        return (0, None)

    monkeypatch.setattr(server_module, "AsyncOmni", _capture_async_omni)
    monkeypatch.setattr(server_module, "build_app", lambda args: MagicMock())
    monkeypatch.setattr(server_module, "omni_init_app_state", _noop_init_app_state)
    monkeypatch.setattr(server_module, "run_uvicorn", _stub_run_uvicorn)

    await server.run_server(cli_args if cli_args is not None else SimpleNamespace())
    return captured


async def test_run_server_strips_default_fault_tolerance_config(monkeypatch):
    captured = await _run_server_capture(monkeypatch, server_module.OmniEngineArgs(model="m"))
    assert "fault_tolerance_config" not in captured
    assert captured["enable_fault_tolerance"] is False


async def test_run_server_keeps_fault_tolerance_when_explicitly_enabled(monkeypatch):
    engine_args = server_module.OmniEngineArgs(model="m")
    engine_args.enable_fault_tolerance = True
    captured = await _run_server_capture(monkeypatch, engine_args)
    assert isinstance(captured["fault_tolerance_config"], dict)
    assert captured["enable_fault_tolerance"] is True


@pytest.mark.parametrize(
    ("dit_tp", "ulysses_sp", "encoder_tp", "vae_parallel"),
    [(8, 1, 8, 8), (4, 1, 4, 4), (8, 1, 1, 1), (2, 4, 8, 8), (4, 2, 8, 8), (1, 8, 8, 8)],
)
async def test_run_server_preserves_diffusion_parallel_cli_args(
    monkeypatch, dit_tp, ulysses_sp, encoder_tp, vae_parallel
):
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    cli_args = SimpleNamespace(
        model="m",
        tensor_parallel_size=dit_tp,
        ulysses_degree=ulysses_sp,
        ring_degree=1,
        num_gpus=dit_tp * ulysses_sp,
        text_encoder_tp_size=encoder_tp,
        vae_patch_parallel_size=vae_parallel,
    )
    engine_args = server_module.OmniEngineArgs.from_cli_args(cli_args)

    captured = await _run_server_capture(monkeypatch, engine_args, cli_args=cli_args, diffusion=True)

    assert captured["tensor_parallel_size"] == dit_tp
    assert captured["ulysses_degree"] == ulysses_sp
    assert captured["ring_degree"] == 1
    assert captured["num_gpus"] == dit_tp * ulysses_sp
    assert captured["text_encoder_tp_size"] == encoder_tp
    assert captured["vae_patch_parallel_size"] == vae_parallel

    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg(captured)[0]
    parallel = stage["engine_args"]["parallel_config"]
    assert parallel.tensor_parallel_size == dit_tp
    assert parallel.sequence_parallel_size == ulysses_sp
    assert parallel.text_encoder_tp_size == encoder_tp
    assert parallel.vae_patch_parallel_size == vae_parallel
    assert parallel.world_size == dit_tp * ulysses_sp
    assert parallel.data_parallel_size == 1
    assert len(stage["runtime"]["devices"].split(",")) == dit_tp * ulysses_sp
