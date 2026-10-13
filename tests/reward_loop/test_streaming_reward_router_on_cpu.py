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
"""Resident streaming reward routing without Ray or installed serving packages."""

import __future__

import ast
import asyncio
import dataclasses
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[2]


def execute(relative, namespace, names=None):
    tree = ast.parse((ROOT / relative).read_text())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.Import | ast.ImportFrom) and (names is None or getattr(node, "name", None) in names)
    ]
    exec(compile(tree, relative, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def router_class():
    namespace = {"asyncio": asyncio}
    execute("verl_omni/reward_loop/reward_model_executor.py", namespace, {"_await_owned"})
    return execute("verl_omni/reward_loop/streaming_reward.py", namespace)["StreamingRewardRouter"]


class Config(dict):
    def __getattr__(self, key):
        return self[key]


class StreamingRewardTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_term_exactly_once_with_bounded_replica_concurrency(self):
        calls, inflight, maximum = [], {}, {}

        def worker(name, replica, reward):
            identity = name, replica
            inflight[identity] = maximum[identity] = 0

            async def score(data):
                inflight[identity] += 1
                maximum[identity] = max(maximum[identity], inflight[identity])
                calls.append((name, data[0]))
                await asyncio.sleep(0.001)
                inflight[identity] -= 1
                return {"reward_score": reward, "reward_extra_info": {name: reward, "reward/combined": reward}}

            return SimpleNamespace(
                compute_score=SimpleNamespace(remote=score), wake_up_reward_model=SimpleNamespace(remote=AsyncMock())
            )

        groups = {"quality": [worker("quality", i, 2) for i in range(2)], "alignment": [worker("alignment", 0, 3)]}
        router = router_class()(groups, {"quality"})
        results = await asyncio.gather(*(router.compute_score([i]) for i in range(7)))
        self.assertEqual(sorted(calls), sorted((name, i) for name in groups for i in range(7)))
        self.assertTrue(all(value == 1 for value in maximum.values()))
        self.assertTrue(all(r["reward_score"] == 5 for r in results))
        self.assertTrue(
            all(r["reward_extra_info"] == {"quality": 2, "alignment": 3, "reward/combined": 5} for r in results)
        )
        for worker_handle in groups["quality"]:
            worker_handle.wake_up_reward_model.remote.assert_awaited_once_with("quality")
        self.assertEqual(router.available["quality"].qsize(), 2)

    async def test_failure_and_cancellation_drain_other_groups_and_release_workers(self):
        release, entered = asyncio.Event(), asyncio.Event()

        async def slow(data):
            entered.set()
            await release.wait()
            return {"reward_score": 1}

        async def failed(data):
            raise RuntimeError("scorer failed")

        def handle(fn):
            return SimpleNamespace(compute_score=SimpleNamespace(remote=fn))

        router = router_class()({"slow": [handle(slow)], "bad": [handle(failed)]}, set())
        task = asyncio.create_task(router.compute_score([0]))
        await entered.wait()
        self.assertFalse(task.done())
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(queue.qsize() == 1 for queue in router.available.values()))
        with self.assertRaisesRegex(RuntimeError, "scorer failed"):
            await router.compute_score([1])
        self.assertTrue(all(queue.qsize() == 1 for queue in router.available.values()))

    def test_only_dedicated_resident_named_models_enable_streaming(self):
        namespace = {
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "BaseConfig": object,
            "Mapping": Mapping,
            "OmegaConf": SimpleNamespace(is_config=lambda value: False),
        }
        enabled = execute("verl_omni/workers/config/reward.py", namespace)["streaming_reward_enabled"]
        config = SimpleNamespace(
            reward=Config(
                reward_model=Config(enable=False, enable_resource_pool=True),
                models={"llm": {"backend": "engine", "offload": False}},
            )
        )
        self.assertTrue(enabled(config))
        config.reward.models["llm"]["offload"] = True
        self.assertFalse(enabled(config))
        config.reward.models["llm"]["offload"] = False
        config.reward.reward_model["enable_resource_pool"] = False
        self.assertFalse(enabled(config))
        config.reward.reward_model["enable_resource_pool"] = True
        self.assertTrue(enabled(config))


if __name__ == "__main__":
    unittest.main()
