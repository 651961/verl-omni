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
"""Bounded streaming dispatch for resident models in a dedicated reward pool."""

import asyncio

from .reward_model_executor import _await_owned


class StreamingRewardRouter:
    """Route each sample through every reward group exactly once.

    Only actor handles and available-worker queues live here. Requests retain
    their payload until their accepted RPCs finish; no replay cache is created.
    GPU models stay resident throughout streaming, so one sample cannot offload
    a model while another sample is still using it.
    """

    def __init__(self, worker_groups, native_names):
        self.worker_groups = worker_groups
        self.native_names = native_names
        self._ready_lock = asyncio.Lock()
        self._ready = False
        self.available = {}
        for name, workers in worker_groups.items():
            if not workers:
                raise ValueError(f"Reward group {name!r} has no workers.")
            queue = asyncio.Queue()
            for worker in workers:
                queue.put_nowait(worker)
            self.available[name] = queue

    async def _ensure_ready(self):
        async with self._ready_lock:
            if self._ready:
                return
            results = await asyncio.gather(
                *(
                    worker.wake_up_reward_model.remote(name)
                    for name in self.native_names
                    for worker in self.worker_groups[name]
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self._ready = True

    async def compute_score(self, data):
        """Score one sample while retaining ownership until all accepted work settles."""
        if len(data) != 1:
            raise ValueError("Streaming reward expects a single sample per request.")
        return await _await_owned(self._compute_score(data))

    async def _compute_score(self, data):
        await self._ensure_ready()

        async def score_group(name):
            queue = self.available[name]
            worker = await queue.get()
            try:
                return await worker.compute_score.remote(data)
            finally:
                queue.put_nowait(worker)

        results = await asyncio.gather(*(score_group(name) for name in self.available), return_exceptions=True)
        total, info = 0.0, {}
        for result in results:
            if isinstance(result, BaseException):
                raise result
            total += float(result["reward_score"])
            for key, value in result.get("reward_extra_info", {}).items():
                if key == "reward/combined":
                    continue
                if key in info:
                    raise ValueError(f"Duplicate reward extra-info key {key!r} across worker groups")
                info[key] = value
        info["reward/combined"] = total
        return {"reward_score": total, "reward_extra_info": info}
