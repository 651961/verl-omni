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

"""ART window coverage, frozen targets, optimizer boundaries and DP batch sizes."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.diffusion_art.data import pack_art_trajectories, prompt_window_indices
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

ROOT = Path(__file__).resolve().parents[3]


class _Trainer(PolicyGradientDiffusionTrainerV1):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def config(*overrides):
    """Compose ART's real defaults without creating workers or loading weights."""
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        return compose(
            config_name="diffusion_trainer",
            overrides=[
                "algorithm.trainer_type=direct_preference",
                "actor_rollout_ref.model.model_type=diffusion_art_model",
                "actor_rollout_ref.model.algorithm=diffusion_art",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
                "trainer.use_v1=true",
                "trainer.v1.trainer_mode=separate_async",
                "trainer.v1.separate_async.hybrid_rollout.enable_switch=false",
                "trainer.n_gpus_per_node=8",
                "trainer.nnodes=1",
                *overrides,
            ],
        )


@pytest.mark.parametrize("p,k", [(1, 2), (4, 8), (8, 8), (64, 18)])
def test_windows_preserve_all_prompts_and_trajectory_slots(p, k):
    # Interleaved, nonnumeric prompt identities must work too.
    uids = [f"prompt-{group}" for _ in range(k) for group in range(p)]
    generator = torch.Generator().manual_seed(42)
    windows = prompt_window_indices(uids, generator=generator)
    assert len(windows) == 2
    for rows in windows:
        assert Counter(uids[row] for row in rows) == {uid: k // 2 for uid in set(uids)}
    assert sorted(row for rows in windows for row in rows) == list(range(p * k))


def test_windows_shuffle_by_default_and_support_reproducible_seeds():
    uids = ["a"] * 8 + ["b"] * 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        windows = prompt_window_indices(uids)
    assert windows == prompt_window_indices(uids, generator=torch.Generator().manual_seed(42))
    assert windows != prompt_window_indices(uids, generator=torch.Generator().manual_seed(43))
    assert any(rows != sorted(rows) for rows in windows)


@pytest.mark.parametrize("uids", [[], ["a"] * 3 + ["b"] * 3])
def test_invalid_groups_fail_before_any_update(uids):
    with pytest.raises(ValueError):
        prompt_window_indices(uids)


@pytest.mark.parametrize(
    "overrides,error",
    [
        (["actor_rollout_ref.rollout.n=3"], "even K"),
        (["actor_rollout_ref.actor.ppo_epochs=2"], "ppo_epochs=1"),
        (["actor_rollout_ref.actor.ppo_mini_batch_size=1"], "divisible by actor DP size"),
        (["algorithm.paired_preference=true"], "paired_preference=false"),
    ],
)
def test_unsupported_schedules_rejected_before_worker_allocation(overrides, error):
    with pytest.raises(ValueError, match=error):
        DiffusionARTLoss.validate_trainer_config(config(*overrides))


@pytest.mark.parametrize("sync_steps", [1, 3])
def test_lr_schedule_counts_every_art_window(monkeypatch, sync_steps):
    trainer = object.__new__(_Trainer)
    trainer.config = config("trainer.total_training_steps=7")
    trainer.parameter_sync_step = sync_steps
    trainer.tokenizer = None
    trainer.processor = None
    trainer._generation_batch_size = lambda: 8
    monkeypatch.setattr("verl_omni.utils.dataset.rl_dataset.create_rl_dataset", lambda *args, **kwargs: list(range(64)))
    monkeypatch.setattr("verl_omni.utils.dataset.rl_dataset.create_rl_sampler", lambda *args: None)
    monkeypatch.setattr("verl_omni.utils.dataset.rl_dataset.get_collate_fn", lambda *args: None)
    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.trainer_base.StatefulDataLoader", lambda **kwargs: None)
    trainer._init_dataloader()
    assert trainer.total_training_steps == 7
    assert trainer.config.actor_rollout_ref.actor.optim.total_training_steps == 7 * sync_steps * 2


@pytest.mark.parametrize("skip_first", [False, True])
def test_sync_cycle_counts_all_windows_before_returning(skip_first):
    trainer = object.__new__(_Trainer)
    trainer.config = config()
    trainer._loss_fn = DiffusionARTLoss()
    trainer.config.data.train_batch_size = 8
    trainer.parameter_sync_step = 2
    trainer.global_steps = 3
    trainer.prepare_step = lambda: {}
    trainer._should_prefetch_local_batches = lambda: False
    seen = []

    def update_actor(window):
        seen.append((trainer.local_trigger_step, window.batch["row_id"].tolist()))
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/loss": 1.0}})

    def step_once(metrics, timing_raw, prompt_count):
        assert prompt_count == 4
        data = DataProto.from_dict(
            tensors={
                "row_id": torch.arange(32),
                "art_directions": torch.ones(32, 1, 1),
                "art_magnitudes": torch.full((32,), float(not skip_first or trainer.local_trigger_step != 0)),
            },
            non_tensors={"uid": np.repeat([f"p{i}" for i in range(4)], 8)},
        )
        metrics.update(trainer._update_actor_windows(data))
        return KVBatchMeta(
            partition_id="train", keys=[f"{trainer.local_trigger_step}-{i}" for i in range(32)], tags=[{}] * 32
        )

    trainer._update_actor = update_actor
    trainer._step_once = step_once
    metrics = {}
    trainer.step(metrics, {})
    assert [trigger for trigger, _ in seen] == ([1, 1] if skip_first else [0, 0, 1, 1])
    assert all(len(rows) == 16 for _, rows in seen)
    assert metrics["actor/art_update_windows"] == (2 if skip_first else 4)
    assert metrics["actor/art_trajectories_per_update"] == 16
    assert metrics["actor/art_skipped_zero_guidance"] == int(skip_first)


@pytest.mark.parametrize(
    "magnitudes,directions,skipped",
    [
        ([0, 0, 0, 0], [1, 1, 1, 1], True),
        ([1, 1, 1, 1], [0, 0, 0, 0], True),
        ([1, 1, 0, 0], [0, 0, 1, 1], True),
        ([0, 0, 0, 1], [0, 0, 0, 1], False),
    ],
)
def test_zero_guidance_skips_both_optimizer_and_scheduler_steps(magnitudes, directions, skipped):
    with torch.enable_grad():
        trainer = object.__new__(_Trainer)
        trainer.config = config()
        trainer.global_steps = 3
        trainer.parameter_sync_step = 1
        trainer.local_trigger_step = 0
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        def worker_update(batch):
            optimizer.zero_grad()
            model(torch.ones(len(batch), 1)).square().mean().backward()
            optimizer.step()
            scheduler.step()
            return tu.get_tensordict({}, non_tensor_dict={"metrics": {"loss": 1.0}})

        # Populate Adam momentum before checking that a skipped batch leaves it intact.
        worker_update(torch.ones(1))
        weight_before = model.weight.detach().clone()
        state_before = {key: value.clone() for key, value in optimizer.state[model.weight].items()}
        epoch_before, lr_before = scheduler.last_epoch, scheduler.get_last_lr()
        worker = Mock(side_effect=worker_update)
        trainer.actor_rollout_wg = SimpleNamespace(update_actor=worker)
        data = DataProto.from_dict(
            tensors={
                "art_magnitudes": torch.tensor(magnitudes, dtype=torch.float32),
                "art_directions": torch.tensor(directions, dtype=torch.float32).reshape(4, 1, 1),
            },
            non_tensors={"uid": np.array(["a", "a", "b", "b"])},
        )
        metrics = trainer._update_actor_windows(data)
        updates = 0 if skipped else 2
        assert worker.call_count == updates
        assert metrics["actor/art_update_windows"] == updates
        assert metrics["actor/art_skipped_zero_guidance"] == int(skipped)
        assert scheduler.last_epoch == epoch_before + updates
        assert optimizer.state[model.weight]["step"].item() == state_before["step"].item() + updates
        if skipped:
            torch.testing.assert_close(model.weight, weight_before, rtol=0, atol=0)
            for key, value in state_before.items():
                torch.testing.assert_close(optimizer.state[model.weight][key], value, rtol=0, atol=0)
            assert scheduler.get_last_lr() == lr_before
        else:
            assert metrics["actor/art_trajectories_per_update"] == 2
            assert scheduler.get_last_lr() == pytest.approx([lr_before[0] * 0.5**updates])


def test_trainer_computes_full_guidance_then_updates_frozen_disjoint_windows(monkeypatch):
    # Exercise real DataProto selection, trainer routing and worker metadata. The
    # worker uses a tiny CPU model instead of launching Ray or distributed FSDP.
    with torch.enable_grad():
        c = config()
        trainer = object.__new__(_Trainer)
        trainer.config = c
        trainer.global_steps = 3
        trainer.parameter_sync_step = 1
        trainer.local_trigger_step = 0
        trainer._loss_fn = DiffusionARTLoss()
        trainer._has_old_adapter = False
        trainer.use_reference_policy = False

        generator = torch.Generator().manual_seed(6)
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.data.fill_(0.1)
        reference = torch.nn.Linear(2, 2, bias=False)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        oracle = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0)
        caches = []
        for _ in range(64):
            states = torch.randn(2, 3, 2, generator=generator)
            caches.append(
                dict(
                    states=states,
                    old_velocities=model(states).detach(),
                    endpoints=torch.randn(3, 2, generator=generator),
                    timesteps=torch.tensor([800.0, 300.0]),
                    sigmas=torch.tensor([0.8, 0.3]),
                    dt=torch.tensor([-0.5, -0.3]),
                    conditions={},
                )
            )
        uids = np.repeat([f"p{i}" for i in range(8)], 8)
        rewards = torch.arange(8).float().repeat(8)
        rewards[:8] = 1  # Zero-guidance prompt stays in both loss denominators.
        packed = pack_art_trajectories(caches, rewards, uids)
        frozen = {key: value.clone() for key, value in packed.items()}
        prepare = Mock(side_effect=pack_art_trajectories)
        monkeypatch.setattr(
            DiffusionModelBase,
            "get_class",
            lambda _: SimpleNamespace(
                prepare_art_batch=prepare,
                prepare_art_rewards=lambda batch, rewards: rewards.reshape(len(rewards), -1).sum(-1),
            ),
        )
        monkeypatch.setattr(
            "verl_omni.trainer.diffusion.v1.trainer_base.extract_reward", lambda _: (rewards[:, None], {})
        )
        monkeypatch.setattr(
            "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq", lambda *args, **kwargs: None
        )

        data = DataProto.from_dict(
            tensors={"row_id": torch.arange(64)},
            non_tensors={"uid": uids, "art_cache": np.array(caches, dtype=object)},
        )
        seen = []

        def worker_update(batch):
            rows = batch["row_id"].tolist()
            seen.append(rows)
            assert tu.get_non_tensor_data(batch, "mini_batch_size", None) == len(rows) == 32
            assert tu.get_non_tensor_data(batch, "global_batch_size", None) == len(rows)
            assert tu.get_non_tensor_data(batch, "epochs", None) == 1
            for key, value in frozen.items():
                torch.testing.assert_close(batch[key], value[rows])
            states = batch["art_states"]
            # Independent dense oracle for the frozen, sigma-weighted MSE.
            target = (
                batch["art_old_velocities"]
                - batch["art_magnitudes"][:, None, None, None]
                * batch["art_directions"][:, None]
                / batch["art_directions"].square().flatten(1).mean(1).sqrt().clamp_min(1e-5)[:, None, None, None]
            )
            oracle.zero_grad()
            ((reference(states) - target).square() * batch["art_sigmas"][:, :, None, None]).mean().backward()
            oracle.step()
            optimizer.zero_grad()
            for row in range(len(rows)):
                for step in range(2):
                    loss, _ = DiffusionARTLoss.compute_loss(
                        config=c.actor_rollout_ref.actor,
                        current_velocity=model(states[row : row + 1, step]),
                        old_velocity=batch["art_old_velocities"][row : row + 1, step],
                        state=states[row : row + 1, step],
                        sigma=batch["art_sigmas"][row : row + 1, step],
                        dt=batch["art_dt"][row : row + 1, step],
                        direction=batch["art_directions"][row : row + 1],
                        magnitude=batch["art_magnitudes"][row : row + 1],
                    )
                    # Disable masking for comparison with the unmasked oracle.
                    (loss / (len(rows) * 2)).backward()
            optimizer.step()
            torch.testing.assert_close(model.weight, reference.weight, atol=1e-6, rtol=1e-5)
            return tu.get_tensordict({}, non_tensor_dict={"metrics": {"loss": float(len(seen))}})

        c.actor_rollout_ref.actor.diffusion_loss.art_loss_options = {"corrective_mask_enabled": False}
        trainer.actor_rollout_wg = SimpleNamespace(update_actor=worker_update)
        metrics = {}
        meta = KVBatchMeta(partition_id="train", keys=[str(i) for i in range(64)], tags=[{}] * 64)
        assert trainer._train_direct_preference_batch(metrics, {}, meta, data) is meta
        prepare.assert_called_once()
        assert len(seen) == 2
        assert sorted(row for rows in seen for row in rows) == list(range(64))
        assert all(Counter(uids[row] for row in rows) == {uid: 4 for uid in uids} for rows in seen)
        assert metrics["actor/art_update_windows"] == 2
        assert metrics["actor/art_trajectories_per_update"] == 32
        assert metrics["actor/loss"] == pytest.approx(1.5)
        assert optimizer.state[model.weight]["step"].item() == 2


@pytest.mark.parametrize("active", [False, True])
def test_partitioned_guidance_with_batch_size_different_from_modality_count(active):
    # B=4, modalities=2 must never broadcast a row's magnitude onto another row.
    magnitudes = torch.zeros(4, 2)
    directions = torch.zeros(4, 5, 1)
    if active:
        magnitudes[0, 1] = 1
        directions[0, 3:] = 1
    batch = DataProto.from_dict(
        tensors={"art_magnitudes": magnitudes, "art_directions": directions},
        non_tensors={"uid": np.array(["a", "a", "b", "b"])},
    )
    windows, metrics = DiffusionARTLoss.actor_update_batches(batch, torch.Generator().manual_seed(5))
    assert len(windows) == (2 if active else 0)
    assert metrics["actor/art_skipped_zero_guidance"] == int(not active)
    if active:
        assert all(sorted(window.non_tensor_batch["uid"]) == ["a", "b"] for window in windows)
