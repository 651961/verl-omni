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
"""Validation dumps retain dataset indices through repetition and worker padding."""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from verl import DataProto

pytest.importorskip("diffusers")

import verl_omni.trainer.diffusion.ray_diffusion_trainer as ray_diffusion_trainer
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer


def _validation_batch(prompt_ids, extra_infos=None):
    batch = {
        "input_ids": torch.tensor(prompt_ids).unsqueeze(-1),
        "raw_prompt": np.array(prompt_ids, dtype=object),
        "uid": np.array([f"uid-{value}" for value in prompt_ids], dtype=object),
        "data_source": np.array(["validation"] * len(prompt_ids), dtype=object),
        "reward_model": np.array([{"ground_truth": f"gt-{value}"} for value in prompt_ids], dtype=object),
    }
    if extra_infos is not None:
        batch["extra_info"] = np.array(extra_infos, dtype=object)
    return batch


@pytest.mark.parametrize("media_type", ["image", "video"])
@pytest.mark.parametrize("max_samples", [None, 7])
def test_validation_dump_preserves_sample_indices(tmp_path, monkeypatch, media_type, max_samples):
    padded_prompt_ids = []
    video_exports = []

    def generate_sequences(batch):
        prompt_ids = batch.non_tensor_batch["raw_prompt"].tolist()
        padded_prompt_ids.append(prompt_ids)
        shape = (3, 4, 4) if media_type == "image" else (2, 3, 4, 4)
        return DataProto.from_dict(
            tensors={
                "prompts": torch.tensor(prompt_ids).unsqueeze(-1),
                "responses": torch.stack([torch.full(shape, value, dtype=torch.uint8) for value in prompt_ids]),
                "rm_scores": torch.tensor(prompt_ids, dtype=torch.float32).unsqueeze(-1),
            }
        )

    def extract_reward(batch):
        return batch.batch["rm_scores"], {"quality": batch.batch["prompts"][:, 0].numpy() / 2}

    def export_video(video, path, *, fps, **kwargs):
        video_exports.append((int(video[0, 0, 0, 0]), fps))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(ray_diffusion_trainer, "extract_reward", extract_reward)
    monkeypatch.setattr(ray_diffusion_trainer, "_export_video", export_video)
    metric_update = Mock(return_value={"validation/reward": 1.0})
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 2}, "agent": {"num_workers": 3}}},
                "trainer": {
                    "validation_data_dir": str(tmp_path),
                    "validation_data_max_samples": max_samples,
                    "log_val_generations": 0,
                    "video_fps": 12,
                },
            }
        ),
        global_steps=10,
        val_dataloader=[
            _validation_batch([16, 64], [{"index": np.int64(41)}, {}]),
            _validation_batch([96, 144], [{"index": np.int64(7)}, None]),
            _validation_batch([192]),
        ],
        use_rm=False,
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: f"prompt-{int(ids[0])}"),
        async_rollout_manager=SimpleNamespace(generate_sequences=generate_sequences),
        _val_metrics_update=metric_update,
    )
    for method in ("_get_gen_batch", "_dump_generations", "_maybe_log_val_generations"):
        setattr(trainer, method, MethodType(getattr(BaseRayDiffusionTrainer, method), trainer))

    metrics = BaseRayDiffusionTrainer._validate(trainer)

    assert metrics == {"validation/reward": 1.0}
    assert padded_prompt_ids == [[16, 16, 64, 64, 16, 16], [96, 96, 144, 144, 96, 96], [192, 192, 192]]
    prompt_ids = [16, 16, 64, 64, 96, 96, 144, 144, 192, 192]
    expected_indices = [41, 41, None, None, 7, 7, None, None, None, None]
    metric_update.assert_called_once()
    data_sources, sample_uids, reward_extras, sample_turns = metric_update.call_args.args
    assert data_sources.tolist() == ["validation"] * len(prompt_ids)
    assert sample_uids == [f"uid-{value}" for value in prompt_ids]
    assert reward_extras == {"reward": [float(value) for value in prompt_ids], "quality": [v / 2 for v in prompt_ids]}
    assert "extra_info" not in reward_extras
    assert sample_turns == []

    rows = [json.loads(line) for line in (tmp_path / "10.jsonl").read_text().splitlines()]
    dumped_ids = prompt_ids[:max_samples]
    assert [row["extra_info"] for row in rows] == [{"index": value} for value in expected_indices[:max_samples]]
    assert [row["input"] for row in rows] == [f"prompt-{value}" for value in dumped_ids]
    assert [row["gts"] for row in rows] == [f"gt-{value}" for value in dumped_ids]
    assert [row["score"] for row in rows] == [float(value) for value in dumped_ids]
    assert [row["reward"] for row in rows] == [float(value) for value in dumped_ids]
    assert [row["quality"] for row in rows] == [value / 2 for value in dumped_ids]
    assert [row["step"] for row in rows] == [10] * len(dumped_ids)
    extension = "jpg" if media_type == "image" else "mp4"
    expected_paths = [tmp_path / "10" / f"{i}.{extension}" for i in range(len(dumped_ids))]
    assert [row["output"] for row in rows] == [str(path) for path in expected_paths]
    assert set((tmp_path / "10").iterdir()) == set(expected_paths)
    if media_type == "image":
        for path, value in zip(expected_paths, dumped_ids, strict=True):
            with Image.open(path) as image:
                assert image.getpixel((0, 0)) == (value, value, value)
        assert video_exports == []
    else:
        assert video_exports == [(value, 12) for value in dumped_ids]
