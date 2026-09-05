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
"""Required video specifications, mixed-shape sampling and checkpoint resume."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl_omni.utils.dataset.minimax_h3_video import resolve_video_spec
from verl_omni.utils.dataset.rl_dataset import create_rl_sampler


def _spec(**kwargs):
    return {"resolution": "1344x768", "num_frames": 124, "frame_rate": 24.0, **kwargs}


@pytest.mark.parametrize("key", ["resolution", "num_frames", "frame_rate"])
def test_every_generation_field_is_required(key):
    fields = _spec()
    del fields[key]
    with pytest.raises(ValueError, match=f"Missing required video field: extra_info.{key}"):
        resolve_video_spec({"extra_info": fields})


@pytest.mark.parametrize(
    "fields",
    [
        {"resolution": "1024x1000"},
        {"resolution": "1024x0"},
        {"resolution": "4096x768"},
        {"num_frames": 96},
        {"num_frames": 5},
        {"num_frames": 362},
        {"num_frames": 124.5},
        {"num_frames": True},
        {"frame_rate": 30},
        {"frame_rate": float("nan")},
        {"frame_rate": True},
    ],
)
def test_rejects_shapes_that_the_engine_would_change_or_reject(fields):
    with pytest.raises(ValueError):
        resolve_video_spec({"extra_info": _spec(**fields)})


@pytest.mark.parametrize("resolution", ["1344x768", "768x1344", "1024x1024", "1152x864"])
def test_accepts_general_aligned_resolutions(resolution):
    spec = resolve_video_spec({"extra_info": _spec(resolution=resolution, num_frames=107)})
    assert (spec["width"], spec["height"]) == tuple(map(int, resolution.split("x")))
    assert spec["num_frames"] == 107


def test_conflicting_field_sources_fail_closed():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_video_spec({"extra_info": _spec(), "num_frames": 141})


@pytest.mark.parametrize("shuffle,seed", [(True, 42), (True, None), (False, 42)])
def test_default_sampler_keeps_every_mixed_resolution_row(shuffle, seed):
    rows = [{"extra_info": _spec(resolution="1344x768" if i < 31 else "768x1344"), "index": i} for i in range(32)]
    config = OmegaConf.create({"shuffle": shuffle, "seed": seed})
    sampler = create_rl_sampler(config, rows)
    loader = StatefulDataLoader(rows, batch_size=8, sampler=sampler, collate_fn=list, drop_last=True)
    indices = [row["index"] for batch in loader for row in batch]
    assert sorted(indices) == list(range(32))
    if not shuffle:
        assert indices == list(range(32))


def test_sampler_resume_preserves_remaining_rows():
    rows = [{"index": i, "extra_info": _spec()} for i in range(40)]
    config = OmegaConf.create({"shuffle": True, "seed": 42})

    def loader():
        return StatefulDataLoader(
            rows,
            batch_size=8,
            sampler=create_rl_sampler(config, rows),
            collate_fn=list,
        )

    first = loader()
    iterator = iter(first)
    next(iterator)
    state = first.state_dict()
    expected = [[row["index"] for row in batch] for batch in iterator]
    resumed = loader()
    resumed.load_state_dict(state)
    assert [[row["index"] for row in batch] for batch in resumed] == expected


@pytest.mark.parametrize("mode", ["fl2va", "ref2va"])
def test_preparation_preserves_mixed_shapes_and_rejects_missing_fields(tmp_path, mode):
    path = Path(__file__).parents[2] / f"examples/diffusionnft_trainer/minimax_h3/prepare_{mode}_data.py"
    spec = importlib.util.spec_from_file_location(f"prepare_{mode}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "image.png").write_bytes(b"image")
    rows = [
        {"prompt": "Animate", "images": ["image.png"], "extra_info": _spec()},
        {
            "prompt": "Animate another",
            "images": ["image.png"],
            "extra_info": _spec(resolution="1024x1024", num_frames=141),
        },
    ]
    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    kwargs = {"frame_mode": "first"} if mode == "fl2va" else {}
    frame = module._convert_split(tmp_path, "train", max_samples=-1, **kwargs)
    output = tmp_path / "train.parquet"
    frame.to_parquet(output)
    actual = pd.read_parquet(output)["extra_info"].tolist()
    for info, row in zip(actual, rows, strict=True):
        for key in ("resolution", "num_frames", "frame_rate"):
            assert info[key] == row["extra_info"][key]
    del rows[1]["extra_info"]["num_frames"]
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="train row 1.*num_frames"):
        module._convert_split(tmp_path, "train", max_samples=-1, **kwargs)
