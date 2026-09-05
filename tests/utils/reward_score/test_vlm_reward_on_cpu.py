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
"""Reference media and score contracts for the shared VLM reward."""

import asyncio
import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from verl_omni.utils.reward_score import vlm_reward as reward

VIDEO_SPEC = {"resolution": "1024x1024", "num_frames": 124, "frame_rate": 24.0}
GENERATED_URL = "data:video/mp4;base64,Z2VuZXJhdGVk"
SCORES = {"instruction_following": 3}


@pytest.fixture
def judge(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=r"Final score: \boxed{3}"), finish_reason="stop")]
    )
    complete = AsyncMock(return_value=response)
    encode = Mock(return_value=GENERATED_URL)
    monkeypatch.setattr(reward, "_chat_complete", complete)
    monkeypatch.setattr(reward, "_video_tensor_to_data_url", encode)
    return SimpleNamespace(complete=complete, encode=encode)


def _score(extra_info):
    return asyncio.run(
        reward.compute_score_vlm(
            data_source="minimax_h3_ref2va",
            solution_image=torch.zeros(124, 3, 32, 32, dtype=torch.uint8),
            ground_truth="Continue the action.",
            extra_info={**VIDEO_SPEC, **extra_info},
            reward_router_address="judge:8000",
            model_name="test-vlm",
        )
    )


def _request_media(judge):
    judge.complete.assert_awaited_once()
    router, request = judge.complete.call_args.args
    assert router == "judge:8000"
    assert request["model"] == "test-vlm"
    content = request["messages"][0]["content"]
    return [(item["type"], item[item["type"]]["url"]) for item in content if item["type"] != "text"]


@pytest.mark.parametrize(
    ("image_count", "video_count", "container"),
    [(2, 0, list), (0, 2, tuple), (2, 2, np.array)],
    ids=["images", "videos", "mixed-numpy"],
)
def test_reward_sends_all_references_before_generated_video(tmp_path, judge, image_count, video_count, container):
    extra_info = {"source_audios": [str(tmp_path / "unused.wav")], "audio": torch.ones(2, 100)}
    expected = []
    for modality, suffix, count in (("image", "png", image_count), ("video", "mp4", video_count)):
        paths = []
        for index in range(count):
            path = tmp_path / f"reference_{index}.{suffix}"
            payload = f"{modality}-{index}".encode()
            path.write_bytes(payload)
            paths.append(str(path))
            encoded = base64.b64encode(payload).decode("ascii")
            expected.append((f"{modality}_url", f"data:{modality}/{suffix};base64,{encoded}"))
        if paths:
            extra_info[f"source_{modality}s"] = container(paths)

    result = _score(extra_info)

    assert _request_media(judge) == [*expected, ("video_url", GENERATED_URL)]
    video, fps = judge.encode.call_args.args
    assert video.shape == (124, 3, 32, 32)
    assert fps == 24
    assert result["score"] == pytest.approx(0.5)
    assert {key: result[key] for key in SCORES} == SCORES
    prompt = judge.complete.call_args.args[1]["messages"][0]["content"][-1]["text"]
    assert "第一步：检查生成的目标视频中是否发生物理畸变" in prompt
    assert prompt.endswith("请逐步推理，并将最终答案放在 \\\\boxed{} 中。")
    assert "Continue the action." in prompt


def test_prepared_video_only_row_can_be_scored(tmp_path, judge):
    path = Path(__file__).parents[3] / "examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py"
    spec = importlib.util.spec_from_file_location("prepare_ref2va_reward_test", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    (tmp_path / "reference.mp4").write_bytes(b"reference-video")
    record = {"prompt": "Continue the action.", "video": "reference.mp4", **VIDEO_SPEC}
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    row = prepare._convert_split(tmp_path, "train", max_samples=-1).iloc[0]
    assert row["extra_info"]["source_images"] == []

    result = _score(row["extra_info"])

    assert result["score"] == pytest.approx(0.5)
    assert _request_media(judge) == [
        ("video_url", "data:video/mp4;base64," + base64.b64encode(b"reference-video").decode("ascii")),
        ("video_url", GENERATED_URL),
    ]


@pytest.mark.parametrize("extra_info", [{}, {"source_images": [], "source_videos": np.array([])}])
def test_reward_requires_a_visual_reference(judge, extra_info):
    with pytest.raises(ValueError, match="reference"):
        _score(extra_info)
    judge.complete.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "filename"),
    [("source_images", "reference.mp4"), ("source_videos", "reference.png"), ("source_videos", "reference.wav")],
)
def test_reward_rejects_reference_with_wrong_media_type(tmp_path, judge, field, filename):
    path = tmp_path / filename
    path.write_bytes(b"reference")
    with pytest.raises(ValueError):
        _score({field: [str(path)]})
    judge.complete.assert_not_awaited()


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), 0, 6, -1])
def test_reward_rejects_nonfinite_or_out_of_range_scores(tmp_path, judge, invalid_score):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference")
    judge.complete.return_value.choices[0].message.content = rf"\boxed{{{invalid_score}}}"
    with pytest.raises(ValueError, match="malformed scores"):
        _score({"source_images": [str(path)]})


def test_reward_leaves_generation_parameters_to_the_model(tmp_path, judge):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference")

    _score({"source_images": [str(path)]})

    request = judge.complete.call_args.args[1]
    assert set(request) == {"model", "messages"}


@pytest.mark.parametrize(
    "content",
    ["Let me carefully evaluate the video. " * 100, r"\boxed{", r"Initially \boxed{6}, but on further consideration"],
)
def test_reward_reports_truncated_reply_without_returning_a_score(tmp_path, judge, content):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference")
    choice = judge.complete.return_value.choices[0]
    choice.message.content = content
    choice.finish_reason = "length"

    with pytest.raises(ValueError, match="malformed scores.*finish_reason='length'") as exc:
        _score({"source_images": [str(path)]})
    assert len(str(exc.value)) < 650


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (r"\boxed{1}", 1.0),
        (r"\boxed{5}", 5.0),
        (r"First \boxed{4}, finally $\boxed{3.5}$", 3.5),
        ("Final answer: \\boxed{\n 3.5 \n}", 3.5),
    ],
)
def test_reward_uses_last_boxed_score_from_content(tmp_path, judge, content, expected):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference")
    message = judge.complete.return_value.choices[0].message
    message.content = content
    message.reasoning_content = r"My initial assessment is \boxed{1}"

    result = _score({"source_images": [str(path)]})

    assert result["instruction_following"] == pytest.approx(expected)
    assert result["score"] == pytest.approx((expected - 1.0) / 4.0)
    assert result["judge_response"] == content


@pytest.mark.parametrize("content", [None, "No final score.", r"Initially \boxed{6}, finally \boxed{invalid}"])
def test_reward_does_not_fall_back_to_thinking_or_earlier_score(tmp_path, judge, content):
    path = tmp_path / "reference.png"
    path.write_bytes(b"reference")
    message = judge.complete.return_value.choices[0].message
    message.content = content
    message.reasoning_content = r"My initial assessment is \boxed{8}"

    with pytest.raises(ValueError, match="malformed scores"):
        _score({"source_images": [str(path)]})
