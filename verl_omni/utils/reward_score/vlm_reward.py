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

import asyncio
import base64
import mimetypes
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

# ruff: noqa: E501 -- preserve the judge prompt's authored line breaks.
SCORE_KEYS = ("instruction_following",)

VLM_PROMPT = r"""你是首帧图像生视频任务的质量评审专家。你将收到以下内容：

1. 首帧图像：用户输入的图像，作为目标视频中的第一帧
2. 生成的目标视频：根据首帧图像和用户指令生成的视频
3. 用户指令：{{prompt}}

你的任务是对生成的目标视频进行严格、客观的视觉质量评估，并输出 1～5 分。

请严格按照以下顺序进行评估，不得跳过步骤。评分采用“首个失败项决定分数”的原则：一旦某一步不满足要求，立即确定分数，不再因为后续项目表现良好而提高分数。

第一步：检查生成的目标视频中是否发生物理畸变，违背物理规律

- 如果生成的目标视频存在畸变，包括物体悬空、人物肢体畸变，直接评为 1 分，停止评估。
- 只有当生成的目标视频没有物理畸变时，才能进入第二步。

第二步：检查生成的目标视频中每一个前景人物挥手打招呼的动作是否完成

- 视频中往往会有前景人物和背景人物，如果生成的目标视频中多个前景中的人物里面，没有人物没有挥手打招呼，直接评为 2 分，停止评估。
- 只有当生成的目标视频中有人物完成了挥手打招呼的动作时，才能进入第三步。

第三步：检查生成的目标视频中每一个前景人物嘴巴闭合程度是否不变

- 如果生成的目标视频中有任何一个前景人物的嘴巴发生了变化，直接评为 3 分，停止评估。
- 只有当生成的目标视频中所有前景人物的嘴巴闭合程度都保持不变时，才能进入第四步。

第四步：检查生成的目标视频中人物的身体运动是否自然

- 如果生成的目标视频中人物的身体运动不自然，头部完全静止，僵硬不动，直接评为 4 分，停止评估。
- 只有当生成的目标视频中人物的身体运动自然时，比如头部有微微倾斜摆动，下半身也有一些正常运动，给5分。

请逐步推理，并将最终答案放在 \\boxed{} 中。""".strip()


def _media_path_to_data_url(path_value: str, media_kind: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Reference {media_kind} does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None or not mime_type.startswith(f"{media_kind}/"):
        raise ValueError(f"Expected a reference {media_kind} file, got: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _video_tensor_to_data_url(video: torch.Tensor, fps: int) -> str:
    """Encode the complete generated tensor as a silent MP4 data URL."""
    from verl_omni.utils.tracking import _export_video

    with tempfile.TemporaryDirectory(prefix="vlm_reward_") as tmp_dir:
        video_path = Path(tmp_dir) / "generated.mp4"
        _export_video(video, str(video_path), fps=fps, audio=None)
        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _build_judge_prompt(prompt: str = "") -> str:
    """Inject the row's user instruction into the judge prompt template."""
    return VLM_PROMPT.replace("{{prompt}}", prompt or "")


def _build_content(
    reference_images: list[str], reference_videos: list[str], video_url: str, prompt: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, media_type, urls in (
        ("Reference images, in input order:", "image_url", reference_images),
        ("Reference videos, in input order:", "video_url", reference_videos),
        ("Generated video:", "video_url", [video_url]),
    ):
        if urls:
            content.append({"type": "text", "text": label})
            content.extend({"type": media_type, media_type: {"url": url}} for url in urls)
    content.append({"type": "text", "text": _build_judge_prompt(prompt)})
    return content


def _reference_paths(extra_info: dict, key: str) -> list[str]:
    values = extra_info.get(key)
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        raise ValueError(f"Ref2VA VLM reward requires extra_info['{key}'] to be a list of paths.")
    return list(values)


def _parse_scores(model_output: str) -> dict[str, float]:
    """Extract the last boxed numeric score from the final message content."""
    # Accept one or more literal backslashes so the parser handles both the
    # raw prompt spelling (``\\\\boxed``) and the model's usual LaTeX output
    # (``\\boxed``).
    boxed = re.findall(r"\\+boxed\s*\{([^{}]*)\}", model_output)
    if not boxed:
        return {}
    try:
        score = float(boxed[-1].strip())
    except ValueError:
        return {}
    if np.isfinite(score) and 1.0 <= score <= 5.0:
        return {"instruction_following": score}
    return {}


async def _chat_complete(router_address: str, request: dict[str, Any]) -> ChatCompletion:
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=request) as response:
            response.raise_for_status()
            payload = await response.json()
    return ChatCompletion(**payload)


async def compute_score_vlm(
    data_source: str,
    solution_image: torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
):
    """Score a generated uint8 (T, C, H, W) video against reference images and videos."""
    del data_source, reward_model_tokenizer

    if not reward_router_address:
        raise ValueError("Ref2VA VLM reward requires reward.reward_model.enable=True.")
    if not model_name:
        raise ValueError("Ref2VA VLM reward requires reward.reward_model.model_path.")
    if not isinstance(solution_image, torch.Tensor) or solution_image.ndim != 4:
        shape = getattr(solution_image, "shape", None)
        raise ValueError(f"Ref2VA VLM reward expects one video tensor with four dimensions, got {shape}.")

    extra_info = extra_info or {}
    source_images = _reference_paths(extra_info, "source_images")
    source_videos = _reference_paths(extra_info, "source_videos")
    if not source_images and not source_videos:
        raise ValueError("Ref2VA VLM reward requires at least one reference image or video.")

    from verl_omni.utils.dataset.minimax_h3_video import resolve_video_spec

    fps = resolve_video_spec({"extra_info": extra_info})["frame_rate"]

    loop = asyncio.get_running_loop()
    reference_images, reference_videos, video_url = await asyncio.gather(
        asyncio.gather(*(loop.run_in_executor(None, _media_path_to_data_url, path, "image") for path in source_images)),
        asyncio.gather(*(loop.run_in_executor(None, _media_path_to_data_url, path, "video") for path in source_videos)),
        loop.run_in_executor(None, _video_tensor_to_data_url, solution_image, fps),
    )
    request = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": _build_content(reference_images, reference_videos, video_url, ground_truth or ""),
            }
        ],
    }
    result = await _chat_complete(reward_router_address, request)
    choice = result.choices[0]
    model_output = choice.message.content or ""
    scores = _parse_scores(model_output)
    if choice.finish_reason == "length" or not scores:
        raise ValueError(
            f"Ref2VA VLM reward returned malformed scores (finish_reason={choice.finish_reason!r}): "
            f"{model_output[:512]!r}"
        )

    # Map the judge's inclusive 1-5 scale to the inclusive [0, 1] range.
    normalized_score = (sum(scores.values()) / len(SCORE_KEYS) - 1.0) / 4.0
    return {
        "score": normalized_score,
        **scores,
        "judge_response": model_output,
    }
