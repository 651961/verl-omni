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

"""Prepare the checkpoint's Qwen3-VL inputs once, in the agent loop."""

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def prepare_prompt_inputs(processor, text, images, max_sequence_length):
    """Match the native pipeline's template, image resize and alpha compositing."""
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import OUTPUT_RESOLUTION, calculate_dimensions

    template = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
    )
    text = text or " "
    image_sizes, resized = [], []
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True)
    for image in images or []:
        width, height = calculate_dimensions(OUTPUT_RESOLUTION**2, image.width / image.height)
        image_sizes.append((width, height))
        image = image_processor.resize(image, height=height, width=width)
        if image.mode == "RGBA":
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.getchannel("A"))
            image = white
        resized.append(image)
    if resized:
        prefix = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(resized)))
        text = prefix + text
    inputs = dict(
        processor(
            text=[template.format(text)],
            **({"images": resized} if resized else {}),
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
    )
    # Image expansion belongs to the VLM, not the user-text length budget.
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    non_image_tokens = inputs["attention_mask"].sum() - (inputs["input_ids"] == image_token_id).sum()
    template_tokens = processor.tokenizer(template.format(""), return_tensors="pt")["input_ids"].shape[1]
    if int(non_image_tokens) + len(resized) > max_sequence_length + template_tokens:
        raise ValueError("Qwen-Image 2.1 prompt exceeds pipeline.max_sequence_length.")
    return {
        **{key: value.cpu() for key, value in inputs.items() if isinstance(value, torch.Tensor)},
        "image_sizes": image_sizes,
    }
