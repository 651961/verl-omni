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

"""Token-id-native encoding shared by Qwen-Image 2.1 RL adapters."""

import torch
from torch.nn import functional as F


class QwenImage21TokenIdPromptMixin:
    """Reuse native VLM encoding and layout helpers with agent-prepared inputs."""

    def _extract_prompts(self, prompts):
        if not prompts or isinstance(prompts[0], str):
            return super()._extract_prompts(prompts)
        prepared = [(p.get("mm_processor_kwargs") or {}).get("qwen21") for p in prompts]
        if not any(prepared):
            if any("prompt_ids" in p for p in prompts):
                raise ValueError("Qwen-Image 2.1 RL requires the qwen_image21_agent processor inputs.")
            return super()._extract_prompts(prompts)
        positive = [
            {**item["positive"], "input_ids": torch.as_tensor(p["prompt_ids"]).reshape(1, -1)}
            for p, item in zip(prompts, prepared, strict=True)
        ]
        negative = None
        if prepared[0].get("negative") is not None:
            negative = [
                {**item["negative"], "input_ids": torch.as_tensor(p["negative_prompt_ids"]).reshape(1, -1)}
                for p, item in zip(prompts, prepared, strict=True)
            ]
        return positive, negative

    def _get_qwen_prompt_embeds(self, prompt, images_per_prompt=None, max_sequence_length=None, prompt_name="prompt"):
        if isinstance(prompt, str) or not isinstance(prompt[0], dict):
            # The native engine uses raw text for its warm-up request.
            return super()._get_qwen_prompt_embeds(
                prompt, images_per_prompt, max_sequence_length=max_sequence_length, prompt_name=prompt_name
            )
        width = max(item["input_ids"].shape[1] for item in prompt)
        kwargs = {}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key not in prompt[0]:
                continue
            fill = self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
            kwargs[key] = torch.cat(
                [F.pad(item[key], (width - item[key].shape[1], 0), value=fill) for item in prompt]
            ).to(self.device)
        for key in ("pixel_values", "image_grid_thw"):
            if key in prompt[0]:
                kwargs[key] = torch.cat([item[key] for item in prompt]).to(self.device)
        kwargs["output_hidden_states"] = True
        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            hidden = self.text_encoder(**kwargs).hidden_states[-1]
        finally:
            handle.remove()
        hidden_rows = [row[self._drop_idx :] for row in self._extract_masked_hidden(hidden, kwargs["attention_mask"])]
        image_masks = [
            (ids[mask.bool()] == self._img_token_id)[self._drop_idx :]
            for ids, mask in zip(kwargs["input_ids"], kwargs["attention_mask"], strict=True)
        ]
        if any(item["image_sizes"] for item in prompt):
            image_sizes = [item["image_sizes"] for item in prompt]
            slots = [
                [(h // self.vae_scale_factor) * (w // self.vae_scale_factor) // 4 for w, h in sizes]
                for sizes in image_sizes
            ]
            hidden_rows, image_masks = self._downsample_image_pad_tokens(hidden_rows, image_masks, slots, image_sizes)
        width = max(row.shape[0] for row in hidden_rows)
        embeds = torch.stack([F.pad(row, (0, 0, 0, width - row.shape[0])) for row in hidden_rows])
        masks = torch.stack(
            [
                F.pad(torch.ones(row.shape[0], device=self.device, dtype=torch.long), (0, width - row.shape[0]))
                for row in hidden_rows
            ]
        )
        image_masks = torch.stack([F.pad(row, (0, width - row.shape[0])) for row in image_masks])
        return embeds.to(self.text_encoder.dtype), masks, image_masks
