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

"""ART adapter contracts against the installed native pipeline, without model weights."""

from __future__ import annotations

import ast
import copy
import operator
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/trainer/diffusion"))
from test_art_core_on_cpu import capture, recorder  # noqa: E402

native_pipeline = pytest.importorskip("vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21")
NATIVE_ROOT = Path(native_pipeline.__file__).resolve().parents[3]


def native_class():
    """Execute upstream method bodies with component and transport test doubles."""
    namespace = {
        "torch": torch,
        "copy": copy,
        "F": torch.nn.functional,
        "OUTPUT_RESOLUTION": 32,
        "normalize_min_aligned_size": lambda h, w, scale: (h, w),
        "split_diffusion_output_by_request": lambda output, req, **kwargs: [
            types.SimpleNamespace(output=output.output[row : row + 1]) for row in range(len(req.prompts))
        ],
    }
    path = NATIVE_ROOT / "diffusion/models/qwen_image_21/pipeline_qwen_image_21.py"
    methods = {
        "forward",
        "prepare_encode",
        "denoise_step",
        "_denoise_step_homogeneous",
        "step_scheduler",
        "post_decode",
        "_extract_prompts",
        "_extract_request_images",
        "_append_target_slots",
        "_assemble_kv_cache",
        "_scatter_kv_cache",
        "_kv_cache_phase",
        "_decode_group_key",
        "_split_decode_groups",
        "_needs_decode_seq_len_split",
        "_resolve_prompt_seq_len",
        "_pad_rows_to_seq_len",
    }
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "QwenImage21Pipeline"
    )
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    node.body = [child for child in node.body if getattr(child, "name", None) in methods]
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    cls = namespace["QwenImage21Pipeline"]
    cfg_path = NATIVE_ROOT / "diffusion/models/qwen_image_21/cfg_parallel.py"
    node = next(node for node in ast.parse(cfg_path.read_text()).body if isinstance(node, ast.ClassDef))
    node.bases = [ast.Name(id="object", ctx=ast.Load())]
    node.body = [child for child in node.body if getattr(child, "name", None) == "diffuse"]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(cfg_path), "exec"), namespace)
    cls.diffuse = namespace[node.name].diffuse
    cls.supports_request_batch = True
    cls.supports_step_execution = True
    return cls


class Transformer(torch.nn.Module):
    """A deterministic prediction and native dictionary-KV boundary double."""

    causal_condition = True
    transformer_blocks = [object()]
    in_channels = 4

    def forward(self, hidden_states, encoder_hidden_states, kv_cache=None, cache_branch="cond", **kwargs):
        if kv_cache is not None:
            kv_cache[0].setdefault(
                cache_branch,
                {
                    "key": torch.ones(hidden_states.shape[0], encoder_hidden_states.shape[1], 1, 1),
                    "value": torch.ones(hidden_states.shape[0], encoder_hidden_states.shape[1], 1, 1),
                },
            )
        return (hidden_states * 0.25 + encoder_hidden_states.mean((1, 2))[:, None, None],)


def pipeline_class():
    native = native_class()

    def initialize(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.transformer = Transformer()
        self.vae_scale_factor = 16
        self.device = "cpu"
        self._max_length = 8192
        self._attention_kwargs = {}
        self.graph_releases = 0

    def prepare_context(self, prompt, height, width, generator, true_cfg_scale, **kwargs):
        batch = len(prompt)
        tokens = height * width // 256
        generators = generator if isinstance(generator, list) else [generator] * batch
        latents = torch.cat([torch.randn(1, tokens, 4, generator=g).bfloat16() for g in generators])
        embeds = torch.stack([torch.full((3, 8), float(p)).bfloat16() for p in prompt])
        image_pad_mask = torch.zeros(batch, 3, dtype=torch.bool)
        self.scheduler.set_timesteps(2, sigmas=[1.0, 0.5])
        cfg = true_cfg_scale > 1
        return {
            "latents": latents,
            "image_latents": None,
            "timesteps": self.scheduler.timesteps,
            "prompt_embeds": embeds,
            "prompt_embeds_mask": torch.ones(batch, 3, dtype=torch.bool),
            "negative_prompt_embeds": embeds * 0.5 if cfg else None,
            "negative_prompt_embeds_mask": torch.ones(batch, 3, dtype=torch.bool) if cfg else None,
            "image_pad_mask": image_pad_mask,
            "negative_image_pad_mask": image_pad_mask if cfg else None,
            "img_mask": self._append_target_slots(image_pad_mask, tokens),
            "negative_img_mask": self._append_target_slots(image_pad_mask, tokens) if cfg else None,
            "img_shapes": [[(1, height // 16, width // 16)]] * batch,
            "do_true_cfg": cfg,
        }

    def predict(self, do_cfg, scale, positive, negative, cfg_normalize=False, output_slice=None):
        result = self.transformer(**positive)[0][:, -output_slice:]
        if do_cfg:
            uncond = self.transformer(**negative)[0][:, -output_slice:]
            result = uncond + scale * (result - uncond)
        return result

    native.__init__ = initialize
    native._prepare_generation_context = prepare_context
    native.predict_noise_maybe_with_cfg = predict
    native.scheduler_step_maybe_with_cfg = lambda self, pred, t, latents, cfg, per_request_scheduler=None: (
        per_request_scheduler or self.scheduler
    ).step(pred, t, latents, return_dict=False)[0]
    native._decode_latents = lambda self, latents, *args: types.SimpleNamespace(output=latents)
    native.progress_bar = lambda self, **kwargs: nullcontext(types.SimpleNamespace(update=lambda: None))
    native.interrupt = False
    native.attention_kwargs = {}
    native.release_captured_graphs = lambda self: setattr(self, "graph_releases", self.graph_releases + 1)
    common = {}
    common_path = ROOT / "verl_omni/pipelines/qwen_image21/common.py"
    exec(compile(common_path.read_text(), str(common_path), "exec"), common)

    def envelope(output, rl=None, **kwargs):
        return types.SimpleNamespace(output=output.output, cache=(rl or {}).get("art_cache"))

    path = ROOT / "verl_omni/pipelines/qwen_image21_diffusion_art/vllm_omni_rollout_adapter.py"
    namespace = {
        "torch": torch,
        "QwenImage21Pipeline": native,
        "QwenImage21TokenIdPromptMixin": common["QwenImage21TokenIdPromptMixin"],
        "ARTTrajectoryRecorder": recorder.ARTTrajectoryRecorder,
        "frozen_cpu": recorder.frozen_cpu,
        "capture_art_rollout": capture.capture_art_rollout,
        "with_rollout_data": envelope,
        "with_visual_artifacts": lambda output, **kwargs: output,
        "DiffusionIOSpec": lambda **kwargs: None,
        "MediaSpec": lambda *args: None,
    }
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[node.name]


def sampling(seed, cfg=1.0, capture_trajectory=True):
    return types.SimpleNamespace(
        extra_args={"art_step_indices": [1], "art_capture_trajectory": capture_trajectory},
        true_cfg_scale=cfg,
        num_outputs_per_prompt=1,
        height=32,
        width=32,
        num_inference_steps=2,
        sigmas=None,
        generator=None,
        seed=seed,
        max_sequence_length=8192,
        output_type="latent",
        cfg_normalize=False,
    )


def state(prompt, params):
    return types.SimpleNamespace(
        sampling=params,
        prompt=prompt,
        extra={},
        request_id=str(params.seed),
        current_timestep=None,
        step_index=0,
    )


def request(params):
    return types.SimpleNamespace(
        sampling_params_list=params,
        prompts=[str(i + 1) for i in range(len(params))],
        collate_request_generators=lambda *args: [torch.Generator().manual_seed(p.seed) for p in params],
        collate_request_tensors=lambda *args: None,
    )


class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = pipeline_class()

    def test_native_single_request_and_step_outputs_match_cfg_sizes_and_versions(self):
        for cfg in (1.0, 2.0):
            full = self.pipeline()
            full.set_published_policy_version(17)
            self.assertEqual(full.graph_releases, 1)
            step = self.pipeline()
            step.set_published_policy_version(17)
            for width in (32, 64):
                params = sampling(10, cfg)
                params.width = width
                expected = full.forward(request([params]))[0]
                s = step.prepare_encode(state("1", copy.deepcopy(params)))
                for index in range(2):
                    s.current_timestep = s.timesteps[index]
                    batch = types.SimpleNamespace(
                        states=[s],
                        latents=s.latents,
                        timesteps=s.current_timestep,
                        do_true_cfg=s.do_true_cfg,
                        true_cfg_scale=cfg,
                        cfg_normalize=False,
                        prompt_embeds=s.prompt_embeds,
                        prompt_embeds_mask=s.prompt_embeds_mask,
                        negative_prompt_embeds=s.negative_prompt_embeds,
                        negative_prompt_embeds_mask=s.negative_prompt_embeds_mask,
                        img_shapes=s.img_shapes,
                    )
                    pred = step.denoise_step(batch)
                    step.step_scheduler(s, pred)
                actual = step.post_decode(s)
                torch.testing.assert_close(actual.output, expected.output, rtol=0, atol=0)
                for key in ("states", "old_velocities", "timesteps", "sigmas", "dt", "endpoints"):
                    torch.testing.assert_close(actual.cache[key], expected.cache[key], rtol=0, atol=0)
                for branch in ("inputs", "negative_inputs") if cfg > 1 else ("inputs",):
                    self.assertEqual(
                        actual.cache["conditions"][branch]["img_shapes"],
                        expected.cache["conditions"][branch]["img_shapes"],
                    )
                    for key in ("encoder_hidden_states", "encoder_hidden_states_mask", "img_mask"):
                        torch.testing.assert_close(
                            actual.cache["conditions"][branch][key], expected.cache["conditions"][branch][key]
                        )
                self.assertEqual(actual.cache["policy_version"], 17)
                self.assertEqual(actual.cache["conditions"]["negative_inputs"] is not None, cfg > 1)

    def test_validation_does_not_collect_trajectories_or_hooks(self):
        pipeline = self.pipeline()
        output = pipeline.forward(request([sampling(3, capture_trajectory=False)]))[0]
        self.assertIsNone(output.cache)
        self.assertEqual(len(pipeline.transformer._forward_pre_hooks), 0)
        s = pipeline.prepare_encode(state("1", sampling(3, capture_trajectory=False)))
        self.assertNotIn("art_recorder", s.extra)

    def test_config_requires_one_request_including_engine_overrides(self):
        from unittest.mock import patch

        path = ROOT / "verl_omni/pipelines/qwen_image21_diffusion_art/diffusers_training_adapter.py"
        cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if getattr(node, "name", None) == "validate_art_config")
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        rollout = types.SimpleNamespace(max_num_seqs=1, engine_kwargs={})
        config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(rollout=rollout))
        with patch("importlib.util.find_spec", return_value=object()):
            namespace["validate_art_config"](None, config)
            for max_seqs, overrides in ((2, {}), (1, {"max_num_seqs": 2}), (1, {"max-num-seqs": 2})):
                rollout.max_num_seqs = max_seqs
                rollout.engine_kwargs = {"vllm_omni": overrides}
                with self.subTest(max_num_seqs=max_seqs, overrides=overrides):
                    with self.assertRaisesRegex(ValueError, "max_num_seqs=1"):
                        namespace["validate_art_config"](None, config)

    def test_token_id_encoder_does_not_retokenize_and_keeps_norm_inputs(self):
        pipeline = self.pipeline()
        pipeline._drop_idx = 1
        pipeline._img_token_id = 99
        tokenizer = types.SimpleNamespace(pad_token_id=0)
        pipeline.processor = types.SimpleNamespace(tokenizer=tokenizer)

        class Encoder(torch.nn.Module):
            dtype = torch.bfloat16

            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace(language_model=types.SimpleNamespace(norm=torch.nn.Identity()))
                self.seen = None

            def forward(self, input_ids, **kwargs):
                self.seen = input_ids
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
                return types.SimpleNamespace(hidden_states=[self.model.language_model.norm(hidden)])

        pipeline.text_encoder = Encoder()
        pipeline._extract_masked_hidden = lambda h, m: [row[mask.bool()] for row, mask in zip(h, m, strict=True)]
        payload = [
            {
                "prompt_ids": [1, 2, 3],
                "mm_processor_kwargs": {
                    "qwen21": {"positive": {"attention_mask": torch.ones(1, 3), "image_sizes": []}}
                },
            },
            {
                "prompt_ids": [1, 4],
                "mm_processor_kwargs": {
                    "qwen21": {"positive": {"attention_mask": torch.ones(1, 2), "image_sizes": []}}
                },
            },
        ]
        positive, negative = pipeline._extract_prompts(payload)
        embeds, mask, image_mask = pipeline._get_qwen_prompt_embeds(positive)
        self.assertIsNone(negative)
        torch.testing.assert_close(pipeline.text_encoder.seen, torch.tensor([[1, 2, 3], [0, 1, 4]]))
        torch.testing.assert_close(embeds[:, :, 0], torch.tensor([[2, 3], [4, 0]], dtype=torch.bfloat16))
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])
        self.assertFalse(image_mask.any())
        self.assertEqual(len(pipeline.text_encoder.model.language_model.norm._forward_hooks), 0)

    def test_prompt_processor_matches_native_image_resize_and_white_alpha(self):
        import math
        from unittest.mock import patch

        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image

        namespace = {
            "torch": torch,
            "Image": Image,
            "VaeImageProcessor": VaeImageProcessor,
            "OUTPUT_RESOLUTION": 1024,
            "SYSTEM_PROMPT": "Comprehend and analyze the provided prompt.",
            "math": math,
        }
        path = NATIVE_ROOT / "diffusion/models/qwen_image/pipeline_qwen_image_edit.py"
        node = next(
            node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "calculate_dimensions"
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
        path = ROOT / "verl_omni/pipelines/qwen_image21/prompt.py"
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)

        class Tokenizer:
            def convert_tokens_to_ids(self, token):
                return 99

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.ones(1, 2, dtype=torch.long)}

        class Processor:
            tokenizer = Tokenizer()

            def __call__(self, **kwargs):
                self.seen = kwargs
                return {"input_ids": torch.tensor([[1, 99, 99, 2]]), "attention_mask": torch.ones(1, 4)}

        processor = Processor()
        image = Image.new("RGBA", (4, 2), (255, 0, 0, 0))
        module = "vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21"
        with patch.dict(
            sys.modules,
            {
                module: types.SimpleNamespace(
                    OUTPUT_RESOLUTION=1024, calculate_dimensions=namespace["calculate_dimensions"]
                )
            },
        ):
            result = namespace["prepare_prompt_inputs"](processor, "", [image], 8)
        expected_size = namespace["calculate_dimensions"](1024**2, 2.0)
        self.assertEqual(result["image_sizes"], [expected_size])
        self.assertEqual(processor.seen["images"][0].size, expected_size)
        self.assertEqual(processor.seen["images"][0].mode, "RGB")
        self.assertEqual(processor.seen["images"][0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0, 0))
        self.assertIn("<image1><|vision_start|><|image_pad|><|vision_end|> <|im_end|>", processor.seen["text"][0])

    def test_agent_requires_dataset_geometry_for_training_and_validation(self):
        import asyncio
        from unittest.mock import AsyncMock

        calls = []

        class Base:
            async def run(self, sampling_params, **kwargs):
                calls.append(sampling_params)
                return sampling_params, kwargs

        namespace = {"operator": operator, "DiffusionSingleTurnAgentLoop": Base}
        path = ROOT / "verl_omni/pipelines/qwen_image21/agent_loop.py"
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
        loop = namespace[node.name]()
        loop.process_multi_modal_info = AsyncMock(
            return_value={"images": [types.SimpleNamespace(width=400, height=200)]}
        )
        inputs = {"height": 512, "width": 512, "true_cfg_scale": 1.0}
        rows = (
            {"height": None, "width": None, "extra_info": {"height": 736, "width": 1280}},
            {"height": 1024, "width": 1536, "extra_info": {"height": 736, "width": 1280}},
        )
        before = copy.deepcopy(rows)
        for validate in (False, True):
            for row, size in zip(rows, ((736, 1280), (1024, 1536)), strict=True):
                actual, kwargs = asyncio.run(
                    loop.run(
                        inputs,
                        raw_prompt=[{"role": "user", "content": "paint"}],
                        raw_negative_prompt=[{"role": "user", "content": "bad"}],
                        validate=validate,
                        **row,
                    )
                )
                self.assertEqual((actual["height"], actual["width"]), size)
                self.assertEqual(actual["art_capture_trajectory"], not validate)
                self.assertNotIn("raw_negative_prompt", kwargs)
        self.assertEqual(inputs, {"height": 512, "width": 512, "true_cfg_scale": 1.0})
        self.assertEqual(rows, before)

        calls.clear()
        loop.process_multi_modal_info.reset_mock()
        for validate in (False, True):
            for key in ("height", "width"):
                for value in (None, 0, -32, 720, 32.5, True, "1024"):
                    row = {"height": 1024, "width": 1536, key: value}
                    with self.assertRaisesRegex(ValueError, key):
                        asyncio.run(loop.run(inputs, extra_info=row, validate=validate))
                row = {"height": 1024, "width": 1536}
                del row[key]
                with self.assertRaisesRegex(ValueError, f"requires {key}"):
                    asyncio.run(loop.run(inputs, extra_info=row, validate=validate))
        self.assertEqual(calls, [])
        loop.process_multi_modal_info.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
