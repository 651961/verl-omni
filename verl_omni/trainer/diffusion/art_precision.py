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

"""Fail early when the agreed full-parameter ART precision contract is violated."""


def validate_art_precision(model, actor):
    """Validate FP32 master/optimizer storage, BF16 compute and supported execution modes."""
    engine = actor.get("fsdp_config", {})
    if actor.get("strategy") not in ("fsdp", "fsdp2"):
        raise ValueError("ART requires FSDP/FSDP2.")
    if model.get("lora_rank", 0) or model.get("use_regional_compile", False):
        raise ValueError("This ART integration requires full parameters and regional_compile=false.")
    if engine.get("model_dtype", "fp32") not in ("fp32", "float32"):
        raise ValueError("ART master weights must be FP32 (fsdp_config.model_dtype=fp32).")
    mp = engine.get("mixed_precision") or {}
    if mp.get("param_dtype", "bf16") not in ("bf16", "bfloat16"):
        raise ValueError("ART model computation must use BF16.")
    if mp.get("reduce_dtype", "fp32") not in ("fp32", "float32"):
        raise ValueError("ART gradient synchronization must use FP32.")
    if actor.get("ppo_micro_batch_size_per_gpu") != 1:
        raise ValueError("ART requires ppo_micro_batch_size_per_gpu=1.")
    if engine.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("ART currently requires training SP=1.")
    if actor.get("use_no_sync_for_gradient_accumulation", False) or engine.get("offload_policy", False):
        raise ValueError("ART requires per-backward gradient synchronization and no FSDP CPUOffloadPolicy.")
    if actor.get("optim", {}).get("optimizer", "adamw").lower() not in ("adam", "adamw"):
        raise ValueError("ART precision contract requires FP32 Adam states.")
    if actor.get("optim", {}).get("optimizer_impl", "torch.optim") != "torch.optim":
        raise ValueError("ART requires torch.optim Adam/AdamW to retain FP32 optimizer states.")
    if "old" in model.get("policy_state_adapters", ["default"]):
        raise ValueError("ART uses frozen rollout caches and must not load an old/EMA adapter.")
