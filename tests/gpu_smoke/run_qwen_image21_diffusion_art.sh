#!/usr/bin/env bash
# Manual one-step native rollout / reward / FSDP smoke using the ART recipe.
# Requires a native Qwen-Image 2.1 vLLM-Omni build, model/data/reward paths,
# and the recipe's actor and rollout GPU pools (8 actor GPUs per node).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$repo_root/examples/diffusionart_trainer/qwen_image21/run_qwen_image21.sh" \
    trainer.total_training_steps=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "$@"
