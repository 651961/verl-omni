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

"""One H3 ART actor update with synthetic variable-size FL2VA/Ref2VA groups."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.special_e2e.build_minimax_h3_tiny_random import ensure_tiny_minimax_h3_checkpoint  # noqa: E402
from tests.special_e2e.create_dummy_h3_data import (  # noqa: E402
    build_dummy_h3_fl2va_data,
    build_dummy_h3_ref2va_data,
)
from tests.special_e2e.run_flowgrpo_minimax_h3_tiny import _tiny_patch_environment  # noqa: E402


def main():
    """Build tiny weights/data and run the production ART recipe in separate GPU pools."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--work-dir", default=str(ROOT / "outputs/h3_art_smoke"))
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    model = ensure_tiny_minimax_h3_checkpoint(str(work / "model"))
    builder = build_dummy_h3_fl2va_data if args.task == "fl2va" else build_dummy_h3_ref2va_data
    train, val = builder(str(work / args.task / "data"), train_size=4, val_size=2, image_width=256, image_height=256)
    for filename in (train, val):
        frame = pd.read_parquet(filename)
        frame["extra_info"] = [
            {**dict(info), "height": 160 + (index % 2) * 32, "width": 288, "num_frames": 97 + (index % 2) * 23}
            for index, info in enumerate(frame["extra_info"])
        ]
        frame.to_parquet(filename, index=False)
    env = _tiny_patch_environment(model, args.task)
    env.update(
        {
            "MODEL_PATH": model,
            "ACTOR_CONFIG_PATH": str(Path(model) / "transformer"),
            "TRAIN_FILE": train,
            "VAL_FILE": val,
            "REWARD_FUNCTION_PATH": str(ROOT / "tests/special_e2e/minimax_h3_dummy_reward.py"),
            "REWARD_FUNCTION_NAME": "compute_score",
            "ACTOR_GPUS_PER_NODE": "2",
            "ACTOR_NNODES": "1",
            "ROLLOUT_GPUS_PER_NODE": "2",
            "ROLLOUT_NNODES": "1",
            "ROLLOUT_TP": "1",
            "TEXT_ENCODER_TP": "1",
            "PROMPTS_PER_BATCH": "2",
            "GROUP_SIZE": "2",
            "ART_STEP_INDICES": "[0,1]",
            "OUTPUT_DIR": str(work / args.task),
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        }
    )
    subprocess.run(
        [
            "bash",
            str(ROOT / f"examples/diffusionart_trainer/minimax_h3/run_minimax_h3_{args.task}.sh"),
            "trainer.total_training_steps=1",
            "trainer.val_before_train=false",
            "trainer.test_freq=-1",
            "trainer.save_freq=-1",
            "actor_rollout_ref.model.attn_backend=native",
            "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
            "actor_rollout_ref.rollout.pipeline.num_inference_steps=2",
            "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=2",
            "trainer.logger=console",
            "trainer.resume_mode=disable",
            "ray_kwargs.ray_init.num_cpus=40",
            "actor_rollout_ref.rollout.pipeline.reference_image_short_edge=256",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=true",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
