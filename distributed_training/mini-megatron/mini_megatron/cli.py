# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import logging

from mini_megatron.config import ConfigContainer
from mini_megatron.recipes import load_recipe


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Tutorial chapter: 8. Launcher and Slurm.
    Reading route: `scripts/training/run_recipe.py`.
    """
    parser = argparse.ArgumentParser(description="Train the mini Qwen3.6 MoE model.")
    parser.add_argument(
        "--recipe",
        default="qwen36_35b_a3b_tiny_config",
        choices=("qwen36_35b_a3b_tiny_config", "qwen36_35b_a3b_debug_config"),
    )
    parser.add_argument("--dataset-path", default=None, help="Optional JSONL file with `tokens` or `text` records.")
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: ConfigContainer, args: argparse.Namespace) -> ConfigContainer:
    """Apply simple command-line overrides to a recipe config."""
    if args.dataset_path is not None:
        cfg.data.dataset_path = args.dataset_path

    if args.train_iters is not None:
        cfg.train.train_iters = args.train_iters

    if args.seq_length is not None:
        cfg.model.max_position_embeddings = args.seq_length
        cfg.data.seq_length = args.seq_length

    if args.micro_batch_size is not None:
        cfg.train.micro_batch_size = args.micro_batch_size

    if args.global_batch_size is not None:
        cfg.train.global_batch_size = args.global_batch_size

    if args.save_dir is not None:
        cfg.train.save_dir = args.save_dir

    if args.dtype is not None:
        cfg.train.dtype = args.dtype

    cfg.validate()
    return cfg


def main() -> None:
    """Run training with a recipe and simple CLI overrides."""
    args = parse_args()
    cfg = load_recipe(args.recipe)
    cfg = apply_cli_overrides(cfg, args)
    from mini_megatron.trainer import run_training

    stats = run_training(cfg)
    logger.info("finished training: %s", stats)
