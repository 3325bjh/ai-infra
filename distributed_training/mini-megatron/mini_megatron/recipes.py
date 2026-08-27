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

from collections.abc import Callable

from mini_megatron.config import (
    ConfigContainer,
    DataConfig,
    DistributedConfig,
    Qwen36ModelConfig,
    TrainConfig,
)


def qwen36_35b_a3b_tiny_config() -> ConfigContainer:
    """Return a tiny Qwen3.6-35B-A3B-inspired pretraining config.

    Tutorial chapter: 1. Config and recipe.
    Reading route: `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`.
    """
    model = Qwen36ModelConfig(
        vocab_size=151_936,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        hf_model_id="Qwen/Qwen3.6-35B-A3B",
    )
    train = TrainConfig(
        train_iters=3000,
        global_batch_size=16,
        micro_batch_size=2,
        warmup_iters=10,
        dtype="bf16",
    )
    data = DataConfig(
        seq_length=model.max_position_embeddings,
        vocab_size=model.vocab_size,
        mock_num_samples=8192,
    )
    return ConfigContainer(
        model=model,
        train=train,
        data=data,
        distributed=DistributedConfig(),
    )


def qwen36_35b_a3b_debug_config() -> ConfigContainer:
    """Return an even smaller config for CPU smoke tests and notebooks."""
    model = Qwen36ModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        hf_model_id="Qwen/Qwen3.6-35B-A3B",
    )
    train = TrainConfig(
        train_iters=2,
        global_batch_size=2,
        micro_batch_size=1,
        warmup_iters=1,
        weight_decay=0.0,
        dtype="fp32",
        save_dir="mini_experiments/qwen36_debug/checkpoints",
    )
    data = DataConfig(
        seq_length=model.max_position_embeddings,
        vocab_size=model.vocab_size,
        mock_num_samples=32,
        pin_memory=False,
    )
    return ConfigContainer(
        model=model,
        train=train,
        data=data,
        distributed=DistributedConfig(backend="gloo", ddp=False),
    )


RECIPE_REGISTRY: dict[str, Callable[[], ConfigContainer]] = {
    "qwen36_35b_a3b_tiny_config": qwen36_35b_a3b_tiny_config,
    "qwen36_35b_a3b_debug_config": qwen36_35b_a3b_debug_config,
}


def load_recipe(name: str) -> ConfigContainer:
    """Load a mini recipe by name.

    Args:
        name: Recipe function name.

    Returns:
        A config container.
    """
    try:
        recipe_factory = RECIPE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(RECIPE_REGISTRY))
        raise KeyError(f"Unknown recipe {name!r}. Available recipes: {available}") from exc
    return recipe_factory()
