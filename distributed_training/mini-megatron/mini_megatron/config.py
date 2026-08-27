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

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class Qwen36ModelConfig:
    """Qwen3.6-style causal language model configuration skeleton.

    Tutorial chapter: 1. Config and recipe.
    Reading route: `src/megatron/bridge/training/config.py`.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int
    hf_model_id: str = "Qwen/Qwen3.6-35B-A3B"
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    moe_layer_frequency: int = 1
    router_aux_loss_coef: float = 0.0

    def __post_init__(self) -> None:
        """Validate shape constraints after construction."""
        self.validate()

    @property
    def head_dim(self) -> int:
        """Return per-head hidden dimension."""
        return self.hidden_size//self.num_attention_heads

    def validate(self) -> None:
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must be <= num_experts")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        positive_int_fields = [
            'vocab_size', 'hidden_size', 'intermediate_size',
            'num_layers', 'num_attention_heads', 'num_key_value_heads',
            'num_experts', 'num_experts_per_tok', 'max_position_embeddings'
        ]
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

@dataclass(kw_only=True)
class TrainConfig:
    """Training loop settings skeleton."""

    train_iters: int
    micro_batch_size: int
    global_batch_size: int
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 1
    save_interval: int = 100
    eval_interval: int = 0
    dtype: str = "bf16"
    seed: int = 1234
    save_dir: str = "mini_experiments/qwen36_tiny/checkpoints"

    def __post_init__(self) -> None:
        """Validate train settings after construction."""
        self.validate()

    def validate(self) -> None:
        """Validate core batch and optimization settings."""
        if self.train_iters <= 0:
            raise ValueError("train_iters must be positive")
        if self.micro_batch_size <= 0 or self.global_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by micro_batch_size")
        if self.max_lr <= 0 or self.min_lr < 0:
            raise ValueError("max_lr must be positive and min_lr must be non-negative")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr must be less than max_lr")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.log_interval <= 0 or self.save_interval <= 0:
            raise ValueError("log_interval and save_interval must be positive")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be non-negative")
        if self.dtype not in ["bf16","fp16","fp32"]:
            raise ValueError("dtype must in [bf16,fp16,fp32]")


@dataclass(kw_only=True)
class DataConfig:
    """Dataset settings for mock-token or JSONL-token training skeleton."""

    seq_length: int
    vocab_size: int
    dataset_path: str | None = None
    mock_num_samples: int = 8192
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """Validate data settings after construction."""
        self.validate()

    def validate(self) -> None:
        """Validate data settings."""
        positive_int_fields = [
            "seq_length", "vocab_size", "mock_num_samples"
        ]
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive,got{value}")


@dataclass(kw_only=True)
class DistributedConfig:
    """Distributed runtime settings skeleton."""

    backend: str = "nccl"
    init_method: str = "env://"
    ddp: bool = True
    destroy_process_group_on_exit: bool = True


@dataclass(kw_only=True)
class ConfigContainer:
    """Top-level mini training configuration skeleton."""

    model: Qwen36ModelConfig
    train: TrainConfig
    data: DataConfig
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def __post_init__(self) -> None:
        """Validate cross-section constraints after construction."""
        self.validate()

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        self.data.validate()
        if self.model.vocab_size!=self.data.vocab_size:
            raise ValueError("model.vocab_size must equal data.vocab_size")
        if self.model.max_position_embeddings!=self.data.seq_length:
            raise ValueError("model.max_position_embeddings must equal data.seq_length")

    def gradient_accumulation_steps(self, world_size: int) -> int:
        """Return the number of microbatches accumulated per optimizer step.

        Args:
            world_size: Number of data-parallel workers.

        Returns:
            Positive gradient accumulation count.
        """
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        denominator = self.train.micro_batch_size * world_size
        if self.train.global_batch_size % denominator != 0:
            raise ValueError(
                "global_batch_size must be divisible by "
                "micro_batch_size * world_size"
            )

        return self.train.global_batch_size // denominator
