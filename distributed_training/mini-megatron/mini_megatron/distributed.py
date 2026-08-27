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

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from mini_megatron.config import DistributedConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    """Resolved process identity for torchrun or Slurm launches."""

    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_distributed(self) -> bool:
        """Return whether this process participates in a distributed job."""
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _get_int_env(primary: str, fallback: str | None, default: int) -> int:
    """Resolve an integer environment variable with an optional fallback."""
    value = os.environ.get(primary)
    if value is None and fallback is not None:
        value = os.environ.get(fallback)
    if value is None:
        return default
    return int(value)


def _torch_cuda_device(local_rank: int) -> str:
    """Return `cuda:{local_rank}` when CUDA is available, otherwise `cpu`."""
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def resolve_distributed_context() -> DistributedContext:
    """Resolve rank information from torchrun or Slurm environment variables."""
    rank = _get_int_env("RANK", "SLURM_PROCID", 0)
    local_rank = _get_int_env("LOCAL_RANK", "SLURM_LOCALID", 0)
    world_size = _get_int_env("WORLD_SIZE", "SLURM_NTASKS", 1)

    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if local_rank < 0:
        raise ValueError(f"local_rank must be non-negative, got {local_rank}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=_torch_cuda_device(local_rank),
    )


def init_distributed(config: DistributedConfig) -> DistributedContext:
    """Initialize torch.distributed when launched with more than one process."""
    ctx = resolve_distributed_context()

    if not ctx.is_distributed:
        return ctx
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable for a multi-process launch")
    if dist.is_initialized():
        return ctx

    backend = config.backend
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back from nccl to gloo.")
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=config.init_method,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )
    return ctx


def barrier() -> None:
    """Synchronize all distributed ranks if the process group is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(config: DistributedConfig) -> None:
    """Destroy the process group if this trainer owns it."""
    if not config.destroy_process_group_on_exit:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_log(ctx: DistributedContext, message: str, *args: object) -> None:
    """Log a message only on global rank 0."""
    if ctx.is_rank0:
        logger.info(message, *args)

