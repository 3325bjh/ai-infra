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
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from mini_megatron.config import ConfigContainer
from mini_megatron.data import build_dataloader
from mini_megatron.distributed import (
    DistributedContext,
    cleanup_distributed,
    init_distributed,
    rank0_log,
)
from mini_megatron.forward_step import causal_lm_forward_step
from mini_megatron.modeling_qwen36 import Qwen36ForCausalLM
from mini_megatron.training_core import (
    MiniOptimizerParamScheduler,
    OptimizerLike,
    SchedulerLike,
    setup_model_and_optimizer,
    train_step,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: str, dtype: str):
    """Return a CUDA autocast context or a no-op context."""
    if device.startswith("cuda") and dtype in {"bf16", "fp16"}:
        autocast_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[dtype]

        return torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
        )

    if dtype == "fp32" or not device.startswith("cuda"):
        return nullcontext()

    raise ValueError(f"unsupported dtype: {dtype}")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a batch dictionary to the selected device."""
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped by DDP."""
    return getattr(model, "module", model)



def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    cfg: ConfigContainer,
    ctx: DistributedContext,
) -> None:
    """Save a compact single-rank checkpoint on rank 0."""
    if not ctx.is_rank0:
        return

    checkpoint_dir = Path(cfg.train.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"iter_{iteration}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
        },
        checkpoint_path,
    )


def build_model(cfg: ConfigContainer, device: torch.device) -> nn.Module:
    """Build the Qwen3.6 mini model on the requested device."""
    return Qwen36ForCausalLM(cfg.model).to(device)


def maybe_wrap_ddp(model: nn.Module, cfg: ConfigContainer, ctx: DistributedContext) -> nn.Module:
    """Wrap model in PyTorch DDP when distributed training is active."""
    if not cfg.distributed.ddp or not ctx.is_distributed:
        return model

    if ctx.device.startswith("cuda"):
        return DDP(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
        )

    return DDP(model)



def run_training(cfg: ConfigContainer) -> dict[str, Any]:
    """Run the mini causal-LM training loop.

    Tutorial chapter: 7. Trainer wiring.
    Reading route:
    - `src/megatron/bridge/training/setup.py`
    - `src/megatron/bridge/training/train.py`

    Args:
        cfg: Validated mini config.

    Returns:
        Dictionary with final iteration and loss for smoke tests or notebooks.
    """
    ctx = init_distributed(cfg.distributed)
    set_seed(cfg.train.seed + ctx.rank)
    device = torch.device(ctx.device)

    try:
        def model_provider(
            _cfg: ConfigContainer,
            _ctx: DistributedContext,
        ) -> nn.Module:
            model = build_model(_cfg, device)
            return maybe_wrap_ddp(model, _cfg, _ctx)

        def optimizer_provider(
            _cfg: ConfigContainer,
            model: object,
        ) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                model.parameters(),  # type: ignore[union-attr]
                lr=_cfg.train.max_lr,
                weight_decay=_cfg.train.weight_decay,
            )

        def scheduler_provider(
            _cfg: ConfigContainer,
            optimizer: OptimizerLike,
        ) -> SchedulerLike:
            return MiniOptimizerParamScheduler(optimizer, _cfg.train)

        def data_iterator_provider(
            _cfg: ConfigContainer,
            _ctx: DistributedContext,
        ):
            return iter(
                build_dataloader(
                    _cfg.data,
                    _ctx,
                    seed=_cfg.train.seed,
                    micro_batch_size=_cfg.train.micro_batch_size,
                )
            )

        setup = setup_model_and_optimizer(
            cfg,
            ctx,
            model_provider=model_provider,
            optimizer_provider=optimizer_provider,
            scheduler_provider=scheduler_provider,
            data_iterator_provider=data_iterator_provider,
        )

        def forward_step_with_runtime(
            batch: object,
            model: object,
        ) -> tuple[object, dict[str, float]]:
            if not isinstance(batch, dict):
                raise TypeError("training batches must be dictionaries of tensors")

            moved_batch = _move_batch(batch, device)
            with _autocast_context(ctx.device, cfg.train.dtype):
                return causal_lm_forward_step(moved_batch, model)

        def clip_gradients(model: object, max_norm: float) -> float:
            parameters = model.parameters()  # type: ignore[union-attr]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
            return float(grad_norm)

        last_loss: dict[str, float] = {}
        last_consumed_samples = 0

        for iteration in range(1, cfg.train.train_iters + 1):
            result = train_step(
                forward_step_with_runtime,
                setup,
                cfg,
                grad_clip_func=clip_gradients,
            )
            last_loss = result.loss_dict
            last_consumed_samples = result.consumed_samples

            if iteration % cfg.train.log_interval == 0:
                rank0_log(
                    ctx,
                    "iteration=%d loss=%s grad_norm=%s consumed_samples=%d",
                    iteration,
                    last_loss,
                    result.grad_norm,
                    result.consumed_samples,
                )

            if iteration % cfg.train.save_interval == 0:
                save_checkpoint(
                    model=setup.model,  # type: ignore[arg-type]
                    optimizer=setup.optimizer,  # type: ignore[arg-type]
                    iteration=iteration,
                    cfg=cfg,
                    ctx=ctx,
                )

        return {
            "iteration": cfg.train.train_iters,
            "loss": last_loss,
            "consumed_samples": last_consumed_samples,
        }
    finally:
        cleanup_distributed(cfg.distributed)

