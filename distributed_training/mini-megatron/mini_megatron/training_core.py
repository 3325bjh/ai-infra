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


import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from mini_megatron.config import ConfigContainer, TrainConfig
from mini_megatron.distributed import DistributedContext
import torch.nn as nn


class OptimizerLike(Protocol):
    """Minimal optimizer surface used by the training core."""

    param_groups: list[dict[str, Any]]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear gradients before the forward-backward pass."""
        ...


    def step(self) -> object:
        """Apply one optimizer update."""


class SchedulerLike(Protocol):
    """Minimal sample-count scheduler surface."""

    def step(self, *, increment: int) -> None:
        """Advance the schedule by consumed global samples."""
        ...


ForwardStepFunc = Callable[[object, object], tuple[object, Mapping[str, float]]]
ForwardBackwardFunc = Callable[..., list[Mapping[str, float]]]


@dataclass(kw_only=True)
class TrainingSetup:
    """Objects materialized before entering the training loop."""

    model: object
    optimizer: OptimizerLike
    scheduler: SchedulerLike
    data_iterator: Iterator[object]
    num_microbatches: int
    world_size: int


@dataclass(frozen=True, kw_only=True)
class TrainStepResult:
    """Small return object for logging and tests."""

    loss_dict: dict[str, float]
    consumed_samples: int
    grad_norm: float | None = None


class MiniOptimizerParamScheduler:
    """Iteration LR scheduler skeleton that advances by Megatron-style sample counts."""

    def __init__(self, optimizer: OptimizerLike, train_config: TrainConfig) -> None:
        """Store optimizer and train config, then initialize LR state."""
        self.optimizer=optimizer
        self.train_config=train_config
        self.samples_seen=0
        self.iteration=0

    def step(self, *, increment: int) -> None:
        """Update the learning rate and sample counter after an optimizer step."""
        self.samples_seen+=increment
        self.iteration+=1
        lr=cosine_lr(self.iteration,self.train_config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"]=lr



def cosine_lr(iteration: int, train_config: TrainConfig) -> float:
    """Return warmup plus cosine-decay learning rate."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")

    warmup_iters = train_config.warmup_iters
    train_iters = train_config.train_iters
    max_lr = train_config.max_lr
    min_lr = train_config.min_lr

    # Linear warmup: 0 -> max_lr
    if warmup_iters > 0 and iteration < warmup_iters:
        return max_lr * iteration / warmup_iters

    # After training: keep min_lr
    if iteration >= train_iters:
        return min_lr

    # Avoid an invalid decay interval.
    if train_iters <= warmup_iters:
        return min_lr

    # Cosine decay: max_lr -> min_lr
    progress = (iteration - warmup_iters) / (train_iters - warmup_iters)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (max_lr - min_lr) * cosine_factor



def setup_model_and_optimizer(
    cfg: ConfigContainer,
    ctx: DistributedContext,
    *,
    model_provider: Callable[[ConfigContainer, DistributedContext], object],
    optimizer_provider: Callable[[ConfigContainer, object], OptimizerLike],
    scheduler_provider: Callable[[ConfigContainer, OptimizerLike], SchedulerLike],
    data_iterator_provider: Callable[[ConfigContainer, DistributedContext], Iterator[object]],
) -> TrainingSetup:
    """Materialize model, optimizer, scheduler, and data iterator.

    Tutorial chapter: 6. Megatron-LM training core.
    Reading route: `3rdparty/Megatron-LM/megatron/training/training.py`.
    """
    model=model_provider(cfg,ctx)
    optimizer=optimizer_provider(cfg,model)
    scheduler=scheduler_provider(cfg,optimizer)
    data_iterator=data_iterator_provider(cfg,ctx)
    num_microbatches = cfg.gradient_accumulation_steps(ctx.world_size)

    return TrainingSetup(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        world_size=ctx.world_size,
    )

def forward_backward_no_pipeline(
    *,
    forward_step_func: ForwardStepFunc,
    data_iterator: Iterator[object],
    model: object,
    num_microbatches: int,
    forward_only: bool = False,
) -> list[Mapping[str, float]]:
    """Run a no-pipeline forward-backward schedule over microbatches.

    Reading route:
    `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`.
    """
    metrics=[]
    for i in range(num_microbatches):
        loss,metric=forward_step_func(next(data_iterator),model)
        metrics.append(metric)
        if not forward_only:
            scaled_loss = loss / num_microbatches
            scaled_loss.backward()

    return metrics


def train_step(
    forward_step_func: ForwardStepFunc,
    setup: TrainingSetup,
    cfg: ConfigContainer,
    *,
    forward_backward_func: ForwardBackwardFunc = forward_backward_no_pipeline,
    grad_clip_func: Callable[[object, float], float | None] | None = None,
) -> TrainStepResult:
    """Run one Megatron-style optimizer step."""
    optimizer = setup.optimizer
    model = setup.model
    scheduler = setup.scheduler

    optimizer.zero_grad(set_to_none=True)

    metrics_by_microbatch = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=setup.data_iterator,
        model=model,
        num_microbatches=setup.num_microbatches,
        forward_only=False,
    )

    grad_norm = None
    if grad_clip_func is not None:
        grad_norm = grad_clip_func(model, cfg.train.grad_clip)

    optimizer.step()

    consumed_samples = (
            setup.num_microbatches
            * cfg.train.micro_batch_size
            * setup.world_size
    )

    scheduler.step(increment=consumed_samples)

    return TrainStepResult(
        loss_dict=average_metrics(metrics_by_microbatch),
        consumed_samples=consumed_samples,
        grad_norm=grad_norm,
    )



def average_metrics(metrics_by_microbatch: list[Mapping[str, float]]) -> dict[str, float]:
    """Average numeric metrics returned by each microbatch."""
    if not metrics_by_microbatch:
        return {}

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for metrics in metrics_by_microbatch:
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    return {
        name: totals[name] / counts[name]
        for name in totals
    }
