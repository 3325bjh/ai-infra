"""Benchmark DDP variants and optimizer-state sharding on one node.

The defaults are the assignment's xl configuration on two GPUs. ``batch_size``
is per rank, so the effective global batch size is ``batch_size * world_size``.

Examples:
    uv run python -m cs336_systems.ddp_benchmark --ddp-implementation naive --optimizer-implementation adamw
    uv run python -m cs336_systems.ddp_benchmark --ddp-implementation naive --optimizer-implementation sharded
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDPBucket, DDP_overlap_individaul_parameters, DDP_with_flat_gradients, Naive_DDP
from cs336_systems.fsdp import FSDP
from cs336_systems.optimizer_state_sharding import Optimizer_state_sharding


DDP_IMPLEMENTATIONS = {
    "naive": Naive_DDP,
    "flat": DDP_with_flat_gradients,
    "overlap": DDP_overlap_individaul_parameters,
    "bucketed": DDPBucket,
    "fsdp": FSDP,
}

COMPUTE_DTYPES = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Table 1 in the assignment handout.  Vocabulary size, context length, and
# per-rank batch size remain independent command-line controls.
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def setup(rank: int, world_size: int, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{rank}")


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def model_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(tensor_bytes(parameter) for parameter in model.parameters())


def model_gradient_bytes(model: torch.nn.Module) -> int:
    return sum(tensor_bytes(parameter.grad) for parameter in model.parameters())


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count tensor state actually owned by this rank's underlying optimizer."""
    if isinstance(optimizer, Optimizer_state_sharding):
        optimizer = optimizer.local_optimizer
    return sum(
        tensor_bytes(value)
        for parameter_state in optimizer.state.values()
        for value in parameter_state.values()
        if isinstance(value, torch.Tensor)
    )


def make_optimizer(params, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if args.optimizer_implementation == "sharded":
        return Optimizer_state_sharding(params, AdamW, **optimizer_kwargs)
    return AdamW(params, **optimizer_kwargs)


def make_parallel_model(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    """Construct the selected DDP/FSDP wrapper around ``model``."""
    if args.ddp_implementation == "bucketed":
        return DDPBucket(model, bucket_size_mb=args.bucket_size_mb)
    if args.ddp_implementation == "fsdp":
        return FSDP(model, compute_dtype=COMPUTE_DTYPES[args.compute_dtype])
    return DDP_IMPLEMENTATIONS[args.ddp_implementation](model)


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    time_step: bool,
) -> tuple[float, float]:
    """Run an update and return (total_ms, gradient_sync_wait_ms)."""
    if time_step:
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        sync_start = torch.cuda.Event(enable_timing=True)
        sync_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()

    if time_step:
        sync_start.record()
    model.finish_gradient_synchronization()
    if time_step:
        sync_end.record()
    optimizer.step()

    if not time_step:
        return 0.0, 0.0
    total_end.record()
    torch.cuda.synchronize()
    return total_start.elapsed_time(total_end), sync_start.elapsed_time(sync_end)


def profile_memory(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    init_peak_bytes: int,
) -> tuple[int, int, int, int, int, int]:
    """Run one update and record the three memory checkpoints in the handout."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()
    model.finish_gradient_synchronization()
    torch.cuda.synchronize()

    before_step_peak = torch.cuda.max_memory_allocated()
    parameter_bytes = model_parameter_bytes(model)
    gradient_bytes = model_gradient_bytes(model)

    optimizer.step()  # Lazily materializes AdamW's state tensors.
    torch.cuda.synchronize()
    after_step_peak = torch.cuda.max_memory_allocated()
    state_bytes = optimizer_state_bytes(optimizer)
    return init_peak_bytes, before_step_peak, after_step_peak, parameter_bytes, gradient_bytes, state_bytes


def reduce_max(values: tuple[int, ...], device: torch.device) -> list[float]:
    metrics = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return metrics.tolist()


def benchmark_worker(rank: int, args: argparse.Namespace) -> None:
    device = setup(rank, args.world_size, args.master_port)
    try:
        torch.manual_seed(args.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=10_000.0,
        ).to(device)
        ddp_model = make_parallel_model(model, args)
        optimizer = make_optimizer(ddp_model.parameters(), args)
        torch.cuda.synchronize()
        init_peak_bytes = torch.cuda.max_memory_allocated()

        # Keep input generation outside timed sections.
        input_ids = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        targets = torch.randint_like(input_ids, high=args.vocab_size)

        memory_metrics = profile_memory(ddp_model, optimizer, input_ids, targets, init_peak_bytes)
        max_memory_metrics = reduce_max(memory_metrics, device)

        for _ in range(args.warmup_steps):
            run_training_step(ddp_model, optimizer, input_ids, targets, time_step=False)
        torch.cuda.synchronize()
        dist.barrier()

        total_times: list[float] = []
        sync_times: list[float] = []
        for _ in range(args.measurement_steps):
            total_ms, sync_ms = run_training_step(ddp_model, optimizer, input_ids, targets, time_step=True)
            total_times.append(total_ms)
            sync_times.append(sync_ms)

        local_times = torch.tensor(
            [sum(total_times) / len(total_times), sum(sync_times) / len(sync_times)], device=device
        )
        mean_times = local_times.clone()
        wall_times = local_times.clone()
        dist.all_reduce(mean_times, op=dist.ReduceOp.SUM)
        dist.all_reduce(wall_times, op=dist.ReduceOp.MAX)
        mean_times.div_(args.world_size)

        if rank == 0:
            mib = 1024**2
            init_peak, before_step, after_step, parameter_bytes, gradient_bytes, state_bytes = [value / mib for value in max_memory_metrics]
            wall_clock_ms, sync_ms = wall_times.tolist()
            print(f"DDP 基准测试：ddp={args.ddp_implementation}，optimizer={args.optimizer_implementation}")
            print(f"  配置：{args.world_size} 张 GPU，{args.model_size} 模型，每个 rank 的 batch={args.batch_size}，context={args.context_length}")
            print(f"  模型：d_model={args.d_model}，d_ff={args.d_ff}，层数={args.num_layers}，heads={args.num_heads}")
            print("  每个 rank 的峰值已分配显存（取所有 rank 中的最大值）：")
            print(f"    模型/optimizer 初始化后：{init_peak:.1f} MiB")
            print(f"    optimizer.step() 前：{before_step:.1f} MiB")
            print(f"    optimizer.step() 后：{after_step:.1f} MiB")
            print("  profiling step 后的 tensor 显存拆分（取所有 rank 中的最大值）：")
            print(f"    参数：{parameter_bytes:.1f} MiB；梯度：{gradient_bytes:.1f} MiB；optimizer state：{state_bytes:.1f} MiB")
            print(f"  平均 rank 本地 iteration：{mean_times[0].item():.3f} ms")
            print(f"  平均 rank 本地梯度同步等待：{mean_times[1].item():.3f} ms")
            print(f"  分布式 iteration（最慢 rank）：{wall_clock_ms:.3f} ms")
            print(f"  分布式梯度同步等待（最慢 rank）：{sync_ms:.3f} ms（占 step 的 {100 * sync_ms / wall_clock_ms:.1f}%）")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP and optimizer-state sharding on the CS336 xl model.")
    parser.add_argument(
        "--ddp-implementation",
        choices=sorted(DDP_IMPLEMENTATIONS),
        default="naive",
        help="并行策略：DDP 变体、bucketed DDP，或 FSDP。",
    )
    parser.add_argument("--optimizer-implementation", choices=["adamw", "sharded"], default="adamw")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl", help="Assignment Table 1 model preset.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size.")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model from --model-size.")
    parser.add_argument("--d-ff", type=int, default=None, help="Override d_ff from --model-size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override num_layers from --model-size.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override num_heads from --model-size.")
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--bucket-size-mb", type=float, default=25.0, help="bucketed DDP 的目标梯度桶大小。")
    parser.add_argument("--compute-dtype", choices=sorted(COMPUTE_DTYPES), default="fp32", help="FSDP 通信和计算精度。")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, value in MODEL_CONFIGS[args.model_size].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.ddp_implementation == "fsdp" and args.optimizer_implementation != "adamw":
        parser.error("FSDP 当前只能与 --optimizer-implementation adamw 组合使用。")
    return args


if __name__ == "__main__":
    benchmark_args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and the NCCL backend.")
    if torch.cuda.device_count() < benchmark_args.world_size:
        raise RuntimeError(f"Requested {benchmark_args.world_size} GPUs, but only {torch.cuda.device_count()} are visible.")
    mp.spawn(benchmark_worker, args=(benchmark_args,), nprocs=benchmark_args.world_size, join=True)
