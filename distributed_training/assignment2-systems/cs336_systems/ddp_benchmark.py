"""Benchmark the section 5 naïve DDP implementation on one node.

Example:
    uv run python -m cs336_systems.ddp_benchmark

The defaults are the assignment's xl configuration on two GPUs.  ``batch_size``
is per rank, so the effective global batch size is
``batch_size * world_size``.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP_overlap_individaul_parameters, DDP_with_flat_gradients,Naive_DDP


DDP_IMPLEMENTATIONS = {
    "naive": Naive_DDP,
    "flat": DDP_with_flat_gradients,
    "overlap": DDP_overlap_individaul_parameters,
}


def setup(rank: int, world_size: int, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{rank}")


def run_training_step(
    model: torch.nn.Module,
    optimizer: AdamW,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    time_step: bool,
) -> tuple[float, float]:
    """Run one complete update and return (total_ms, gradient_comm_ms)."""
    if time_step:
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        communication_start = torch.cuda.Event(enable_timing=True)
        communication_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()

    if time_step:
        communication_start.record()
    # For ``overlap``, this waits for the remaining asynchronous communication.
    # For the synchronous variants, it is the full gradient communication time.
    model.finish_gradient_synchronization()
    if time_step:
        communication_end.record()

    optimizer.step()

    if not time_step:
        return 0.0, 0.0

    total_end.record()
    torch.cuda.synchronize()
    return total_start.elapsed_time(total_end), communication_start.elapsed_time(communication_end)


def benchmark_worker(rank: int, args: argparse.Namespace) -> None:
    device = setup(rank, args.world_size, args.master_port)
    try:
        # Different rank-local batches are required for data parallel training.
        torch.manual_seed(args.seed + rank)
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=10_000.0,
        ).to(device)
        ddp_model = DDP_IMPLEMENTATIONS[args.ddp_implementation](model)
        optimizer = AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # Generate data before timing so that host-side sampling and transfer do
        # not contaminate the training-step measurement.
        input_ids = torch.randint(
            args.vocab_size,
            (args.batch_size, args.context_length),
            device=device,
        )
        targets = torch.randint_like(input_ids, high=args.vocab_size)

        for _ in range(args.warmup_steps):
            run_training_step(ddp_model, optimizer, input_ids, targets, time_step=False)
        torch.cuda.synchronize()
        dist.barrier()

        total_times: list[float] = []
        communication_times: list[float] = []
        for _ in range(args.measurement_steps):
            total_ms, communication_ms = run_training_step(ddp_model, optimizer, input_ids, targets, time_step=True)
            total_times.append(total_ms)
            communication_times.append(communication_ms)

        local_metrics = torch.tensor(
            [sum(total_times) / len(total_times), sum(communication_times) / len(communication_times)],
            device=device,
        )
        mean_metrics = local_metrics.clone()
        max_metrics = local_metrics.clone()
        dist.all_reduce(mean_metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_metrics, op=dist.ReduceOp.MAX)
        mean_metrics.div_(args.world_size)

        if rank == 0:
            wall_clock_ms, communication_ms = max_metrics.tolist()
            print(f"DDP benchmark ({args.ddp_implementation})")
            print(f"  configuration: {args.world_size} GPUs, xl-compatible model, per-rank batch={args.batch_size}, context={args.context_length}")
            print(f"  measured steps: {args.measurement_steps} (after {args.warmup_steps} warm-up steps)")
            print(f"  mean rank-local training step: {mean_metrics[0].item():.3f} ms")
            print(f"  mean rank-local finish_gradient_synchronization: {mean_metrics[1].item():.3f} ms")
            print(f"  distributed wall-clock training step (slowest rank): {wall_clock_ms:.3f} ms")
            print(f"  distributed gradient synchronization (slowest rank): {communication_ms:.3f} ms ({100 * communication_ms / wall_clock_ms:.1f}% of step)")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP implementations on the CS336 xl language model.")
    parser.add_argument(
        "--ddp-implementation",
        choices=sorted(DDP_IMPLEMENTATIONS),
        default="naive",
        help="Gradient synchronization strategy to benchmark.",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size.")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--d-ff", type=int, default=10240)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark_args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and the NCCL backend.")
    if torch.cuda.device_count() < benchmark_args.world_size:
        raise RuntimeError(f"Requested {benchmark_args.world_size} GPUs, but only {torch.cuda.device_count()} are visible.")
    mp.spawn(benchmark_worker, args=(benchmark_args,), nprocs=benchmark_args.world_size, join=True)
