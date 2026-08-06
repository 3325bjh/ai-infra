"""PyTorch 分布式通信基础示例。

运行（CPU / Gloo，最适合先学习通信语义）：
    python WritingDistributedApplications.py --example blocking
    python WritingDistributedApplications.py --example nonblocking
    python WritingDistributedApplications.py --example all-reduce

运行（CUDA / NCCL，需两张 GPU）：
    python WritingDistributedApplications.py --example all-reduce --backend nccl

教程要点：Gloo 适合 CPU 开发；NCCL 是 CUDA 集合通信的首选后端。
MPI 也可用，但 PyTorch 通常需要从源码、配合 MPI 实现编译才支持。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def device_for(backend: str, rank: int) -> torch.device:
    """为当前进程选择设备；NCCL 只能处理 CUDA 张量。"""
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL 后端需要 CUDA，但当前 CUDA 不可用。")
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def blocking_point_to_point(rank: int, world_size: int, device: torch.device) -> None:
    """演示阻塞式 send / recv：调用返回时通信一定已完成。"""
    tensor = torch.zeros(1, device=device)
    if rank == 0:
        tensor += 1
        # rank 0 会在此阻塞，直到 rank 1 的 recv 配对并完成传输。
        dist.send(tensor=tensor, dst=1)
    else:
        # 接收端必须预先分配好正确形状、dtype 和 device 的缓冲区。
        dist.recv(tensor=tensor, src=0)
    print(f"[blocking] rank={rank}, tensor={tensor.item()}", flush=True)


def nonblocking_point_to_point(rank: int, world_size: int, device: torch.device) -> None:
    """演示非阻塞 isend / irecv：必须在访问张量前等待 Work 完成。"""
    tensor = torch.zeros(1, device=device)
    if rank == 0:
        tensor += 1
        request = dist.isend(tensor=tensor, dst=1)
        print("[nonblocking] rank=0 已发起 isend", flush=True)
    else:
        request = dist.irecv(tensor=tensor, src=0)
        print("[nonblocking] rank=1 已发起 irecv", flush=True)

    # wait() 前：发送缓冲区不能修改，接收缓冲区不能读取；否则行为未定义。
    request.wait()
    print(f"[nonblocking] rank={rank}, tensor={tensor.item()}", flush=True)


def all_reduce_example(rank: int, world_size: int, device: torch.device) -> None:
    """演示集合通信：所有 rank 的 1 相加后，每个 rank 都得到 world_size。"""
    group = dist.new_group(ranks=list(range(world_size)))
    tensor = torch.ones(1, device=device)
    # all_reduce 将规约结果写回每个 rank 的 tensor；SUM、MAX、MIN 等均可作为规约操作。
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"[all_reduce] rank={rank}, sum={tensor.item()}", flush=True)


EXAMPLES = {
    "blocking": blocking_point_to_point,
    "nonblocking": nonblocking_point_to_point,
    "all-reduce": all_reduce_example,
}


def init_process(rank: int, world_size: int, example_name: str, backend: str, master_port: int) -> None:
    """设置 rendezvous 信息、初始化默认进程组、运行示例并清理资源。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    # init_process_group 使各进程相互发现，并创建默认通信组（world）。
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        device = device_for(backend, rank)
        EXAMPLES[example_name](rank, world_size, device)
        dist.barrier()  # 确保所有进程完成示例后再销毁进程组。
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch 分布式通信示例")
    parser.add_argument("--example", choices=EXAMPLES, default="blocking")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    if args.world_size != 2 and args.example in {"blocking", "nonblocking"}:
        raise ValueError("点对点示例固定演示 rank 0 向 rank 1 通信，因此 world_size 必须为 2。")
    if args.backend == "nccl" and args.world_size > torch.cuda.device_count():
        raise ValueError("NCCL 的 world_size 不能超过当前机器可用 GPU 数量。")

    # mp.spawn 仅创建进程；每个子进程仍须在 init_process 中初始化进程组。
    mp.spawn(
        init_process,
        args=(args.world_size, args.example, args.backend, args.master_port),
        nprocs=args.world_size,
        join=True,
    )
