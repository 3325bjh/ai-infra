"""分布式同步 SGD 与官方 DistributedDataParallel（DDP）对照示例。

默认使用 CPU/Gloo：
    python ddp.py --implementation manual --epochs 1
    python ddp.py --implementation ddp --epochs 1

若有两张 GPU，使用 NCCL：
    python ddp.py --implementation ddp --backend nccl --world-size 2 --epochs 1

首次运行会下载 MNIST。教程中的手工梯度平均仅用于理解原理；生产训练请使用 DDP。
"""

import argparse
import os
from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, transforms


class Partition(torch.utils.data.Dataset):
    """只暴露原始数据集指定索引的一部分，实现每个 rank 的数据子集。"""

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.data[self.indices[index]]


class DataPartitioner:
    """以固定随机种子将数据随机且互不重叠地划分给各 rank。"""

    def __init__(self, data, sizes, seed=1234):
        indices = list(range(len(data)))
        rng = Random(seed)
        rng.shuffle(indices)

        self.data = data
        self.partitions = []
        offset = 0
        for fraction in sizes:
            length = int(fraction * len(data))
            self.partitions.append(indices[offset : offset + length])
            offset += length

    def use(self, partition_id):
        return Partition(self.data, self.partitions[partition_id])


class Net(nn.Module):
    """教程使用的简单 MNIST 分类网络，输出 log-probabilities。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def current_device(backend, local_rank):
    """NCCL 使用每进程一张 GPU；Gloo 示例保留在 CPU，便于无 GPU 学习。"""
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL 需要 CUDA。请改用 --backend gloo 或安装 CUDA PyTorch。")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def partition_dataset(rank, world_size, batch_size):
    """下载一次 MNIST，再把样本均分给每个 rank，并保持全局 batch size 不变。"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # 避免多个 rank 同时写入同一数据目录。
    if rank == 0:
        datasets.MNIST("./data", train=True, download=True, transform=transform)
    dist.barrier()
    dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)

    partitions = DataPartitioner(dataset, [1.0 / world_size] * world_size)
    local_batch_size = max(1, batch_size // world_size)
    loader = torch.utils.data.DataLoader(
        partitions.use(rank), batch_size=local_batch_size, shuffle=True
    )
    return loader, local_batch_size


def average_gradients(model):
    """手工实现同步 SGD 的核心：对每个参数梯度求和后除以 rank 数。"""
    world_size = float(dist.get_world_size())
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]


def ring_allreduce(send, recv, dim=-1):
    """使用 reduce-scatter + all-gather 实现标准的分块 Ring All-Reduce（求和）。

    所有 rank 的 ``send`` 必须具有相同形状，且 ``send.shape[dim]`` 必须能被
    world_size 整除。函数不修改 send，而是将全局逐元素和写入 recv。

    以 3 个 rank、每个张量 [a, b, c] 为例，rank 0 的 reduce-scatter 首轮发送 a0，
    接收 c2 并累加到 c0；rank 1 发送 b1，rank 2 发送 c2。每次仅传一个分块。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if send.shape != recv.shape:
        raise ValueError("send 与 recv 的形状必须一致。")
    if send.shape[dim] % world_size != 0:
        raise ValueError(
            f"send.shape[{dim}]={send.shape[dim]} 必须能被 world_size={world_size} 整除。"
        )
    if world_size == 1:
        recv.copy_(send)
        return

    left_rank = (rank - 1) % world_size
    right_rank = (rank + 1) % world_size
    # torch.chunk 返回 tuple；转成 list 后才能原地更新某个分块。
    # clone 使函数不会修改调用方传入的 send。
    chunks = list(torch.chunk(send.clone(), world_size, dim=dim))

    # 阶段 1：reduce-scatter。循环结束时，每个 rank 仅拥有一个已经求和的分块。
    # 第 step 轮：rank r 向右发送 (r-step) 号块，从左接收 (r-step-1) 号块并累加。
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        receive_buffer = torch.empty_like(chunks[receive_index])

        request = dist.isend(tensor=chunks[send_index], dst=right_rank)
        dist.recv(tensor=receive_buffer, src=left_rank)
        # 必须等 send 完成后，才可以修改任何可能仍被通信读取的张量。
        request.wait()
        chunks[receive_index].add_(receive_buffer)

    # 阶段 2：all-gather。将 reduce-scatter 得到的完整分块继续沿环传递。
    # 第 step 轮：rank r 发送 (r-step+1) 号块，接收 (r-step) 号块。
    for step in range(world_size - 1):
        send_index = (rank - step + 1) % world_size
        receive_index = (rank - step) % world_size
        receive_buffer = torch.empty_like(chunks[receive_index])

        request = dist.isend(tensor=chunks[send_index], dst=right_rank)
        dist.recv(tensor=receive_buffer, src=left_rank)
        request.wait()
        chunks[receive_index].copy_(receive_buffer)

    # 所有已归约分块按原来的维度拼接，得到每个 rank 都相同的完整结果。
    recv.copy_(torch.cat(chunks, dim=dim))



def train(rank, world_size, backend, implementation, epochs, batch_size):
    """每个 rank 在自己的数据分区上训练，并通过手工 all_reduce 或 DDP 同步梯度。"""
    device = current_device(backend, rank)
    torch.manual_seed(1234)  # 所有 rank 从相同初始参数开始。
    train_loader, local_batch_size = partition_dataset(rank, world_size, batch_size)

    model = Net().to(device)
    if implementation == "ddp":
        # DDP 在 backward 中自动 bucket 化并 all-reduce 梯度，无需 average_gradients。
        model = DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
        )

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    batches_per_epoch = ceil(len(train_loader.dataset) / float(local_batch_size))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)
            total_loss += loss.item()
            loss.backward()
            if implementation == "manual":
                average_gradients(model)
            optimizer.step()

        print(
            f"rank={rank}, implementation={implementation}, epoch={epoch}, "
            f"mean_loss={total_loss / batches_per_epoch:.4f}",
            flush=True,
        )


def worker(rank, world_size, backend, implementation, epochs, batch_size, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        train(rank, world_size, backend, implementation, epochs, batch_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步 SGD 与 DDP 对照示例")
    parser.add_argument("--implementation", choices=("manual", "ddp"), default="ddp")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--master-port", type=int, default=29501)
    args = parser.parse_args()

    if args.backend == "nccl" and args.world_size > torch.cuda.device_count():
        raise ValueError("NCCL 的 world_size 不能超过可用 GPU 数量。")
    mp.spawn(
        worker,
        args=(
            args.world_size,
            args.backend,
            args.implementation,
            args.epochs,
            args.batch_size,
            args.master_port,
        ),
        nprocs=args.world_size,
        join=True,
    )
