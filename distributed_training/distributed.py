"""PyTorch 分布式 API 与通信调试示例。
本机启动两个进程：
    torchrun --standalone --nproc_per_node=2 distributed.py --debug
"""

import argparse
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import TCPStore
from torch.distributed.algorithms.model_averaging.utils import (
    average_parameters,
    average_parameters_or_parameter_groups,
    get_params_to_average,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import _get_default_store
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.rendezvous import register_rendezvous_handler


LOGGER: logging.Logger | None = None


def logger() -> logging.Logger:
    """在配置 LOGLEVEL 后返回 Elastic 日志记录器。"""
    global LOGGER
    if LOGGER is None:
        LOGGER = get_logger(__name__)
    return LOGGER


class ToyModule(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def configure_debug(enabled: bool) -> None:
    """在调用 ``init_process_group`` 前配置常用调试信息。"""
    if not enabled:
        return

    # DETAIL 还会检查各个 rank 是否以相同顺序调用集合通信操作。
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,COLL")
    os.environ.setdefault("LOGLEVEL", "INFO")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger()


def log_network_configuration() -> None:
    """打印相关配置，不猜测当前机器的网卡名称。"""
    for name in (
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "TORCH_DISTRIBUTED_DEBUG",
    ):
        logger().info("%s=%s", name, os.environ.get(name, "<not set>"))


def log_process_group_information(rank: int) -> None:
    """打印默认进程组的后端、调试与本机 rank 信息。"""
    logger().info("get_backend()=%s", dist.get_backend())
    logger().info("get_backend_config()=%s", dist.get_backend_config())
    logger().info("get_debug_level()=%s", dist.get_debug_level())
    logger().info(
        "get_node_local_rank(fallback_rank=%s)=%s",
        rank,
        dist.get_node_local_rank(fallback_rank=rank),
    )
    logger().info("get_pg_count()=%s", dist.get_pg_count())


def myenv_handler(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    """``myenv://`` URL 的自定义 rendezvous handler 示例。

    它从 ``torchrun`` 提供的环境变量读取连接信息。此定义仅作演示；
    常规 torchrun 任务仍应使用内置的 ``env://`` rendezvous。
    """
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    store = TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
    )
    yield store, rank, world_size


def average_model_parameters(model: nn.Module) -> None:
    """在一次训练步骤后演示模型参数平均。"""
    params = get_params_to_average(model.parameters())
    average_parameters(params, process_group=None)

    # 若输入改为 optimizer.param_groups，可使用下方等价 API。
    # 同一轮平均不要同时调用两种形式：
    # average_parameters_or_parameter_groups(optimizer.param_groups, None)


def batch_point_to_point_example(rank: int, world_size: int, device: torch.device) -> torch.Tensor | None:
    """使用 batch_isend_irecv 向相邻 rank 发送并接收一个张量。"""
    if world_size == 1:
        logger().info("world_size=1，跳过点对点通信示例")
        return None

    send_tensor = torch.arange(2, dtype=torch.float32, device=device) + 2 * rank
    recv_tensor = torch.empty(2, dtype=torch.float32, device=device)
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1 + world_size) % world_size

    send_op = dist.P2POp(dist.isend, send_tensor, next_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, previous_rank)
    requests = dist.batch_isend_irecv([send_op, recv_op])

    # 在使用接收张量前，必须等待所有异步通信请求完成。
    for request in requests:
        request.wait()

    logger().info(
        "P2P 完成：rank=%s 向 rank=%s 发送 %s；从 rank=%s 接收 %s",
        rank,
        next_rank,
        send_tensor.cpu().tolist(),
        previous_rank,
        recv_tensor.cpu().tolist(),
    )
    return recv_tensor


def synchronous_point_to_point_example(rank: int, world_size: int, device: torch.device) -> None:
    """演示 send() 与 recv() 的阻塞式张量通信。"""
    if world_size < 2:
        return

    if rank == 0:
        send_tensor = torch.tensor([10.0, 20.0], device=device)
        dist.send(send_tensor, dst=1)
        logger().info("同步 send：rank=0 向 rank=1 发送 %s", send_tensor.cpu().tolist())
    elif rank == 1:
        recv_tensor = torch.empty(2, device=device)
        sender_rank = dist.recv(recv_tensor, src=0)
        logger().info("同步 recv：rank=1 从 rank=%s 接收 %s", sender_rank, recv_tensor.cpu().tolist())

    # 其余 rank 也参与屏障，保证下一段示例从同一阶段开始。
    dist.barrier()


def asynchronous_point_to_point_example(rank: int, world_size: int, device: torch.device) -> None:
    """演示 isend()、irecv() 和 Work.wait() 的异步张量通信。"""
    if world_size < 2:
        return

    if rank == 0:
        send_tensor = torch.tensor([30.0, 40.0], device=device)
        work = dist.isend(send_tensor, dst=1)
        # 在 work.wait() 前不得修改 send_tensor。
        work.wait()
        logger().info("异步 isend 已完成：rank=0 发送 %s", send_tensor.cpu().tolist())
    elif rank == 1:
        recv_tensor = torch.empty(2, device=device)
        work = dist.irecv(recv_tensor, src=0)
        work.wait()
        logger().info("异步 irecv 已完成：rank=1 接收 %s", recv_tensor.cpu().tolist())

    dist.barrier()


def object_point_to_point_example(rank: int, world_size: int, device: torch.device) -> None:
    """演示 send_object_list() 与 recv_object_list() 的可信对象通信。"""
    if world_size < 2:
        return

    if rank == 0:
        objects = ["训练状态", 1, {"loss": 0.25}]
        # 对象通信使用 pickle；只能向可信 rank 发送或接收数据。
        dist.send_object_list(objects, dst=1, device=device)
        logger().info("对象发送完成：%s", objects)
    elif rank == 1:
        objects = [None, None, None]  # 列表长度必须与发送方保持一致。
        sender_rank = dist.recv_object_list(objects, src=0, device=device)
        logger().info("对象接收完成：发送方 rank=%s，内容=%s", sender_rank, objects)

    dist.barrier()


def work_collective_example(rank: int, world_size: int, device: torch.device) -> None:
    """演示异步集合通信返回的 Work 对象及其 CUDA/Future 用法。"""
    output = torch.tensor([float(rank + 1)], device=device)
    work = dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=True)
    logger().info("all_reduce 返回 Work；初始 is_completed()=%s", work.is_completed())

    if device.type == "cuda":
        # Work 不阻塞 CPU；它在新 CUDA 流上建立对通信结果的依赖。
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            work.block_current_stream()
            output.add_(100)
        stream.synchronize()  # 仅为打印确定结果；实际训练中可继续安排其他工作。
        logger().info("block_current_stream 后的 CUDA 结果=%s", output.cpu().tolist())
    else:
        # CPU 后端通常通过 wait() 等待通信完成后再使用结果。
        work.wait()
        logger().info("Work.wait() 后的 CPU 结果=%s", output.tolist())

    if dist.get_backend() == "nccl":
        future_tensor = torch.tensor([float(rank + 1)], device=device)
        future = dist.all_reduce(future_tensor, async_op=True).get_future()

        def divide_by_world_size(completed_future):
            return completed_future.value()[0].div_(world_size)

        averaged = future.then(divide_by_world_size).wait()
        logger().info("get_future().then() 得到平均值=%s", averaged.cpu().tolist())


def store_api_example(store, rank: int, world_size: int) -> None:
    """在 TCPStore 的独立前缀下演示常用键值存储 API。"""
    demo_store = dist.PrefixStore("example/", store)

    if rank == 0:
        demo_store.set("status", "ready")
        demo_store.multi_set(["epoch", "phase"], ["1", "train"])
        demo_store.append("word", "Py")
        demo_store.append("word", "Torch")
        demo_store.compare_set("leader", "", "rank0")
        demo_store.set("temporary", "delete me")

    # wait() 等待 rank 0 写入控制信息；之后 get() 不会因 key 缺失而阻塞。
    demo_store.wait(["status", "epoch", "phase", "word", "leader"])
    worker_count = demo_store.add("worker_count", 1)
    demo_store.barrier("all_workers_added", world_size)

    if rank == 0:
        epoch, phase = [value.decode() for value in demo_store.multi_get(["epoch", "phase"])]
        cloned_store = demo_store.clone()
        logger().info(
            "Store：status=%s，epoch=%s，phase=%s，word=%s，leader=%s，worker_count=%s，check=%s",
            demo_store.get("status").decode(),
            epoch,
            phase,
            demo_store.get("word").decode(),
            cloned_store.get("leader").decode(),
            int(demo_store.get("worker_count")),
            demo_store.check(["status", "missing_key"]),
        )

    # TCPStore 支持扩展队列 API；由 rank 0 入队、rank 1（或单进程时 rank 0）出队。
    queue_receiver = 1 if world_size > 1 else 0
    if rank == 0 and demo_store.has_extended_api():
        demo_store.queue_push("tasks", "checkpoint")
    demo_store.barrier("task_enqueued", world_size)
    if rank == queue_receiver and demo_store.has_extended_api():
        logger().info(
            "Store 队列：弹出=%s，剩余长度=%s",
            demo_store.queue_pop("tasks").decode(),
            demo_store.queue_len("tasks"),
        )
    demo_store.barrier("task_dequeued", world_size)

    if rank == 0:
        deleted = demo_store.delete_key("temporary")
        logger().info("Store：num_keys=%s，delete_key('temporary')=%s", demo_store.num_keys(), deleted)
    demo_store.barrier("store_demo_finished", world_size)


def create_and_inspect_subgroup(
    rank: int,
    world_size: int,
    device_type: str,
) -> tuple[object | None, DeviceMesh | None]:
    """创建偶数 rank 子组，并将其转换为一维 DeviceMesh。"""
    # 所有 rank 都必须以相同顺序调用 new_group。
    subgroup_ranks = list(range(0, world_size, 2))
    subgroup = dist.new_group(ranks=subgroup_ranks)

    # 非成员不能使用这个子组进行通信或查询组内 rank。
    if rank not in subgroup_ranks:
        logger().info("rank=%s 不是子组 %s 的成员", rank, subgroup_ranks)
        return None, None

    group_ranks = dist.get_process_group_ranks(subgroup)
    group_rank = dist.get_group_rank(subgroup, global_rank=rank)
    recovered_global_rank = dist.get_global_rank(subgroup, group_rank=group_rank)
    logger().info(
        "子组成员=%s；全局 rank=%s 对应组内 rank=%s；反向映射=%s",
        group_ranks,
        rank,
        group_rank,
        recovered_global_rank,
    )

    # 将已有 ProcessGroup 包装成一维 DeviceMesh，而不是重新创建通信器。
    device_mesh = DeviceMesh.from_group(
        group=subgroup,
        device_type=device_type,
        mesh_dim_names=("even_rank",),
    )
    logger().info(
        "DeviceMesh：device_type=%s，mesh=%s，mesh_dim_names=%s，"
        "全局 rank=%s，坐标=%s，组内 rank=%s，维度组数量=%s",
        device_mesh.device_type,
        device_mesh.mesh.tolist(),
        device_mesh.mesh_dim_names,
        device_mesh.get_rank(),
        device_mesh.get_coordinate(),
        device_mesh.get_local_rank("even_rank"),
        len(device_mesh.get_all_groups()),
    )
    # 以维度名称获取的组，就是该一维 DeviceMesh 的唯一通信组。
    mesh_group = device_mesh.get_group("even_rank")
    logger().info("DeviceMesh.get_group('even_rank') 的成员=%s", dist.get_process_group_ranks(mesh_group))
    return subgroup, device_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="启用 PyTorch/NCCL 调试日志")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default=None)
    args = parser.parse_args()

    configure_debug(args.debug)
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")

    # 注册不会自动启用该 handler；下方仍使用默认的 ``env://``。
    register_rendezvous_handler("myenv", myenv_handler)
    # 有限超时可将网络或网卡不匹配转为明确报错，避免训练无限卡住。
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=60),
    )

    #检查默认进程组是否已初始化
    print(torch.distributed.is_initialized())

    #检查此进程是否使用torchelastic启动
    if dist.is_torchelastic_launched():
        print("使用可恢复、可重试的训练逻辑")
    else:
        print("使用普通训练逻辑")

    subgroup = None
    try:
        rank = dist.get_rank()
        logger().info("process group ready: rank=%s, world_size=%s, backend=%s", rank, dist.get_world_size(), backend)
        log_network_configuration()
        log_process_group_information(rank)
        # 默认 Store 是 init_process_group 创建的 rendezvous Store。
        # _get_default_store 是内部 API，此处仅用于学习 Store 的用法。
        store_api_example(_get_default_store(), rank, dist.get_world_size())

        if backend == "nccl":
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        subgroup, device_mesh = create_and_inspect_subgroup(
            rank,
            dist.get_world_size(),
            device.type,
        )
        synchronous_point_to_point_example(rank, dist.get_world_size(), device)
        asynchronous_point_to_point_example(rank, dist.get_world_size(), device)
        object_point_to_point_example(rank, dist.get_world_size(), device)
        work_collective_example(rank, dist.get_world_size(), device)
        batch_point_to_point_example(rank, dist.get_world_size(), device)
        model = ToyModule(5, 10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss = model(torch.randn(2, 5, device=device)).sum()
        loss.backward()
        optimizer.step()

        # 所有 rank 必须以相同顺序调用。
        average_model_parameters(model)
        logger().info("parameter averaging completed")
    finally:
        # 先销毁额外子组，再销毁默认组；组成员的销毁顺序必须一致。
        if subgroup is not None:
            dist.destroy_process_group(subgroup)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
