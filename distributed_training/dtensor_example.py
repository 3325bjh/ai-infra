"""DTensor / DeviceMesh 的两种本机启动方式。

推荐（torchrun，自动提供 RANK、LOCAL_RANK、WORLD_SIZE 等环境变量）：
    torchrun --standalone --nproc_per_node=4 dtensor_example.py --launcher torchrun

也可直接用 Python 运行（脚本内部通过 mp.spawn 创建子进程）：
    python dtensor_example.py --launcher spawn --world-size 4

前提：有至少对应数量的 CUDA GPU，并且 PyTorch 安装支持 NCCL。
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
    empty as dtensor_empty,
    full as dtensor_full,
    init_device_mesh,
    ones as dtensor_ones,
    rand as dtensor_rand,
    randn as dtensor_randn,
    zeros as dtensor_zeros,
)


class MyModule(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.fc1=nn.Linear(8,8)
        self.fc2=nn.Linear(8,8)
        self.relu=nn.ReLU()

    def forward(self,input):
        return self.relu(self.fc1(input)+self.fc2(input))

def run_dtensor(local_rank: int, rank: int, world_size: int) -> None:
    """在已初始化的默认进程组上创建一维 CUDA DeviceMesh。"""
    # 每个进程独占一张 GPU；NCCL 通信和后续 CUDA 张量都会使用该设备。
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # (world_size,) 表示一维 mesh：rank 0, 1, ... 分别对应一张 GPU。
    mesh = init_device_mesh("cuda", (world_size,))

    # distribute_tensor 默认以 rank 0 的数据作为全局张量，再按 Shard(0) 切分。
    # 因此只有 rank 0 需要构造完整张量；其他 rank 的占位张量不会成为源数据。
    if rank == 0:
        full_tensor = torch.randn(16, 8, device=device)
    else:
        full_tensor = torch.empty(0, device=device)

    my_dtensor = distribute_tensor(full_tensor, mesh, [Shard(dim=0)])
    print(
        f"rank={rank}, local_rank={local_rank}, "
        f"global_shape={tuple(my_dtensor.shape)}, "
        f"local_shape={tuple(my_dtensor.to_local().shape)}, "
        f"placement={my_dtensor.placements}",
        flush=True,
    )

def print_local(label: str, tensor: DTensor, rank: int) -> None:
    """打印 DTensor 的全局信息，以及当前 rank 实际保存的本地张量。"""
    local = tensor.to_local()
    print(
        f"[{label}] rank={rank}: 全局形状={tuple(tensor.shape)}, "
        f"placements={tensor.placements}, 本地形状={tuple(local.shape)}, "
        f"本地张量值=\\n{local.cpu()}",
        flush=True,
    )

def documented_api_examples(local_rank: int, rank: int, world_size: int) -> tuple:
    """演示文档中的稳定核心 API：DeviceMesh、DTensor 和分布式工厂函数。"""
    if world_size != 4:
        raise ValueError("本 API 示例使用 4 个 rank，因此 world_size 必须为 4。")

    # 每个进程绑定自己的 GPU；NCCL 及未显式指定 device 的 CUDA 操作会使用它。
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh_1d = init_device_mesh("cuda", (4,), mesh_dim_names=("tp",))

    # DeviceMesh 是分布式通信器：get_group 返回当前 rank 在该 mesh 维度所属的通信组。
    # 一维 mesh 中 local rank 与全局 rank 相同；二维 mesh 中二者可能不同。
    print(
        f"[DeviceMesh 1D] rank={rank}, coordinate={mesh_1d.get_coordinate()}, "
        f"size={mesh_1d.size()}, local_rank={mesh_1d.get_local_rank()}, "
        f"group_size={dist.get_world_size(mesh_1d.get_group('tp'))}",
        flush=True,
    )

    # 命名二维 mesh 后，可以按名称切出子 mesh、查询局部坐标及底层通信组。
    mesh_2d = init_device_mesh(
        "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
    )
    replicate_mesh = mesh_2d["replicate"]
    shard_mesh = mesh_2d["shard"]
    print(
        f"[DeviceMesh 2D] rank={rank}, coordinate={mesh_2d.get_coordinate()}, "
        f"replicate_local_rank={mesh_2d.get_local_rank('replicate')}, "
        f"shard_local_rank={mesh_2d.get_local_rank('shard')}, "
        f"replicate_group_size={dist.get_world_size(replicate_mesh.get_group())}, "
        f"shard_group_size={dist.get_world_size(shard_mesh.get_group())}",
        flush=True,
    )

    # distribute_tensor：将 rank 0 的全局 Tensor 分发为 Shard(0) DTensor。
    source = (
        torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
        if rank == 0
        else torch.empty(0, device=device)
    )
    sharded = distribute_tensor(source, mesh_1d, [Shard(0)])
    print_local("distribute_tensor + Shard(0)", sharded, rank)

    # __create_chunk_list__：返回当前 rank 本地分片的大小与全局偏移量元数据。
    # 它主要服务 Distributed Checkpoint；一个 DTensor 在每个 rank 通常只有一个本地块。
    chunk_metadata = sharded.__create_chunk_list__()
    print(f"[__create_chunk_list__] rank={rank}, metadata={chunk_metadata}", flush=True)

    # DTensor.from_local：各 rank 已有局部存储时直接包装；全局形状由 placement 推导。
    local_rows = torch.full((2, 4), float(rank), device=device, requires_grad=True)
    from_local = DTensor.from_local(local_rows, mesh_1d, [Shard(0)])
    print_local("DTensor.from_local", from_local, rank)

    # to_local 返回当前 rank 的物理 Tensor；full_tensor 会 all_gather，返回完整普通 Tensor。
    local_view = from_local.to_local()
    full_view = from_local.full_tensor()
    print(
        f"[to_local / full_tensor] rank={rank}, local={tuple(local_view.shape)}, "
        f"full={tuple(full_view.shape)}",
        flush=True,
    )

    # redistribute 保留逻辑全局张量，改变物理布局；Shard(0) -> Replicate() 会 all_gather。
    replicated = from_local.redistribute(mesh_1d, [Replicate()])
    print_local("DTensor.redistribute -> Replicate()", replicated, rank)

    # 分布式工厂函数直接创建 DTensor，不需要先在 rank 0 构建完整 Tensor。
    # rand/randn 的 DTensor RNG 要求所有 rank 进入操作前拥有相同随机数状态。
    torch.manual_seed(2026)
    factories = {
        "ones": dtensor_ones(8, 4, device_mesh=mesh_1d, placements=[Shard(0)]),
        "zeros": dtensor_zeros(8, 4, device_mesh=mesh_1d, placements=[Shard(0)]),
        "empty": dtensor_empty(8, 4, device_mesh=mesh_1d, placements=[Shard(0)]),
        "full": dtensor_full(8, 4, fill_value=7.0, device_mesh=mesh_1d, placements=[Shard(0)]),
        "rand": dtensor_rand(8, 4, device_mesh=mesh_1d, placements=[Shard(0)]),
        "randn": dtensor_randn(8, 4, device_mesh=mesh_1d, placements=[Shard(0)]),
    }
    for name, tensor in factories.items():
        print(
            f"[factory:{name}] rank={rank}, global={tuple(tensor.shape)}, "
            f"local={tuple(tensor.to_local().shape)}, placements={tensor.placements}",
            flush=True,
        )

    # Partial 表示“局部结果尚未归约”。例如下式会产生等待 sum 归约的 DTensor 布局。
    partial = DTensor.from_local(
        torch.full((2, 4), float(rank), device=device), mesh_1d, [Partial("sum")]
    )
    reduced = partial.redistribute(mesh_1d, [Replicate()])
    print(
        f"[Partial -> Replicate] rank={rank}, reduced_local_value={reduced.to_local()[0, 0].item()}",
        flush=True,
    )

    # CommDebugMode 统计其上下文内发生的功能性集合通信，适合定位意外重分布。
    try:
        from torch.distributed.tensor.debug import CommDebugMode

        comm_mode = CommDebugMode()
        with comm_mode:
            _ = sharded.full_tensor()  # Shard -> 完整 Tensor，通常需要 all_gather。
        if rank == 0:
            print(f"[CommDebugMode] 通信计数={comm_mode.get_comm_counts()}", flush=True)
    except ImportError:
        # 调试模块或其可选依赖在某些 PyTorch 版本中可能未安装，不影响核心示例。
        if rank == 0:
            print("[CommDebugMode] 当前 PyTorch 安装未提供该可选调试 API。", flush=True)
    return mesh_1d, sharded


def placement_and_debug_api_examples(mesh, sharded: DTensor, rank: int) -> None:
    """补充页面列出的 Placement 私有接口和 CommDebugMode 全部查询/导出接口。

    这些 Placement 名称以下划线开头，属于框架内部实现；示例用于阅读源码和理解检查点，
    不建议将其作为生产代码的稳定依赖。
    """
    from torch.distributed.tensor.debug import CommDebugMode, visualize_sharding
    from torch.distributed.tensor.placement_types import _MaskPartial, _StridedShard

    shard, replicate, partial = Shard(0), Replicate(), Partial("sum")
    print(
        f"[Placement predicates] Shard={shard.is_shard()}, Replicate={replicate.is_replicate()}, "
        f"Partial={partial.is_partial()}, ALL_REDUCE_OPS={Partial.ALL_REDUCE_OPS}, "
        f"LINEAR_REDUCE_OPS={Partial.LINEAR_REDUCE_OPS}",
        flush=True,
    )
    local_size, offset = Shard.local_shard_size_and_offset(
        8, mesh.size(), mesh.get_local_rank()
    )
    strided_shard = _StridedShard(0, split_factor=2)
    strided_size, strided_offset = strided_shard.local_shard_size_and_offset(
        8, mesh.size(), mesh.get_local_rank()
    )
    mask_partial = _MaskPartial("sum", offset_shape=torch.Size([8, 4]), offset_dim=0)
    print(
        f"[Shard metadata] size={local_size}, offset={offset}; "
        f"[_StridedShard] split_factor={strided_shard.split_factor}, "
        f"size={strided_size}, offset={strided_offset}; "
        f"[_MaskPartial] offset_shape={mask_partial.offset_shape}, offset_dim={mask_partial.offset_dim}, "
        f"mask_buffer={mask_partial.mask_buffer}",
        flush=True,
    )

    comm_mode = CommDebugMode()
    with comm_mode:
        _ = sharded.full_tensor()
    # 完整的 CommDebugMode 查询 API。
    print(
        f"[CommDebugMode] counts={comm_mode.get_comm_counts()}, total={comm_mode.get_total_counts()}, "
        f"parameters={comm_mode.get_parameter_info()}, sharding={comm_mode.get_sharding_info()}",
        flush=True,
    )
    if rank == 0:
        # 这三个 API 会打印或写入调试产物；文件保存在当前工作目录。
        comm_mode.generate_comm_debug_tracing_table(noise_level=0)
        comm_mode.generate_json_dump("comm_mode_log.json", noise_level=0)
        comm_mode.log_comm_debug_tracing_table_to_file("comm_mode_log.txt", noise_level=0)
        visualize_sharding(sharded, header="Shard(0) 布局")


def experimental_api_examples(mesh, sharded: DTensor) -> None:
    """页面全部实验 API 的最小示例；需显式调用，避免默认运行时改写算子/缓冲区。"""
    import torch.nn.functional as F
    from torch.distributed.tensor.experimental import (
        context_parallel,
        implicit_replication,
        local_map,
        register_sharding,
    )

    # implicit_replication：将混入运算的普通 Tensor 暂时视为 Replicate DTensor。
    with implicit_replication():
        _ = sharded + torch.ones_like(sharded.to_local())

    # local_map：以普通 Tensor 编写局部函数，再声明 DTensor 的输入和输出布局。
    local_double = local_map(
        lambda local_x: local_x * 2,
        out_placements=[Shard(0)],
        in_placements=([Shard(0)],),
        device_mesh=mesh,
    )
    _ = local_double(sharded)

    # register_sharding：为算子添加/覆盖 DTensor 布局策略（示例注册 softmax）。
    @register_sharding(torch.ops.aten._softmax.default)
    def custom_softmax_sharding(x, dim, half_to_float):
        return [([Replicate()], [Replicate(), None, None])]

    # context_parallel：对 SDPA 打补丁，并按序列维原地分片 buffers。
    # 真实使用需要 query/key/value 形状与 mesh 匹配；此处展示官方调用形式。
    query = key = value = torch.randn(1, 1, 8, 4, device=sharded.device_mesh.device_type)
    with context_parallel(mesh, buffers=[query, key, value], buffer_seq_dims=[2, 2, 2]):
        _ = F.scaled_dot_product_attention(query, key, value)

def dtensor_example(local_rank: int, rank: int, world_size: int) -> None:
    """演示一维、二维 DeviceMesh 的布局；需要使用 4 个进程启动。"""
    if world_size != 4:
        raise ValueError("本示例使用 (4,) 和 (2, 2) mesh，因此 world_size 必须是 4。")

    device = torch.device("cuda", local_rank)

    # 一维 mesh 的逻辑结构为：[rank 0, rank 1, rank 2, rank 3]。
    mesh_1d = init_device_mesh("cuda", (4,))
    rowwise_placement = [Shard(0)]
    colwise_placement = [Shard(1)]
    replicate_placement = [Replicate()]

    # 完整的全局张量只在 rank 0 上创建。对于这个 8 x 4 的例子：
    # - Shard(0)：rank 0/1/2/3 分别得到行 [0:2]/[2:4]/[4:6]/[6:8]。
    # - Shard(1)：rank 0/1/2/3 分别得到列 [0:1]/[1:2]/[2:3]/[3:4]。
    # - Replicate()：每个 rank 都得到完整的 8 x 4 张量。
    big_tensor = (
        torch.arange(32, device=device, dtype=torch.float32).reshape(8, 4)
        if rank == 0
        else torch.empty(0, device=device)
    )

    rowwise_tensor = distribute_tensor(big_tensor, mesh_1d, rowwise_placement)
    print_local("一维 Shard(0)：按行切分", rowwise_tensor, rank)

    colwise_tensor = distribute_tensor(big_tensor, mesh_1d, colwise_placement)
    print_local("一维 Shard(1)：按列切分", colwise_tensor, rank)

    replica_tensor = distribute_tensor(big_tensor, mesh_1d, replicate_placement)
    print_local("一维 Replicate()：每个 rank 保存完整副本", replica_tensor, rank)

    # 同样的四个 rank 可以组成如下二维 mesh：
    # [[rank 0, rank 1],
    #  [rank 2, rank 3]]
    mesh_2d = init_device_mesh("cuda", (2, 2))
    replica_then_row_shard = [Replicate(), Shard(0)]
    # 第一个 placement 作用于 mesh 第 0 维（竖直方向）：rank 0 与 rank 2 的数据相同，
    # rank 1 与 rank 3 的数据相同。第二个 placement 作用于 mesh 第 1 维（水平方向）：
    # rank 0/2 获得全局张量的行 [0:4]，rank 1/3 获得行 [4:8]。
    two_dim_tensor = distribute_tensor(big_tensor, mesh_2d, replica_then_row_shard)
    print_local("二维 [Replicate(), Shard(0)]", two_dim_tensor, rank)

    # from_local 表示每个 rank 从自己已持有的本地张量开始构造 DTensor。
    # 对 [Replicate(), Shard(0)] 而言，rank 0 和 rank 2 保存相同的前四行；
    # rank 1 和 rank 3 保存相同的后四行；四份局部存储共同描述一个 8 x 8 全局张量。
    # rank 0/2 填入 0，rank 1/3 填入 1，确保第一个 Replicate() 所要求的
    # mesh 第 0 维复制关系确实成立。
    local_tensor = torch.full((4, 8), float(rank % 2), device=device, requires_grad=True)
    from_local_rowwise = DTensor.from_local(local_tensor, mesh_2d, replica_then_row_shard)
    print_local("from_local：[Replicate(), Shard(0)]", from_local_rowwise, rank)

    # redistribute 会改变物理存储布局，但逻辑上的全局张量保持不变。
    # 变为 [Replicate(), Shard(1)] 后，rank 0/2 保存列 [0:4]，rank 1/3 保存列 [4:8]。
    replica_then_col_shard = [Replicate(), Shard(1)]
    from_local_colwise = from_local_rowwise.redistribute(mesh_2d, replica_then_col_shard)
    print_local("重新分布 -> [Replicate(), Shard(1)]", from_local_colwise, rank)

    # 最后，每个 rank 都物化出一份完整的 8 x 8 张量副本。
    fully_replicated = from_local_colwise.redistribute(mesh_2d, [Replicate(), Replicate()])
    print_local("重新分布 -> [Replicate(), Replicate()]", fully_replicated, rank)


def high_level_dtensor_example(local_rank: int, rank: int, world_size: int) -> None:
    """演示 distribute_module：将 nn.Linear 的参数批量转换为 DTensor。"""
    if world_size != 4:
        raise ValueError("本示例使用一维 (4,) mesh，因此 world_size 必须是 4。")

    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (4,))

    # 固定随机种子仅为了让每个 rank 的初始模型便于比较。
    # 真正分布参数时，distribute_tensor 默认以 rank 0 上的完整参数作为源数据。
    torch.manual_seed(2026)
    model = MyModule().to(device)

    def shard_params(module_name: str, module: nn.Module, device_mesh) -> None:
        """将每个 Linear 的 weight、bias 沿输出特征维（第 0 维）切分。"""
        if not isinstance(module, nn.Linear):
            return

        # 对 Linear(in_features=8, out_features=8) 而言：
        # weight 的全局形状为 (8, 8)，Shard(0) 后每个 rank 保存 (2, 8)；
        # bias 的全局形状为 (8,)，Shard(0) 后每个 rank 保存 (2,)。
        # 这就是按输出特征切分（column-wise / 输出通道切分）的张量并行布局。
        for parameter_name, parameter in module.named_parameters(recurse=False):
            distributed_parameter = nn.Parameter(
                distribute_tensor(parameter, device_mesh, [Shard(0)]),
                requires_grad=parameter.requires_grad,
            )
            # 用 DTensor 参数替换原来的 torch.Tensor 参数；模块名参数可用于
            # 为不同层选择不同布局，例如只对 "fc1" 的参数做分片。
            module.register_parameter(parameter_name, distributed_parameter)

        print(
            f"rank={rank}: 已分片模块 {module_name}，"
            f"weight 本地形状={tuple(module.weight.to_local().shape)}，"
            f"bias 本地形状={tuple(module.bias.to_local().shape)}",
            flush=True,
        )

    # input_fn 是注册在整个模型上的 forward pre-hook。虽然部分 IDE 的类型提示将它标为
    # -> None，但 PyTorch 官方实现会将其返回值交给 forward_pre_hook，用来替换真实输入。
    # 因此必须返回与 inputs 结构相同的新元组。
    def input_fn(module: nn.Module, inputs: tuple, device_mesh) -> tuple:
        local_input = inputs[0]
        # rank 0 持有完整输入，其他 rank 使用空张量占位。Replicate() 会将 rank 0 的
        # (2, 8) 输入复制到所有 rank，以匹配按输出特征分片的 Linear 参数。
        distributed_input = distribute_tensor(local_input, device_mesh, [Replicate()])
        print(
            f"rank={rank}: input_fn 将普通输入转换为 {distributed_input.placements}，"
            f"本地输入形状={tuple(distributed_input.to_local().shape)}",
            flush=True,
        )
        return (distributed_input,)

    # output_fn 是注册在整个模型上的 forward hook。它的返回值会替换模型最终输出。
    def output_fn(module: nn.Module, output: DTensor, device_mesh) -> torch.Tensor:
        # Linear 的输出特征此前按 rank 分片，例如每个 rank 只持有 (2, 2)。
        print(
            f"rank={rank}: output_fn 收到分片输出，placements={output.placements}，"
            f"本地形状={tuple(output.to_local().shape)}",
            flush=True,
        )
        # full_tensor() 会触发 all_gather，将四个局部输出沿输出特征维拼接，
        # 返回完整的普通 torch.Tensor；该返回值成为 sharded_module(...) 的最终结果。
        return output.full_tensor()

    # distribute_module 会遍历 model 的子模块并执行 partition_fn，因此 fc1、fc2 的
    # weight 和 bias 都会被替换为 DTensor；同时在外层模型注册 input_fn、output_fn。
    sharded_module = distribute_module(
        model,
        mesh,
        partition_fn=shard_params,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    # 调用方只传入普通 Tensor：rank 0 创建完整输入 (2, 8)，其他 rank 使用空张量占位。
    # input_fn 会在真正执行 forward 前自动将其转换为复制布局的 DTensor。
    full_input = (
        torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
        if rank == 0
        else torch.empty(0, device=device)
    )
    output = sharded_module(full_input)
    print(
        f"rank={rank}: output_fn 返回普通 Tensor，"
        f"完整输出形状={tuple(output.shape)}，完整输出值=\\n{output.detach().cpu()}",
        flush=True,
    )


def torchrun_main(all_api_examples: bool) -> None:
    """供 torchrun 调用：启动器负责创建进程和设置环境变量。"""
    local_rank = int(os.environ["LOCAL_RANK"])  # 本机 GPU 编号
    rank = int(os.environ["RANK"]) # 全局通信身份
    world_size = int(os.environ["WORLD_SIZE"])

    # init_method 默认读取 torchrun 设置的环境变量（env://）。
    dist.init_process_group(backend="nccl")
    try:
        mesh, sharded = documented_api_examples(local_rank, rank, world_size)
        if all_api_examples:
            placement_and_debug_api_examples(mesh, sharded, rank)
            experimental_api_examples(mesh, sharded)
        dtensor_example(local_rank, rank, world_size)
        high_level_dtensor_example(local_rank, rank, world_size)
    finally:
        dist.destroy_process_group()


def spawn_worker(local_rank: int, world_size: int, master_port: int, all_api_examples: bool) -> None:
    """供 mp.spawn 调用：每个子进程都必须手动初始化进程组。"""
    # mp.spawn 把第一个参数传为 0、1、...；单机时它可同时作为 global rank。
    rank = local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh, sharded = documented_api_examples(local_rank, rank, world_size)
        if all_api_examples:
            placement_and_debug_api_examples(mesh, sharded, rank)
            experimental_api_examples(mesh, sharded)
        dtensor_example(local_rank, rank, world_size)
        high_level_dtensor_example(local_rank, rank, world_size)
    finally:
        dist.destroy_process_group()


def spawn_main(world_size: int, master_port: int, all_api_examples: bool) -> None:
    if world_size > torch.cuda.device_count():
        raise ValueError(
            f"world_size={world_size}，但当前只有 {torch.cuda.device_count()} 张 CUDA GPU"
        )
    # spawn 只创建 Python 进程；rank、通信地址、进程组均在 spawn_worker 中手动配置。
    mp.spawn(
        spawn_worker,
        args=(world_size, master_port, all_api_examples),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", choices=("torchrun", "spawn"), default="spawn")
    parser.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--all-api-examples",
        action="store_true",
        help="额外运行页面列出的私有 Placement、调试与实验 API 示例。",
    )
    args = parser.parse_args()

    if args.launcher == "torchrun":
        torchrun_main(args.all_api_examples)
    else:
        spawn_main(args.world_size, args.master_port, args.all_api_examples)
