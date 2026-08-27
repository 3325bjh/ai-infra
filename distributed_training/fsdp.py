"""FSDP2（``fully_shard``）最小训练示例。

运行方式（两张 GPU）：
    torchrun --standalone --nproc_per_node=2 fsdp.py --epochs 5

本示例遵循 PyTorch FSDP2 教程：先对每个 TransformerBlock 调用
``fully_shard``，再对根 Transformer 调用它；优化器必须在分片之后创建。

检查点可选两种格式：``dtensor`` 将完整模型状态写入单个文件；``dcp``
使用 Distributed Checkpoint 并行保存模型与优化器的分片状态到一个目录。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import distribute_tensor


class TransformerBlock(nn.Module):
    """一个适合演示 FSDP2 嵌套分片的简化 Transformer block。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attention_norm(x)
        x = residual + self.attention(x, x, x, need_weights=False)[0]
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, num_layers: int, num_heads: int, mlp_ratio: int
    ) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 fully_shard 示例")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=8, help="每个 rank 的 batch 大小。")
    parser.add_argument("--seq-len", type=int, default=128, help="输入序列长度。")
    parser.add_argument("--vocab-size", type=int, default=1024, help="词表大小。")
    parser.add_argument("--dim", type=int, default=256, help="Transformer 隐藏层维度。")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer block 数量。")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数。")
    parser.add_argument("--mlp-ratio", type=int, default=4, help="MLP 中间层扩张倍数。")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率。")
    parser.add_argument("--max-norm", type=float, default=1.0, help="梯度裁剪阈值。")
    parser.add_argument(
        "--explicit-prefetching", action="store_true",
        help="启用显式预取；默认使用 FSDP2 的隐式预取。",
    )
    parser.add_argument(
        "--num-to-forward-prefetch", type=int, default=2,
        help="显式预取时，在前向传播中预取的后续 block 数量。",
    )
    parser.add_argument(
        "--num-to-backward-prefetch", type=int, default=2,
        help="显式预取时，在反向传播中预取的前序 block 数量。",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="前向和反向使用 bfloat16，梯度归约使用 float32。",
    )
    parser.add_argument(
        "--load-state", type=str, default=None,
        help="从指定的检查点恢复训练；格式由 --checkpoint-format 决定。",
    )
    parser.add_argument(
        "--save-state", type=str, default=None,
        help="将训练后的检查点保存到指定位置；格式由 --checkpoint-format 决定。",
    )
    parser.add_argument(
        "--checkpoint-format", choices=("dtensor", "dcp"), default="dtensor",
        help="检查点格式：dtensor 为单个完整模型文件；dcp 为分布式检查点目录。",
    )
    return parser.parse_args()


class AppState(Stateful):
    """让 DCP 统一保存和恢复 FSDP2 模型与优化器状态。"""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optimizer": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def load_full_state_dict(model: nn.Module, path: str) -> None:
    """将磁盘上的完整 Tensor 状态转换为 FSDP2 所需的 DTensor 状态。"""
    full_state_dict = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    meta_sharded_state_dict = model.state_dict()
    sharded_state_dict = {}
    for name, full_tensor in full_state_dict.items():
        sharded_parameter = meta_sharded_state_dict[name]
        sharded_tensor = distribute_tensor(
            full_tensor, sharded_parameter.device_mesh, sharded_parameter.placements
        )
        sharded_state_dict[name] = nn.Parameter(sharded_tensor)
    model.load_state_dict(sharded_state_dict, assign=True)


def save_full_state_dict(model: nn.Module, path: str) -> None:
    """从 FSDP2 的 DTensor 状态按 rank 0 保存完整的 CPU 状态。"""
    cpu_state_dict = {}
    for name, sharded_parameter in model.state_dict().items():
        full_parameter = sharded_parameter.full_tensor()
        if dist.get_rank() == 0:
            cpu_state_dict[name] = full_parameter.cpu()
        else:
            del full_parameter
    if dist.get_rank() == 0:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(cpu_state_dict, path)
        print(f"已保存完整模型状态：{path}", flush=True)
    dist.barrier()


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: str, checkpoint_format: str
) -> None:
    if checkpoint_format == "dtensor":
        load_full_state_dict(model, path)
    else:
        # DCP 原地加载：AppState 会提供当前 world size 下的分片信息。
        dcp.load({"app": AppState(model, optimizer)}, checkpoint_id=path)
    if dist.get_rank() == 0:
        print(f"已加载 {checkpoint_format} 检查点：{path}", flush=True)


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: str, checkpoint_format: str
) -> None:
    if checkpoint_format == "dtensor":
        save_full_state_dict(model, path)
    else:
        # 所有 rank 并行写入各自分片；path 是检查点目录而非单个文件。
        dcp.save({"app": AppState(model, optimizer)}, checkpoint_id=path)
        if dist.get_rank() == 0:
            print(f"已保存 dcp 分布式检查点目录：{path}", flush=True)
        dist.barrier()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("此 FSDP2 示例需要 CUDA GPU。")
    if args.dim % args.num_heads != 0:
        raise ValueError("--dim 必须能被 --num-heads 整除。")
    if args.num_to_forward_prefetch < 0 or args.num_to_backward_prefetch < 0:
        raise ValueError("预取数量不能为负数。")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        model = Transformer(
            args.vocab_size, args.dim, args.num_layers, args.num_heads, args.mlp_ratio
        )

        fsdp_kwargs = {}
        if args.mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )

        # FSDP2：先分片每个 block，再分片根模块。默认会启用隐式预取。
        for layer in model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        assert isinstance(model, FSDPModule)

        if args.explicit_prefetching:
            # 显式指定前向预取顺序：在第 i 层预取后续的若干层。
            for index, layer in enumerate(model.layers):
                following = model.layers[
                    index + 1 : index + 1 + args.num_to_forward_prefetch
                ]
                layer.set_modules_to_forward_prefetch(list(following))
            # 反向传播的访问顺序相反：在第 i 层预取之前的若干层。
            for index, layer in enumerate(model.layers):
                preceding = model.layers[
                    max(0, index - args.num_to_backward_prefetch) : index
                ]
                layer.set_modules_to_backward_prefetch(list(reversed(preceding)))

        # fully_shard() 后参数是 DTensor；优化器必须在此之后创建。
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if args.load_state:
            load_checkpoint(model, optimizer, args.load_state, args.checkpoint_format)

        for epoch in range(args.epochs):
            if args.explicit_prefetching:
                # 提前触发第一个 all-gather，以和模型调用前的 CPU 工作重叠。
                model.unshard()
            tokens = torch.randint(
                args.vocab_size, (args.batch_size, args.seq_len), device=device
            )
            targets = torch.randint(
                args.vocab_size, (args.batch_size, args.seq_len), device=device
            )
            logits = model(tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if dist.get_rank() == 0:
                print(f"epoch={epoch + 1}, loss={loss.item():.4f}", flush=True)
        if args.save_state:
            save_checkpoint(model, optimizer, args.save_state, args.checkpoint_format)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
