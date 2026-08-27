# Linux 集群开发环境配置

这个 mini 库假设你在 Linux GPU 集群或容器里运行。Windows 本地适合读代码、跑配置测试；真正训练建议进 CUDA/PyTorch/NeMo 容器。

## 推荐环境

- Python 3.12
- PyTorch CUDA build
- `uv`
- NCCL 可用
- Slurm + Pyxis/Enroot（多节点时）

## 交互式调试

```bash
salloc --account <YOUR_ACCOUNT> -N 1 \
  -J mini-qwen36-debug \
  -p interactive --gpus-per-node=8 -t 120

srun --mpi=pmix --no-kill \
  --container-image <PATH_TO_CONTAINER>.sqsh \
  --container-mounts <SHARED_FS>:<SHARED_FS>,<PATH_TO_REPO>:/opt/Megatron-Bridge \
  --no-container-mount-home \
  --gpus-per-node=8 \
  --pty bash

cd /opt/Megatron-Bridge/mini-megatron
uv sync --group dev --extra train
uv run python -m compileall -q .
```

`mini-megatron` 初始状态是 TODO 骨架，章节测试会失败。完成对应章节后，再单独验证该章：

```bash
uv run python -m unittest tests.test_training_core.TrainingCoreTest
```

这组测试不依赖 GPU，主要确认你手写的 `setup_model_and_optimizer -> forward_backward_no_pipeline -> train_step` 流程和 Megatron-LM 主链路一致。

## 单机训练

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  scripts/train_qwen36.py \
  --recipe qwen36_35b_a3b_tiny_config \
  --train-iters 20 \
  --dtype bf16
```

## Slurm 训练

先编辑：

- `CONTAINER_IMAGE`
- `CONTAINER_MOUNTS`
- `WORKDIR`
- `LOGDIR`
- `#SBATCH --account`
- `#SBATCH --partition`

然后：

```bash
sbatch scripts/slurm_qwen36.sh
```

## 常见问题

`torch.distributed` 初始化失败：

- 确认 Slurm 脚本设置了 `MASTER_ADDR` 和 `MASTER_PORT`。
- 单机多卡建议先用 `torch.distributed.run` 验证。

显存不足：

- 降低 `--seq-length`。
- 降低 `--micro-batch-size`。
- 改用 `qwen36_35b_a3b_debug_config`。

loss 不下降：

- mock 数据是随机 token，只用于验证训练链路，不用于收敛判断。
- 换 JSONL tokens 数据后再观察趋势。

真实 Megatron Bridge 对照：

- Bridge 训练命令必须使用 `uv run python -m torch.distributed.run`。
- 集群多节点优先用 srun-native；Bridge 会从 Slurm 环境补齐 rank/world/master 信息。
- `NEMO_HOME`、`HF_HOME`、`UV_CACHE_DIR` 应放在共享文件系统。
- `3rdparty/Megatron-LM/` 是上游 submodule，学习时重点读它的训练流程；本 mini 项目不修改 submodule 内容。
