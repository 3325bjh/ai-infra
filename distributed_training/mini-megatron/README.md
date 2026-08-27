# mini-megatron

`mini-megatron` 是一个 **空骨架练习册**，不是参考答案代码库。它只保留学习 Megatron Bridge + Megatron-LM 训练主链路所需的关键文件、类、函数签名和 TODO docstring，核心实现需要你按 `tutorial/README.md` 一章章手写。

目标模型按仓库里的 **Qwen3.6-35B-A3B** 来理解，但练习实现应使用 tiny 配置：只保留 RMSNorm、RoPE、GQA、SwiGLU、top-k MoE、causal LM loss、DDP、Megatron-LM 风格的 `setup_model_and_optimizer -> forward_backward -> train_step`、checkpoint 和 Slurm 启动。

## 怎么使用

1. 先读 [docs/code-reading-route.md](docs/code-reading-route.md)，建立真实项目的主链路地图。
2. 再读 [tutorial/README.md](tutorial/README.md)，按章节手写 TODO。
3. 每章先跑对应测试，看到失败；写完该章后再跑，直到该章变绿。
4. 训练相关章节需要在 Linux GPU 集群或 CUDA 容器里完成。

初始状态下，大多数测试会因为 `NotImplementedError` 失败。这是刻意设计的：测试就是你的练习验收目标。

## uv 环境

只安装轻量基础包：

```bash
cd mini-megatron
uv sync --no-dev
```

开发环境：

```bash
uv sync --group dev
```

训练环境，包含 PyTorch：

```bash
uv sync --group dev --extra train
```

训练入口既可以用脚本，也可以用 uv 生成的 console script：

```bash
uv run python scripts/train_qwen36.py --help
uv run mini-qwen36 --help
```

## 骨架文件

- `mini_megatron/config.py`：配置 dataclass 和校验入口。
- `mini_megatron/recipes.py`：Qwen3.6 tiny/debug recipe 入口。
- `mini_megatron/distributed.py`：torchrun/Slurm rank 环境解析。
- `mini_megatron/data.py`：mock/JSONL token 数据入口。
- `mini_megatron/modeling_qwen36.py`：Qwen 风格模型 block 签名。
- `mini_megatron/forward_step.py`：causal LM forward step 签名。
- `mini_megatron/training_core.py`：Megatron-LM 风格 setup、schedule、train step 签名。
- `mini_megatron/trainer.py`：把 distributed、provider、schedule、日志、checkpoint 串起来的入口。
- `mini_megatron/cli.py`：训练 CLI 入口。
- `scripts/train_qwen36.py`：训练 CLI 骨架。
- `scripts/slurm_qwen36.sh`：Linux 集群 Slurm 模板。

## 真实代码对照

- Bridge 入口：`scripts/training/run_recipe.py`
- Bridge recipe：`src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`
- Bridge config：`src/megatron/bridge/training/config.py`
- Bridge training：`src/megatron/bridge/training/setup.py`、`train.py`、`gpt_step.py`
- Megatron-LM GPT 入口：`3rdparty/Megatron-LM/pretrain_gpt.py`
- Megatron-LM provider/builder：`3rdparty/Megatron-LM/model_provider.py`、`gpt_builders.py`
- Megatron-LM training core：`3rdparty/Megatron-LM/megatron/training/training.py`
- Megatron-Core schedule：`3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`

## 验证方式

骨架语法检查：

```bash
cd mini-megatron
uv run python -m compileall -q .
```

章节测试示例：

```bash
cd mini-megatron
uv run python -m unittest tests.test_config.ConfigTest
uv run python -m unittest tests.test_training_core.TrainingCoreTest
```

集群环境说明见 [docs/linux-cluster-setup.md](docs/linux-cluster-setup.md)。
