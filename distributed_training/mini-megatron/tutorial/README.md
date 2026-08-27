# 手写 mini-megatron 教程

这个目录是你的练习路线。`mini_megatron/` 里的代码刻意只有骨架：类、函数、参数、返回值、docstring 都在，核心实现都留给你写。每章按同一个节奏推进：

1. 先读真实项目代码。
2. 回到 mini 骨架，看 TODO 和函数签名。
3. 先跑该章测试，确认失败。
4. 手写实现。
5. 再跑该章测试。

配合阅读：[../docs/code-reading-route.md](../docs/code-reading-route.md)。

## Chapter 0: 建立主链路地图

你要先回答一个问题：Qwen3.6 训练任务从命令行启动后，哪些对象依次被创建？

先读：

- `scripts/training/run_recipe.py`
- `src/megatron/bridge/training/setup.py`
- `src/megatron/bridge/training/train.py`
- `3rdparty/Megatron-LM/pretrain_gpt.py`
- `3rdparty/Megatron-LM/megatron/training/training.py`

你需要在笔记里画出这条线：

```text
run_recipe.py
  -> recipe/config
  -> model provider
  -> dataset provider
  -> setup_model_and_optimizer
  -> forward_backward schedule
  -> train_step
  -> optimizer/scheduler/checkpoint
```

本章不写代码。

## Chapter 1: Config 和 Recipe

要实现的文件：

- `mini_megatron/config.py`
- `mini_megatron/recipes.py`

先读：

- 走读路线第 1、2、4 节。
- `src/megatron/bridge/training/config.py`
- `src/megatron/bridge/recipes/common.py`
- `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`

你要实现：

- `Qwen36ModelConfig.head_dim`
- `Qwen36ModelConfig.validate()`
- `TrainConfig.validate()`
- `DataConfig.validate()`
- `ConfigContainer.validate()`
- `ConfigContainer.gradient_accumulation_steps(world_size)`
- `qwen36_35b_a3b_tiny_config()`
- `qwen36_35b_a3b_debug_config()`
- `load_recipe(name)`

关键检查点：

- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`
- `num_experts_per_tok <= num_experts`
- `data.vocab_size == model.vocab_size`
- `global_batch_size % (micro_batch_size * world_size) == 0`

先跑：

```bash
cd mini-megatron
uv run python -m unittest tests.test_config.ConfigTest
```

## Chapter 2: Distributed Context

要实现的文件：

- `mini_megatron/distributed.py`

先读：

- 走读路线第 5 节。
- `src/megatron/bridge/training/setup.py`
- `src/megatron/bridge/utils/common_utils.py`
- `src/megatron/bridge/utils/slurm_utils.py`

你要实现：

- `DistributedContext.is_distributed`
- `DistributedContext.is_rank0`
- `_get_int_env(primary, fallback, default)`
- `_torch_cuda_device(local_rank)`
- `resolve_distributed_context()`
- `init_distributed(config)`
- `barrier()`
- `cleanup_distributed(config)`
- `rank0_log(ctx, message, *args)`

关键检查点：

- torchrun 使用 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`。
- Slurm 使用 `SLURM_PROCID`、`SLURM_LOCALID`、`SLURM_NTASKS`。
- CPU 环境下 `nccl` 要降级为 `gloo`。
- 不要用裸 `print()`，只通过 logger 输出 rank 0 日志。

先跑：

```bash
cd mini-megatron
uv run python -m unittest tests.test_distributed.DistributedEnvTest
```

## Chapter 3: Data 和 Batch

要实现的文件：

- `mini_megatron/data.py`

先读：

- 走读路线第 6 节里 `pretrain_gpt.py` 的 `get_batch()`。
- `src/megatron/bridge/data/utils.py`
- `src/megatron/bridge/data/loaders.py`

你要实现：

- `ByteTokenizer.encode(text)`
- `MockTokenDataset`
- `JsonlTokenDataset`
- `collate_lm_batch(samples)`
- `build_dataloader(data_config, ctx, seed, micro_batch_size)`

关键检查点：

- 每条样本生成 `seq_length + 1` 个 token。
- `input_ids = tokens[:-1]`。
- `labels = tokens[1:]`。
- 分布式训练时使用 `DistributedSampler`，单进程时可以 shuffle。
- JSONL 同时支持 `{"tokens": [...]}` 和 `{"text": "..."}`。

先跑：

```bash
cd mini-megatron
uv run python -m unittest tests.test_data.DataTest
```

## Chapter 4: Qwen3.6 Tiny Model

要实现的文件：

- `mini_megatron/modeling_qwen36.py`

先读：

- 走读路线第 3、6 节。
- `src/megatron/bridge/models/qwen/qwen3_bridge.py`
- `src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py`
- `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py`
- `3rdparty/Megatron-LM/model_provider.py`
- `3rdparty/Megatron-LM/gpt_builders.py`

你要按顺序实现：

- `RMSNorm`
- `RotaryEmbedding`
- `apply_rotary`
- `Qwen36Attention`
- `Qwen36MLP`
- `Qwen36SparseMoE`
- `Qwen36DecoderLayer`
- `Qwen36ForCausalLM`

关键检查点：

- Attention 是 GQA：Q heads 多于 KV heads，KV 需要 repeat 到 Q head 数。
- RoPE 应作用在 Q/K。
- causal mask 不能看未来 token。
- MLP 是 SwiGLU：`silu(gate) * up -> down`。
- MoE 是教学版 top-k routing，不需要实现 expert parallel。
- `Qwen36ForCausalLM.forward()` 返回 `CausalLMOutput(logits, loss)`。

先跑：

```bash
cd mini-megatron
uv run python -m unittest tests.test_modeling.ModelingTest
```

## Chapter 5: Forward Step 和 Loss

要实现的文件：

- `mini_megatron/forward_step.py`

先读：

- 走读路线第 6、8 节。
- `3rdparty/Megatron-LM/pretrain_gpt.py` 的 `loss_func()` 和 `forward_step()`。
- `src/megatron/bridge/training/gpt_step.py`
- `src/megatron/bridge/training/losses.py`

你要实现：

- `causal_lm_forward_step(batch, model)`

关键检查点：

- 从 batch 中取 `input_ids` 和 `labels`。
- 调 `model(input_ids=..., labels=...)`。
- 确认模型返回 loss。
- 返回 `(loss, {"lm loss": loss_float})`。

本章的测试在 Chapter 6 的 `test_training_core.py` 中会间接覆盖。

## Chapter 6: Megatron-LM Training Core

要实现的文件：

- `mini_megatron/training_core.py`

先读：

- 走读路线第 6 节。
- `3rdparty/Megatron-LM/megatron/training/training.py`
  - `get_model()`
  - `setup_model_and_optimizer()`
  - `train_step()`
- `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
  - `get_forward_backward_func()`
  - `forward_backward_no_pipelining()`

你要实现：

- `MiniOptimizerParamScheduler`
- `cosine_lr(iteration, train_config)`
- `setup_model_and_optimizer(...)`
- `forward_backward_no_pipeline(...)`
- `train_step(...)`
- `average_metrics(...)`

关键检查点：

- `setup_model_and_optimizer()` 只负责按顺序调用 provider。
- `forward_backward_no_pipeline()` 按 `num_microbatches` 循环。
- 每个 microbatch 的 loss 要除以 `num_microbatches` 再 backward。
- `train_step()` 顺序是 zero grad、forward-backward、clip、optimizer step、scheduler step。
- scheduler step 的样本数是 `num_microbatches * micro_batch_size * world_size`。

先跑：

```bash
cd mini-megatron
uv run python -m unittest tests.test_training_core.TrainingCoreTest
```

## Chapter 7: Trainer Wiring

要实现的文件：

- `mini_megatron/trainer.py`

先读：

- 走读路线第 5、7 节。
- `src/megatron/bridge/training/setup.py`
- `src/megatron/bridge/training/train.py`
- `3rdparty/Megatron-LM/megatron/training/training.py` 的 `pretrain()`

你要实现：

- `set_seed(seed)`
- `_autocast_context(device, dtype)`
- `_move_batch(batch, device)`
- `_unwrap_model(model)`
- `save_checkpoint(...)`
- `build_model(cfg, device)`
- `maybe_wrap_ddp(model, cfg, ctx)`
- `run_training(cfg)`

关键检查点：

- `run_training()` 不应该重新发明训练核心，而是调用 Chapter 6 的 `setup_model_and_optimizer()` 和 `train_step()`。
- provider 在这里组装：model provider、optimizer provider、scheduler provider、data iterator provider。
- DDP 只在 `cfg.distributed.ddp and ctx.is_distributed` 时启用。
- checkpoint 只在 rank 0 保存。
- `finally` 中要清理 distributed process group。

完成后在 Linux/PyTorch 环境跑：

```bash
cd mini-megatron
uv run python scripts/train_qwen36.py \
  --recipe qwen36_35b_a3b_debug_config \
  --train-iters 2 \
  --dtype fp32
```

## Chapter 8: Launcher 和 Slurm

要实现的文件：

- `mini_megatron/cli.py`
- `scripts/train_qwen36.py`
- `scripts/slurm_qwen36.sh`

先读：

- 走读路线第 1 节。
- `scripts/training/run_recipe.py`
- `scripts/training/launch_with_sbatch.sh`
- `skills/multi-node-slurm/SKILL.md`
- `docs/linux-cluster-setup.md`

你要实现：

- `apply_cli_overrides(cfg, args)`
- 检查 CLI 覆盖后调用 `cfg.validate()`
- 确认 `uv run mini-qwen36 --help` 和 `uv run python scripts/train_qwen36.py --help` 都可用
- 根据你的集群填写 Slurm 模板里的 account、partition、container、mount、workdir、logdir

关键检查点：

- `--seq-length` 要同时覆盖 `model.max_position_embeddings` 和 `data.seq_length`。
- `--global-batch-size` 改动后要重新 validate。
- Slurm 中要设置 `MASTER_ADDR` 和 `MASTER_PORT`。
- 多节点建议先用 1 节点 8 卡验证。

脚本语法检查：

```bash
bash -n scripts/slurm_qwen36.sh
```

## Chapter 9: 第一轮集群训练

先跑最小单机：

```bash
cd mini-megatron
uv run python -m torch.distributed.run --nproc_per_node=1 \
  scripts/train_qwen36.py \
  --recipe qwen36_35b_a3b_debug_config \
  --train-iters 2 \
  --dtype fp32
```

再跑单机 8 卡：

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  scripts/train_qwen36.py \
  --recipe qwen36_35b_a3b_tiny_config \
  --train-iters 20 \
  --dtype bf16
```

最后提交 Slurm：

```bash
sbatch scripts/slurm_qwen36.sh
```

你需要观察：

- 每个 iteration 有 `lm loss`。
- lr 按 warmup/cosine 变化。
- rank 0 负责日志和 checkpoint。
- checkpoint 目录里出现 `iter_*.pt`。
