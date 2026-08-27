# Megatron Bridge + Megatron-LM 代码走读路线

这条路线专门服务于理解“一个 Qwen3.6 训练任务如何启动并进入训练循环”。现在仓库已经有 `3rdparty/Megatron-LM/`，建议把代码分成三层读：

1. Megatron Bridge：负责 recipe、HF/Megatron config 转换、训练入口封装。
2. Megatron-LM：负责训练主循环、model provider、forward-backward schedule。
3. Megatron-Core：负责 tensor/pipeline/context/expert parallel 的底层并行和 kernel 选择。

mini 对应：`mini_megatron/trainer.py` 是 Bridge 风格编排骨架，`mini_megatron/training_core.py` 是 Megatron-LM 风格训练骨架，`mini_megatron/modeling_qwen36.py` 是小型模型本体骨架。它们现在只保留签名和 TODO，具体实现按 `tutorial/README.md` 手写。

## 1. 从入口开始

读 `scripts/training/run_recipe.py`：

- `parse_args()`：命令行如何表达 recipe、dataset、step function。
- `load_recipe()`：recipe 名字如何变成 `ConfigContainer`。
- `process_config_with_overrides()`：CLI override 如何覆盖 recipe 默认值。
- `load_forward_step()`：文本模型默认使用 `gpt_step`。
- `main()`：最后选择 `pretrain()` 或 `finetune()`。

mini 对应：在 `scripts/train_qwen36.py` 和 `mini_megatron/recipes.py` 填 CLI、override 和 recipe 逻辑。

## 2. 看 Recipe 如何搭配置

Qwen3.6 在这个仓库里复用 Qwen3.5-VL MoE 支持。重点读：

- `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`
- `src/megatron/bridge/recipes/common.py`
- `src/megatron/bridge/recipes/__init__.py`

你要抓住三个问题：

1. model provider 从哪里来。
2. tokenizer/dataset 默认是什么。
3. TP/PP/EP、batch size、LR、checkpoint 默认在哪里设。

mini 对应：在 `mini_megatron/config.py` 和 `mini_megatron/recipes.py` 填 tiny/debug config。

## 3. 看 Bridge 做了什么

读 `src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py`，再对照 dense Qwen3 的
`src/megatron/bridge/models/qwen/qwen3_bridge.py`。

重点不是背参数名，而是理解：

- Hugging Face config 如何转换成 Megatron provider。
- QKV、gate/up projection、MoE expert 权重为什么需要特殊 mapping。
- Qwen3.6 为什么能复用 Qwen3.5-VL MoE bridge。

mini 对应：在 `mini_megatron/modeling_qwen36.py` 填 tiny Qwen 结构，不做权重转换。

## 4. 看配置总线

读 `src/megatron/bridge/training/config.py`：

- `ConfigContainer`
- `validate()`
- `runtime_config_update()`
- `get_data_parallel_size()`

Megatron Bridge 的核心设计是：recipe 先创建懒配置，用户覆盖后，训练开始前统一验证和补齐派生字段。

mini 对应：在 `ConfigContainer.validate()` 和 `gradient_accumulation_steps()` 填最小配置总线逻辑。

## 5. 看 Setup

读 `src/megatron/bridge/training/setup.py`：

- `initialize_megatron()` 初始化 distributed 和模型并行组。
- `build_tokenizer()` 决定 vocab。
- `_build_distributed_model()` materialize model provider。
- `setup_optimizer()` 创建 optimizer/scheduler。
- `setup_data_iterators()` 创建数据迭代器。
- checkpoint manager 负责恢复/保存状态。

mini 对应：

- `distributed.py`：填 torchrun/Slurm 环境解析。
- `trainer.py`：填 model/optimizer/scheduler/data provider。
- `training_core.py`：填 `setup_model_and_optimizer()` materialize provider 产物。

## 6. 看 Megatron-LM 训练核心

Bridge 最终会落到 Megatron-LM/Megatron-Core 的训练语义上。重点按这个顺序读：

1. `3rdparty/Megatron-LM/pretrain_gpt.py`
   - `train_valid_test_datasets_provider()`：GPT 数据集入口。
   - `get_batch()`：TP rank 广播、CP rank 切片、labels/loss_mask/position_ids/attention_mask。
   - `loss_func()`：token loss mask、NaN/Inf/spiky loss 检查、返回日志字典。
   - `forward_step()`：拿 batch，调用 model，返回给 schedule 使用的 loss function。
   - 文件底部 `pretrain(...)`：把 dataset provider、model provider、forward step 交给训练引擎。

2. `3rdparty/Megatron-LM/model_provider.py`
   - `model_provider()`：根据 `pre_process/post_process/vp_stage` 构建当前 pipeline stage 的模型。
   - 你要注意它不是直接 new 一个完整模型，而是服务 pipeline/virtual pipeline 切分。

3. `3rdparty/Megatron-LM/gpt_builders.py`
   - `gpt_builder()`：选择 Transformer layer spec，组装 `GPTModel`。
   - TE/local/MoE/heterogeneous layer 的选择都在这里汇聚。

4. `3rdparty/Megatron-LM/megatron/training/training.py`
   - `pretrain()`：初始化 Megatron、构建模型/优化器/数据迭代器、进入 train。
   - `get_model()`：处理 PP/VP stage，调用 `model_provider_func`，移动到 CUDA，包装 FP16/DDP/FSDP。
   - `setup_model_and_optimizer()`：构建 model、optimizer、scheduler，并加载 checkpoint。
   - `train_step()`：zero grad、调用 forward-backward schedule、optimizer step、scheduler step、汇总 loss。

5. `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
   - `get_forward_backward_func()`：按 PP world size 和 interleaving 选择 schedule。
   - `forward_backward_no_pipelining()`：mini 版最接近的路径。
   - 其他 1F1B/interleaved schedule 先知道存在，不必第一遍啃完。

mini 对应：

- `forward_step.py`：填最小化 `pretrain_gpt.forward_step()`，直接返回 loss 和 `{"lm loss": ...}`。
- `training_core.py`：填 `setup_model_and_optimizer()`、`forward_backward_no_pipeline()`、`train_step()`。
- `trainer.py`：填 provider、schedule、checkpoint 和日志编排。

## 7. 看 Bridge 训练循环

读 `src/megatron/bridge/training/pretrain.py` 和 `src/megatron/bridge/training/train.py`：

- `pretrain()` 建 `GlobalState` 并调用 `_pretrain()`。
- `_pretrain()` 调 setup，然后进入 `train()`。
- `train()` 管理 iteration、logging、eval、checkpoint。
- `train_step()` 做 forward-backward、optimizer step、scheduler step。

mini 对应：在 `trainer.run_training()` 中调用 `training_core.train_step()`。

## 8. 看单步 forward

读 `src/megatron/bridge/training/gpt_step.py` 和 `src/megatron/bridge/training/losses.py`：

- 从 data iterator 拿 batch。
- 根据 PP/CP 状态选择需要的字段。
- 调 `model(input_ids, position_ids, attention_mask, labels)`。
- 返回一个 loss function 给 Megatron-Core schedule 调用。

mini 对应：

- `forward_step.causal_lm_forward_step()`：从 batch 调模型并抽取 loss。
- `Qwen36ForCausalLM.forward()`：计算 logits 和 causal LM loss。
- `training_core.forward_backward_no_pipeline()`：代替 Megatron-Core schedule 调 backward。

## 9. 第一遍不要钻太深的部分

第一遍目标是跑通训练主链，不建议卡在这些点：

- Transformer Engine kernel 细节。
- pipeline interleaving 和 virtual pipeline stage 的全部边界条件。
- distributed optimizer 的 sharded state。
- checkpoint conversion 的每一个参数名。
- MoE expert/tensor parallel 的跨 rank 权重切片。

先把这条线在脑子里跑起来：

```text
run_recipe.py
  -> recipe/config/bridge provider
  -> Megatron-LM pretrain()
  -> setup_model_and_optimizer()
  -> get_model(model_provider)
  -> get_forward_backward_func()
  -> train_step(forward_step)
  -> optimizer/scheduler/checkpoint
```
