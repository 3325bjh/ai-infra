## 实验结果：DDP、Optimizer State Sharding 与 FSDP

以下结果在单节点、2 张 NVIDIA RTX 2080 Ti 上测得。为了适应显存容量，使用
`small` 模型配置（`d_model=768`、`d_ff=3072`、12 层、12 个 attention heads），每个
rank 的 batch size 为 4，context length 为 512。每组实验先 warm-up 5 步，再测量 10 步；
表中的 iteration 时间取两个 rank 中较慢者的平均单步时间。

| 并行策略 | Optimizer | 分布式 iteration（ms） | 梯度同步等待（ms） | 同步等待占比 |
| --- | --- | ---: | ---: | ---: |
| Naive DDP（逐参数 all-reduce） | AdamW | 275.872 | 23.646 | 8.6% |
| Flat-gradient DDP（一次扁平 all-reduce） | AdamW | 276.622 | 25.188 | 9.1% |
| Overlap DDP（逐参数异步 all-reduce） | AdamW | 264.159 | 5.987 | 2.3% |
| Bucketed DDP（25 MiB bucket） | AdamW | 257.523 | 0.002 | 0.0% |
| Naive DDP | Sharded AdamW | 275.657 | 23.950 | 8.7% |
| FSDP（FP32） | AdamW | 261.686 | 0.866 | 0.3% |


运行示例：
```bash
uv run python cs336_systems/ddp_benchmark.py \
  --ddp-implementation naive \
  --optimizer-implementation adamw \
  --model-size small 
```

```bash
uv run python cs336_systems/ddp_benchmark.py \
  --ddp-implementation flat \
  --optimizer-implementation adamw \
  --model-size small \
  --master-port 29501
```

```bash
uv run python cs336_systems/ddp_benchmark.py \
  --ddp-implementation bucketed \
  --bucket-size-mb 25 \
  --optimizer-implementation adamw \
  --model-size small
```

```bash
uv run python cs336_systems/ddp_benchmark.py \
  --ddp-implementation naive \
  --optimizer-implementation sharded \
  --model-size small 
```

```bash
uv run python cs336_systems/ddp_benchmark.py \
  --ddp-implementation fsdp \
  --optimizer-implementation adamw \
  --compute-dtype fp32 \
  --model-size small 
```


