# Agent Instructions for PyTorch Distributed Learning Project

## 1. 项目背景

这是一个 **PyTorch Distributed 分布式学习项目**。

项目主要用于学习和实践 PyTorch 分布式训练相关技术，包括但不限于：

- `torch.distributed`
- DistributedDataParallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DistributedSampler
- torchrun 多进程启动方式
- 多 GPU / 多节点训练
- 分布式通信机制（collective communication）
- 梯度同步与参数更新流程

---


---

# 2. 注释规范

## 2.1 中文注释要求

当用户要求添加代码注释时：

**必须使用中文添加注释。**

注释要求：

- 使用简洁准确的中文描述代码作用。
- 解释关键逻辑，而不是简单翻译代码。
- 对分布式相关概念进行必要说明。
- 结合一个简单的例子进行解释

# 3.不确定内容处理规则
当无法确定答案：

或者涉及：

- PyTorch API 行为
- torch.distributed 使用方式
- DDP/FSDP 工作机制
- NCCL 通信行为
- torchrun 参数

必须优先查阅 PyTorch 官方文档：
- https://docs.pytorch.org/docs/2.13/distributed.html
- https://docs.pytorch.ac.cn/tutorials/beginner/dist_overview.html
