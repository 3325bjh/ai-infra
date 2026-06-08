# AI Infra 相关学习路线和资料参考


## 1. 基础知识

### 1.1 基础数学

#### 线性代数

- **重点内容**：理解标量、向量、矩阵、张量、线性变换和基本矩阵乘法运算规则；了解这些内容在深度学习中的应用。
- **可选内容**：特征值、SVD、范数、Hessian 矩阵、矩阵求导、特殊矩阵（正交、正定）及其在深度学习中的应用。
- **参考资料**：[UP 主汉语配音《线性代数的本质》合集（3Blue1Brown 官方双语）](https://www.bilibili.com/video/BV1ib411t7YR/)

#### 微积分

- **重点内容**：理解求导、梯度概念和求导链式法则。

#### 优化技术

- **重点内容**：理解基本的梯度下降方法。
- **可选内容**：凸优化技术，如 Proximal Gradient、拉格朗日乘子、岭回归、凸松弛等。这些方法在经典传统机器学习方法中常见，在深度学习方向可以不深入。
- **参考资料**：
  - [如何理解“梯度下降法”？什么是“反向传播”？](https://www.bilibili.com/video/BV1Zg411T71b/)
  - [随机梯度下降、牛顿法、动量法、Nesterov、AdaGrad、RMSprop、Adam](https://www.bilibili.com/video/BV1r64y1s7fU/)

### 1.2 基础理论、模型与工具

#### 多层感知机（MLP）模型与训练

- **重点内容**：理解参数、样本、标签等概念，MLP 结构，训练过程（前向传播与反向传播等），epoch、iter、batch 概念及过拟合相关内容。
- **可选内容**：SGD、Adam 等优化器原理。
- **参考资料**：
  - [吴恩达机器学习](https://www.bilibili.com/video/BV164411b7dx/)
  - [动手学深度学习：多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)

#### CNN

- **重点内容**：理解 CNN 中 kernel 的概念和卷积层。
- **可选内容**：经典视觉模型 VGG、MobileNet、ShuffleNet；目标检测系列 Yolo、SSD、R-CNN。
- **参考资料**：[动手学深度学习：从全连接层到卷积](https://zh.d2l.ai/chapter_convolutional-neural-networks/why-conv.html)，并参考第六章“卷积神经网络”和第七章“现代卷积神经网络”。

#### ResNet

- **重点内容**：理解经典残差网络的结构。ResNet 广泛应用于各种模型，包括 LLM。

#### RNN 相关

- **重点内容**：RNN 模型以前常用于 NLP 处理。虽然目前不常用，但仍需理解相关 NLP 基本概念。

#### LSTM

- **重点内容**：可选。了解处理时序数据的经典模型。
- **参考资料**：[动手学深度学习](https://zh.d2l.ai/)

## 2. LLM 模型

### 2.1 基础

#### 基础 NLP 概念

- **重点内容**：理解词向量、分词、token、Embedding 的含义。
- **参考资料**：
  - [词向量与 Embedding 究竟是怎么回事？](https://kexue.fm/archives/4122)
  - [一文理解 Embedding](https://bytedance.larkoffice.com/docx/VR2kdH3nOoMrVqxATzFc8omwnyc)

#### Transformer（Attention）基本结构

- **重点内容**：Transformer 是目前最有影响力的基础模型，需要非常清楚地理解其结构。
- **参考资料**：
  - [Transformer 论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE/)
  - [从编解码和词嵌入开始，一步一步理解 Transformer](https://www.bilibili.com/video/BV1XH4y1T76e/)

### 2.2 模型

#### LLM 基础

- **重点内容**：理解 LLM 经典结构（GPT-1/2/3）、训练模式、预训练等基础概念。
- **参考资料**：
  - 推荐阅读 GPT-1/2/3 论文，并配合李沐的论文精读。
  - [InstructGPT 论文精读](https://www.bilibili.com/video/BV1hd4y187CR/)
  - [GPT、GPT-2、GPT-3 论文精读](https://www.bilibili.com/video/BV1AF411b7xQ)

#### 多模态

- **重点内容**：
  - 参考 ViT、CLIP 等模型，理解视觉编码器及信息融合原理。
  - 了解 Qwen-VL 如何处理视觉和文本信息融合。
- **参考资料**：[李沐论文精读](https://github.com/mli/paper-reading/)中包含多模态相关模型介绍。

#### MoE 结构

- **重点内容**：
  - 理解稀疏 MoE 模型。
  - 理解 MoE 中的重要问题，如如何路由、如何确保专家之间的平衡、专家数量和规模的影响。
- **参考资料**：
  - [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
  - [混合专家模型（MoE）详解](https://zhuanlan.zhihu.com/p/674698482)
  - [Switch Transformer](https://arxiv.org/pdf/2101.03961)
  - [DeepSeekMoE](https://arxiv.org/html/2401.06066v1)
  - [MoE 环游记：从几何意义出发](https://kexue.fm/archives/10699)，以及该系列第 2、3、4 篇文章。

#### 生成模型与图像生成

- **重点内容**：了解 Diffusion Transformer，将扩散模型与 Transformer 结合；对图像或视频生成感兴趣可进一步研究。
- **建议模型**：参考 DiT。

#### 近期 SOTA 模型

- **重点内容**：学习 DeepSeek-R1、QwQ-32B、Qwen3、Qwen2.5-VL 等模型结构。

## 3. Post-Training

### 3.1 概念

- **重点内容**：理解指令微调、SFT、Alignment 等概念。

### 3.2 强化学习

#### 强化学习基础

- **重点内容**：理解环境、Action、Reward 等基本概念，以及 Q 网络、Actor-Critic 等算法。
- **参考资料**：[Easy-RL](https://github.com/datawhalechina/easy-rl)，可以将代码下载后运行实践。

#### 微调方法

- **重点内容**：理解 RLHF 的基本过程、DPO 技术等。
- **可选内容**：理解并实践 DeepSeek-R1 的 GRPO。

### 3.3 高效微调

- **重点内容**：了解 LoRA、QLoRA、Adapter Tuning、Prompt Tuning 等技术。

### 3.4 蒸馏

- **重点内容**：理解蒸馏的基本框架，以及 Logits-based、Feature-based 等蒸馏方法。

### 3.5 模型评估

#### 模型效果评估

- **重点内容**：
  - 了解效果评估体系，以及 few-shot、zero-shot 等基本概念。
  - 了解性能指标（Pass@、BLEU 等）、文本模型和视觉模型评估原理。
  - 了解常用评估框架，如 lm-eval、OpenCompass、VLMEvalKit。
- **参考资料**：[LLM 模型能力评估方法介绍](https://bytedance.larkoffice.com/wiki/ZRltwSlCviE0rXkRGUTc9bMdnzb)

#### 性能评估

- **重点内容**：理解吞吐、延迟、TTFT、TPOT 等常见指标，并将其与 LLM 推理的 Prefill、Decode 过程联系起来。
- **参考资料**：[DistServe：分离式 LLM 推理服务](https://hao-ai-lab.github.io/blogs/distserve/)

## 4. LLM 生态与应用

### 4.1 Agent

- **重点内容**：
  - 了解基本概念、ReAct 等基本方法。
  - 理解 tool use / function calling 原理和 memory 模块。
  - 了解 LangChain、Coze 等 Agent 框架。

### 4.2 RAG

- **重点内容**：理解知识库基本结构，以及数据处理、Embedding、向量数据库、检索技术、重排序、上下文整合等核心概念。

### 4.3 Compute Use 与 Browser Use

- **主题**：Compute use、Browser use。

### 4.4 协议

#### MCP、A2A 协议

- **参考资料**：[MCP 实践：LangGraph + MCP Adapter + MCP Servers](https://bytedance.larkoffice.com/docx/Q7cMdwGfjowgaFxSQjRcX0XWnyb)

#### OpenAI API 接口

- **重点内容**：理解 OpenAI Completions / Chat Completions API 格式。
- **参考资料**：[2023-08-10 OpenAI API 详解](https://xuqiwei1986.feishu.cn/wiki/K6mJw9P88iSFtAkIHOWcOL2tnlb)

## 5. AI 框架与软件栈

### 5.1 基本内容

#### 了解整体软件栈生态结构

- **重点内容**：从底层硬件、加速库（cuDNN、cuBLAS）、深度学习框架、ONNX，到数据集、模型训练框架、多机训练框架（Horovod、Ray）、多机通信库（*CCL），再到 MLOps，对 AI 整体软件生态进行了解性学习。

### 5.2 PyTorch 框架

#### PyTorch 框架基本使用

- **重点内容**：掌握基本使用，理解典型训练步骤：前向传播 → 计算损失 → 反向传播（计算梯度）→ 更新权重（优化器步骤）→ 清零梯度。

#### Torch 框架原理

- **重点内容**：理解计算图、自动微分原理。
- **可选内容**：完成一遍 MiniTorch 项目，自己实现一个小型 Torch。
- **参考资料**：[MiniTorch](https://minitorch.github.io/)

### 5.3 训练框架

#### 训练框架

- **重点内容**：了解大规模并行训练框架；小规模训练直接使用单机 PyTorch DDP 即可。
- **可选内容**：从 DeepSpeed、Colossal-AI、Megatron、Horovod 中选择一种学习。

#### LLM 后训练框架

- **重点内容**：了解 PEFT、OpenRLHF 以及 veRL。
- **参考资料**：[使用 veRL 进行 Qwen2.5-32B-Instruct GRPO 强化学习训练最佳实践](https://bytedance.larkoffice.com/wiki/A3JOwjYWViij4lkw2nycLXK1nMb)

### 5.4 推理框架

#### 通用模型推理框架

- **重点内容**：了解 MNN、TensorRT、MindSpore 等框架的使用和生态位。
- **可选内容**：深入分析其优化项，如 MNN 的图优化、移动端 kernel 级优化。

#### LLM 模型推理框架

- **重点内容**：了解 vLLM、SGLang 的使用。
- **可选内容**：分析一个框架的结构和代码流程，学习并行计算、缓存管理、算子优化、投机解码等优化项的实现。

### 5.5 Kubernetes

- **重点内容**：了解 Kubeflow、AIBrix。

## 6. 硬件层

### 6.1 计算架构

#### GPGPU、NPU、TPU

- **重点内容**：了解 GPGPU 的基本结构，包括 SM、CUDA Core、Tensor Core、内存层次结构等；了解 NPU、TPU。
- **参考资料**：
  - GPU 相关分享。
  - 《通用图形处理器设计》：[豆瓣页面](https://book.douban.com/subject/35998320/)

### 6.2 通信互联

#### 机内互联

- **重点内容**：PCIe、NVLink、NVSwitch、CXL。

#### 机间互联

- **重点内容**：
  - 了解 RoCE / IB 基本原理，诊断、监控与运维工具。
  - 掌握问题排查、带宽与延迟测试。
  - 了解 GPUDirect RDMA。

## 7. 训练与推理加速

### 7.1 CUDA

#### CUDA 编程模型

- **重点内容**：理解 CUDA 编程中的 thread、block、grid 基本概念。
- **可选内容**：手写一个 CUDA kernel，并在 Torch 中使用。
- **参考资料**：
  - [GPU MODE](https://github.com/gpu-mode)
  - 《Programming Massively Parallel Processors》（PMPP）
  - [Lecture 1: How to profile CUDA kernels in PyTorch](https://www.bilibili.com/video/BV1ry411i7nV?p=13)

#### CUDA 内存模型

- **重点内容**：理解 Registers、Local Memory、Shared Memory、Global Memory 等基本概念，以及 CUDA 内存管理。
- **参考资料**：同上。

#### 执行模型与同步

- **重点内容**：理解 Warp、`__syncthreads`、`cudaStreamSynchronize`，以及 Warp 的调度、执行、分支发散等问题。
- **参考资料**：同上。

#### 流与事件

- **重点内容**：了解 Stream、Event 的概念，以及 Stream 的同步与异步。
- **参考资料**：同上。

#### CUDA 优化技术

- **重点内容**：CUDA Graphs、利用片上内存进行内存优化、Kernel 融合。

### 7.2 Triton

#### Triton 编程模型基础

- **重点内容**：了解 JIT、启动配置、内存操作等。
- **参考资料**：[Triton 官方文档](https://triton-lang.org/main/index.html)

#### `triton.language`

- **重点内容**：掌握 Triton 的基本使用。

### 7.3 性能分析

#### 性能分析工具

- **重点内容**：掌握 NVIDIA Nsight Systems / Compute 的使用和性能数据分析。

### 7.4 Kernel、图优化与编译优化

#### 计算图优化与算子融合

- **重点内容**：算子融合、布局转换等。

#### 算子调优

- **重点内容**：
  - 底层结合具体硬件，从访存、指令集等角度优化。
  - 上层对具体算子使用优化算法，如 Winograd、FFT-based convolution 等。

#### 深度学习编译器

- **主题**：TVM。

#### `torch.compile`

- **主题**：了解并使用 `torch.compile`。

### 7.5 显存空间优化

#### ZeRO

- **重点内容**：理解 ZeRO 1/2/3 分别优化了哪些内存。

#### 混合精度训练

- **主题**：混合精度训练。

#### 模型压缩（量化、剪枝、蒸馏）

- **重点内容**：理解量化的基本概念和流程、量化模型如何推理，以及量化算子优化。

### 7.6 并行计算

- **重点内容**：
  - 理解数据并行、张量并行、流水线并行、专家并行、序列并行（可选）的基本原理。
- **可选内容**：手写一个张量并行的 Transformer Block。
- **参考资料**：
  - [李沐论文精读](https://github.com/mli/paper-reading/)中包含参数服务器、GPipe、Megatron-LM、ZeRO 等论文精读。
  - 深度学习分布式训练简介。
  - [PipeDream](https://arxiv.org/pdf/1811.06965)
  - [Megatron-LM](https://arxiv.org/pdf/1909.08053)

## 8. 通信优化

### 8.1 通信原语

- **重点内容**：
  - 理解 All-Reduce、All-Gather、Broadcast、Reduce、Scatter 等通信原语。
  - 了解这些原语的使用场景，例如 Row-based 张量并行 Linear 层使用了什么原语。
- **可选内容**：使用 `torch.distributed` 中的包进行实践；了解 vLLM、SGLang 对不同硬件通信层的封装。
- **参考资料**：
  - [PyTorch 集合通信原语图解](https://zhuanlan.zhihu.com/p/493092647)
  - [通信原语相关资料](https://zhuanlan.zhihu.com/p/28179871989)

### 8.2 NCCL

- **重点内容**：了解 NCCL 的使用方式、拓扑感知、多机通信测试方式，并使用 nccl-tests 测试。
- **参考资料**：
  - [一文讲清 NCCL 集合通信原理与优化](https://zhuanlan.zhihu.com/p/720502061)
  - [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)

### 8.3 并行策略与通信

- **重点内容**：结合并行计算策略，理解不同策略使用的通信原语并分析通信量。

### 8.4 Post 计算通信 Overlap

- **重点内容**：参考 DeepSeek-V3 训练过程的 two-batch overlap 方法。

## 9. LLM 推理加速

### 9.1 前置知识

- **重点内容**：理解基本 LLM 推理过程，以及向量化、Multi-Head / 其他 Attention、自回归解码等基本概念。
- **参考资料**：[LLM 推理综述](https://arxiv.org/html/2312.15234v1)

### 9.2 计算相关

- **重点内容**：
  - FlashAttention、FlashInfra：通过减少 IO 加速 Attention 计算。
  - Chunk Prefill。
  - 投机解码：
    - 基本概念与流程。
    - MTP。
    - EAGLE。
    - Lookahead。
- **参考资料**：
  - [从 FlashAttention 到 PagedAttention，如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)
  - [FlashAttention 与 PagedAttention](https://zhuanlan.zhihu.com/p/642962397)
  - [DeepSeek 技术解读：MTP 的前世今生](https://zhuanlan.zhihu.com/p/18056041194)
  - [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)

### 9.3 通信相关

- **重点内容**：
  - 计算/通信 overlap。
  - DeepEP。

### 9.4 资源与调度相关

- **重点内容**：
  - Continuous batching。
  - PD 分离：
    - Prefill 与 Decode 的区别。
    - PD 分离理论基础。
    - Dynamo、Mooncake 等开源 PD 分离框架。
    - PD 分离中的调度问题。
    - PD 分离中的 KV Cache 传输问题。
- **参考资料**：
  - [Continuous batching](https://zhuanlan.zhihu.com/p/688551989)
  - [Anyscale：Continuous batching for LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  - [PD 分离相关资料](https://zhuanlan.zhihu.com/p/706218732)

### 9.5 缓存相关

- **重点内容**：
  - KV Cache。
  - PagedAttention。
  - 前缀缓存。
  - KV 缓存量化。
  - 缓存的 Offload 与管理。
- **参考资料**：
  - [LLM Prefix Caching 技术调研](https://bytedance.larkoffice.com/wiki/NdPwwtwFYiMv74k4WBrcjyrqnNg)
  - [vLLM：简单、高效、易用的大模型推理框架](https://www.bilibili.com/video/BV1AdUNYDE8k/)

