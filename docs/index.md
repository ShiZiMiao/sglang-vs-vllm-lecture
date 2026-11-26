# LLM 推理系统：SGLang 与 vLLM 对比讲义

> 面向人工智能专业本科高年级/研究生的系统课讲义  
> 从一次文本生成讲起，逐步建立 LLM 推理引擎的完整心智模型，再深入对比 SGLang 与 vLLM 的设计取舍。

---

## 这门课讲什么

大语言模型的“训练”决定了模型会不会思考，而“推理系统”决定了它能不能便宜、稳定、低延迟地把思考结果交付给用户。vLLM 与 SGLang 是当前开源社区最具代表性的两个 LLM 推理引擎：

- **vLLM**：以 PagedAttention 和通用、成熟的工程生态著称，模型覆盖面广、硬件后端多、生产部署经验丰富。
- **SGLang**：以 RadixAttention、可选的缓存感知调度和结构化程序执行著称，在多轮对话、Agent、RAG、结构化输出、大规模 MoE 部署等场景表现突出。

本讲义不满足于罗列“谁比谁快”的结论，而是希望你能回答三个层次的问题：

1. **原理层**：KV Cache 为什么存在？PagedAttention 与 RadixAttention 各自解决了什么问题？
2. **系统层**：一个请求从 HTTP 到达，到 token 流式返回，中间经过哪些模块？调度器每一步在做什么决策？
3. **工程层**：给定模型、硬件、流量形态和 SLO，应该如何选型、配置、压测和调优？

---

## 适合谁读

**目标读者**：人工智能、计算机科学相关专业的高年级本科生或研究生；也适合**没有系统学过 Transformer 和推理系统**、但会一点 Python 的同学。

- 会 Python 基础（变量、函数、循环）即可；
- 深度学习背景不是必需的：第 0 章会用“文字接龙”“答题时回看前文”等类比补齐 token、注意力、Q/K/V、GPU 显存等概念；
- 如果已经学过 Transformer，第 0 章可以快速浏览，把重点放在 KV Cache 与 GPU 瓶颈上；
- 计算机系统背景（虚拟内存、缓存、并发）会在第 1、2 章逐步建立，不需要提前掌握。

如果暂时不熟悉 GPU 体系结构也没关系：第 0、1、2 章会把必需的背景补齐。

---

## 学完你能做什么

- 用 roofline 模型解释为什么 prefill 是计算密集型、decode 是显存带宽密集型；
- 手算给定模型的权重显存、单 token KV Cache 大小和最大并发数；
- 说清 PagedAttention 的 block table、前缀哈希、copy-on-write 与 LRU 驱逐；
- 说清 RadixAttention 的基数树匹配、插入、分裂、引用计数与 LRU 驱逐；
- 对比两种引擎在 KV 管理、调度、前缀复用、执行后端上的差异；
- 独立启动 vLLM/SGLang 服务，用 OpenAI 兼容 API 调用并做可复现压测；
- 根据 workload 特征做出有理有据的选型，而不是背“某个引擎更快”的结论。

---

## 章节地图

### 开始之前

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 0 章 预备知识：从“文字接龙”到 GPU 服务](00-prerequisites.md) | token、注意力、GPU、延迟与吞吐 | 没学过 Transformer，也能读懂后面的内容吗？ |

### 第一部分：基础

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 1 章 从一次文本生成到推理服务](01-foundations.md) | 推理基础与性能指标 | LLM 推理为什么难？TTFT/TPOT 是什么？KV Cache 有多大？ |
| [第 2 章 推理引擎的通用架构](02-engine-architecture.md) | 调度器、KV 管理器、执行器 | 一个请求如何流过引擎？分页思想从何而来？ |

### 第二部分：两个引擎

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 3 章 vLLM：以 PagedAttention 为核心的通用推理引擎](03-vllm.md) | PagedAttention、V1 引擎 | 块表、块哈希前缀缓存、chunked prefill 如何工作？ |
| [第 4 章 SGLang：以 RadixAttention 与程序级优化为核心的推理栈](04-sglang.md) | RadixAttention、程序级优化 | 基数树、缓存感知调度（可选 LPM）、overlap scheduler 如何工作？ |

### 第三部分：对比与实践

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 5 章 核心机制正面对比：页表 vs 基数树，通用调度 vs 缓存感知](05-mechanism-comparison.md) | 页表 vs 基数树，通用调度 vs 缓存感知 | 差异何时真正转化为性能差异？ |
| [第 6 章 功能、模型与生态对比](06-features-ecosystem.md) | 量化、LoRA、多模态、PD 分离 | 谁的覆盖更广，谁的场景更深？ |
| [第 7 章 性能评测：如何科学地比较 vLLM 与 SGLang](07-benchmarking.md) | benchmark 的坑与公平比较 | 怎样不被“快 N 倍”的数字忽悠？ |
| [第 8 章 动手实践：启动、调用、压测与调参](08-hands-on.md) | 启动、调用、压测、调参 | 如何在你的机器上复现对照实验？ |

### 第四部分：深入与选型

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 9 章 源码地图：一次请求在两个引擎中的调用链](09-source-code-tour.md) | 一次请求的完整调用链 | 调度器、KV 管理、ModelRunner 在哪些文件？ |
| [第 10 章 选型、演进与前沿](10-selection-and-future.md) | 决策树、混合部署、开放问题 | 给定业务应该如何选？未来往哪走？ |

### 附录

- [附录 A 术语表](appendix-a-glossary.md)
- [附录 B 习题参考答案与实验评分标准](appendix-b-exercises.md)
- [附录 C 参考资料](appendix-c-references.md)

---

## 建议学习路径

- **两周速成**：第 0 → 1 → 2 → 3 → 4 → 5 章，配合第 8 章跑通一个服务（第 6、7 章选读）；
- **完整课程（约 32 学时）**：全部正文 + 附录 B 的实验；
- **工程选型**：第 0、1、5、6、7、8、10 章，重点做第 8 章的对照压测。

### 如果你是零基础

1. **第一遍不要抠公式**：先读第 0 章，完成自测；再读第 1 章的 1.1～1.5，知道“prefill 吃算力、decode 吃带宽、KV 很占显存”即可。
2. **第二遍建立结构**：读第 2 章 2.0～2.4，跟着“一条请求的 12 步”走一遍；不懂的组件名先跳过。
3. **第三遍再进细节**：读第 3、4 章的 3.0/4.0 类比和“手把手例子”，再回头看公式与源码。
4. **每章只记三件事**：一个核心问题、一个关键数据结构、一个代价。其余内容可以当手册查。

建议边读边做三件事：

- 在纸上画出调度器每一步的状态变化；
- 用第 1 章的公式估算你手头模型的显存账本；
- 每读完一章，回到第 5 章的对比表补充一行自己的理解。

---

## 实验环境

| 项目 | 最低建议 | 说明 |
|---|---|---|
| GPU | 1 × 24 GB（如 RTX 4090 / L4） | 可跑 7B/8B 的 FP16 或 4-bit 量化 |
| 内存 | 32 GB 以上 | 模型下载与 CPU 侧缓存 |
| 系统 | Linux 或 Windows + WSL2 | 两个引擎的官方支持均以 Linux 为主 |
| CUDA | 12.1 以上 | 以引擎版本要求为准 |
| Python | 3.10+ | 建议使用独立虚拟环境 |
| 磁盘 | 100 GB 空闲 | 模型权重与编译缓存 |

配套脚本位于仓库的 `code/` 目录：`bench_prefix.py` 用于共享前缀对照实验，使用说明见 `code/README.md`。

---

## 版本约定

推理引擎迭代极快。本讲义以 **vLLM 0.34.0** 与 **SGLang 0.5.23** 为讲解基线，事实核对日期为 **2026-11-25**。

> **讲义基准（事实核对日期：2026-11-25）**
>
> | 项目 | 基线版本 | 官方核对来源 | 说明 |
> |---|---|---|---|
> | vLLM | 0.34.0（2026-11-16 发布） | [GitHub Release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)、[PyPI](https://pypi.org/project/vllm/) | V1 引擎自 0.8 起为默认；`block_size` 自 0.34.0 起默认 64 |
> | SGLang | 0.5.23（2026-10-30 发布） | [GitHub Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)、[PyPI](https://pypi.org/project/sglang/) | 0.4.6 起默认调度为 `fcfs`；0.5.x 的参数定义在 `sglang/srt/arg_groups/` |
> | 参数默认值 | 对应版本的 `--help` 与启动日志 | [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.34.0/configuration/engine_args.html)、[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) | 本文不承诺跨版本逐字兼容 |
>
> 教学示例中仍常用 `block_size=16`，便于与 PagedAttention 论文和历史资料对照；但 0.34.0 的实际默认值是 64，启动时以日志或 `--block-size` 为准。SGLang 的 `page_size` 官方默认值为 1，因此 Radix Cache 的共享粒度可以做到 token 级。

在这个基线下：

- **稳定的部分**：PagedAttention、RadixAttention、continuous batching、chunked prefill、CUDA Graph、张量并行等原理不会轻易改变；
- **易变的部分**：具体命令行参数、默认值、支持的模型与量化格式、内部类名和文件路径；
- **实践原则**：运行前以你安装版本的官方文档和 `--help` 为准。

任何“A 比 B 快 N 倍”的说法都高度依赖模型、硬件、流量形态、版本与调参。第 7 章会专门讲如何做公平评测。**本讲义的目标是教你判断方法，而不是让你背结论。**

---

## 开始学习

> [开始第 0 章：预备知识 →](00-prerequisites.md)
