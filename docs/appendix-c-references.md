# 附录 C 参考资料

> 建议阅读顺序：先读两篇核心论文（PagedAttention、SGLang），再看官方文档与源码，最后按兴趣深入 FlashAttention、PD 分离、投机解码等专题。链接以写作时为准，若失效可用标题搜索。

---

## C.1 核心论文

| 论文 | 出处 | 为什么读 |
|---|---|---|
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | SOSP 2023 · [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | vLLM 的奠基论文：分页 KV、块表、COW、continuous batching 的系统化。 |
| **SGLang: Efficient Execution of Structured Language Model Programs** | NeurIPS 2024 · [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | RadixAttention、前端语言、结构化程序执行。 |
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** | OSDI 2022 · [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level scheduling / continuous batching 的源头。 |
| **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | Chunked prefill 与 stall-free batching 的系统分析。 |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** | [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) | PD 分离的动机、建模与 goodput 分析。 |
| **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) | KV Cache 为中心的分离式架构，工业级案例。 |
| **Llumnix: Dynamic Scheduling for Large Language Model Serving** | [arXiv:2406.03243](https://arxiv.org/abs/2406.03243) | 多实例动态调度、尾延迟与公平性。 |
| **DeepSeek-V3 Technical Report** | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | MLA、MoE、MTP 与大规模训练/推理实践。 |
| **DeepSeek-R1** | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) | 推理模型与 RL，理解 rollout 负载的背景。 |

---

## C.2 注意力与 kernel

| 论文/项目 | 链接 | 主题 |
|---|---|---|
| **FlashAttention** | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | Tiling + 在线 softmax，减少 HBM 读写。 |
| **FlashAttention-2** | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | 并行度与工作划分优化。 |
| **FlashAttention-3** | [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) | Hopper 上的异步/低精度优化。 |
| **FlashInfer** | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) · [docs](https://docs.flashinfer.ai/) | 推理专用 attention/kernel 库，paged/ragged KV。 |
| **Triton** | [triton-lang.org](https://triton-lang.org/) | GPU kernel DSL，易读易改。 |
| **CUTLASS** | [github](https://github.com/NVIDIA/cutlass) | 高性能 GEMM/注意力模板。 |

---

## C.3 投机解码

| 论文/方法 | 链接 | 主题 |
|---|---|---|
| **Fast Inference from Transformers via Speculative Decoding** | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | 投机解码的经典 formulation。 |
| **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) | 基于特征的草稿与树验证。 |
| **EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** | [arXiv:2406.16858](https://arxiv.org/abs/2406.16858) | 动态草稿树。 |
| **Medusa** | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) | 多解码头并行预测。 |

> EAGLE-3、MTP 等新工作更新较快，建议直接看两家引擎的文档与代码实现。

---

## C.4 官方文档与代码

| 资源 | 链接 | 说明 |
|---|---|---|
| **vLLM 文档** | [docs.vllm.ai](https://docs.vllm.ai/) | 架构、配置、模型支持、benchmark。 |
| **vLLM GitHub** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 源码、issue、release notes。 |
| **vLLM Blog** | [blog.vllm.ai](https://blog.vllm.ai/) | 版本特性、性能分析。 |
| **SGLang 文档** | [docs.sglang.io](https://docs.sglang.io/) | 安装、后端、PD 分离、HiCache。 |
| **SGLang GitHub** | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) | 源码、示例、benchmark 脚本。 |
| **LMSYS Blog** | [lmsys.org/blog](https://lmsys.org/blog/) | RadixAttention、SGLang 版本、系统优化。 |
| **PyTorch 文档** | [pytorch.org/docs](https://pytorch.org/docs/) | `torch.compile`、CUDA Graph、profiler。 |
| **NVIDIA CUDA Graph** | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) | 图捕获/重放的官方说明。 |
| **NVIDIA Nsight Systems** | [docs](https://docs.nvidia.com/nsight-systems/) | GPU 时间线与通信分析。 |
| **py-spy** | [github](https://github.com/benfred/py-spy) | Python CPU 采样 profiler。 |

---

## C.5 评测与工具

| 工具 | 链接 | 用途 |
|---|---|---|
| **vLLM bench** | `vllm bench --help` | serve/throughput/latency 压测。 |
| **SGLang bench** | `python -m sglang.benchmark.serving --help`（另有 `offline_throughput`、`one_batch`；旧版 `sglang.bench_*`） | 在线/离线压测。 |
| **NVIDIA genai-perf** | [docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html) | OpenAI 兼容端点压测。 |
| **llmperf** | [github](https://github.com/ray-project/llmperf) | Ray 生态压测。 |
| **locust** | [locust.io](https://locust.io/) | 通用 HTTP 负载测试。 |
| **k6** | [k6.io](https://k6.io/) | 通用负载测试。 |
| **Prometheus / Grafana** | [prometheus.io](https://prometheus.io/) | 指标采集与可视化。 |

---

## C.6 背景与系统课程

| 资源 | 主题 |
|---|---|
| **Computer Systems: A Programmer's Perspective (CSAPP)** | 虚拟内存、分页、缓存、并发；理解 PagedAttention 的最佳背景。 |
| **Programming Massively Parallel Processors (Kirk & Hwu)** | GPU 体系结构、GEMM、roofline、kernel 优化。 |
| **MIT 6.824 / 6.5840 Distributed Systems** | 分布式系统基础，理解 TP/PD/容错。 |
| **Stanford CS336: Language Modeling from Scratch** | 从零实现语言模型，理解训练/推理细节。 |
| **vLLM / SGLang 的 release notes** | 了解真实系统的演进与权衡。 |

---

## C.7 专题阅读路线

### 路线 1：KV 管理

1. PagedAttention 论文（第 4～5 节）；
2. vLLM `kv_cache_manager.py` / `block_pool.py`；
3. SGLang 论文的 RadixAttention 部分；
4. SGLang `radix_cache.py`；
5. 自己实现一个迷你模拟器（附录 B 项目）。

### 路线 2：调度

1. Orca 论文；
2. Sarathi-Serve（chunked prefill）；
3. vLLM `scheduler.py`；
4. SGLang `scheduler.py` + `schedule_policy.py`；
5. Llumnix（多实例调度/公平性）。

### 路线 3：PD 分离与大规模部署

1. DistServe；
2. Mooncake；
3. 两家文档的 disaggregation 章节；
4. DeepSeek-V3 技术报告的推理系统部分；
5. 自己画一个 prefill/decode 资源模型。

### 路线 4：kernel 与编译

1. FlashAttention 1/2/3；
2. FlashInfer 论文/文档；
3. Triton 教程；
4. CUDA Graph / torch.compile；
5. 用 nsys 分析一个 decode step。

---

## C.8 如何读论文（给学生的建议）

1. **先读摘要、引言、系统图**：搞清楚它解决什么问题、核心洞察是什么。
2. **带着三个问题读**：瓶颈在哪？关键数据结构/算法是什么？代价是什么？
3. **对照源码**：论文里的“block table”“radix tree”在代码里长什么样？
4. **复现一个实验**：哪怕缩小规模，也能发现论文没写的细节。
5. **找反例**：这个优化在什么 workload 下会失效？
6. **写一页总结**：问题、方法、假设、结果、局限、与另一篇的对比。

---

## C.9 版本与引用说明

- **事实核对日期：2026-11-25。** 讲义基线为 **vLLM 0.34.0**（2026-11-16 发布）与 **SGLang 0.5.23**（2026-10-30 发布）；涉及 0.8/0.4.0 等旧版本行为时会单独标注。
- 版本与默认值的官方来源：[vLLM v0.34.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)、[vLLM Engine Arguments](https://docs.vllm.ai/en/v0.34.0/configuration/engine_args.html)；[SGLang v0.5.23 Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)、[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)。
- vLLM 的 `block_size` 自 0.34.0 起默认 64（此前长期为 16，`DEFAULT_BLOCK_SIZE`）；SGLang 的 `page_size` 官方默认值为 1。
- 讲义中的命令、参数默认值与 benchmark 数字都以“写作时公开文档 + 对应 tag 的源码”为准，不构成跨版本承诺。
- 具体命令行参数、默认值、支持模型/量化格式会随版本变化；实践时请锁定版本并查阅对应文档。
- 文中性能结论均为定性/教学用途，不构成对任何版本的性能承诺。
- 引用论文时，请以出版版本/最新 arXiv 版本为准。

---

> **最后的话**：推理系统的知识一半在论文里，一半在代码和压测里。读懂 PagedAttention 与 RadixAttention 只能让你“知道”，亲手跑一次对照实验、读一次调度器、画一次 KV 账本，才能让你“会”。祝学习顺利。

---

> **返回课程首页**：[课程首页](index.md)
