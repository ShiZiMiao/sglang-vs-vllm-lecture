# 附录 A 术语表

> 按主题分组，便于查阅。英文缩写首次出现时给出全称。

---

## A.1 生成与模型

| 术语 | 英文/全称 | 解释 |
|---|---|---|
| 自回归生成 | Autoregressive Generation | 每次根据已生成的 token 预测下一个 token，循环直到停止。 |
| Prefill | Prefill / Prompt Processing | 一次性并行处理整个 prompt，计算 KV Cache 并产出第一个 token。 |
| Decode | Decode / Generation | 逐 token 生成阶段，每步处理一个新 token，读取历史 KV。 |
| KV Cache | Key-Value Cache | 缓存历史 token 的 Key/Value，避免每步重算 K/V 投影，把 K/V 投影复杂度从 $O(T^2)$ 降到 $O(T)$；注意力本身仍与 $T^2$ 相关。 |
| GQA | Grouped Query Attention | 多个 Query 头共享一组 KV 头，减少 KV Cache 与带宽。 |
| MQA | Multi-Query Attention | 所有 Query 头共享一组 KV 头，压缩更激进，精度可能受影响。 |
| MLA | Multi-head Latent Attention | DeepSeek 提出的低秩压缩注意力，把 KV 压成 latent 向量，大幅降低 KV 显存。 |
| MoE | Mixture of Experts | 每个 token 只激活部分专家，参数量大但计算量相对小。 |
| RoPE | Rotary Position Embedding | 旋转位置编码，主流 LLM 使用。 |
| 上下文窗口 | Context Window | 模型一次能处理的最大 token 数（prompt + output）。 |
| Logits | Logits | 模型输出层未归一化的分数，经 softmax 得到 token 概率。 |
| 采样 | Sampling | 从概率分布选 token；温度、top-k、top-p 等控制随机性。 |
| 温度 | Temperature | 缩放 logits；越低越确定，越高越随机。 |
| Top-k / Top-p | Top-k / Nucleus Sampling | 只从概率最高的 k 个 / 累计概率 p 的 token 中采样。 |
| Beam Search | Beam Search | 维护多个候选序列，适合确定性任务，显存开销大。 |
| 投机解码 | Speculative Decoding | 小模型/草稿头先猜多个 token，大模型一次验证，提升 decode 速度。 |
| 草稿模型 | Draft Model | 投机解码中负责快速起草的小模型。 |
| 接受率 | Acceptance Rate | 草稿 token 被目标模型接受的比例，决定投机加速比。 |
| MTP | Multi-Token Prediction | 一次预测多个 token 的训练/推理头，DeepSeek 等模型使用。 |
| EAGLE | EAGLE | 一类基于特征/树的投机解码方法，EAGLE-2/3 持续改进。 |

---

## A.2 性能指标

| 术语 | 英文/全称 | 解释 |
|---|---|---|
| TTFT | Time To First Token | 请求到第一个输出 token 的时间，主要受排队与 prefill 影响。 |
| TPOT | Time Per Output Token | 平均每输出 token 时间：（E2E − TTFT）/（输出 token 数 − 1）。 |
| ITL | Inter-Token Latency | 相邻 token 的时间间隔；关注 P99 抖动。 |
| E2E | End-to-End Latency | 请求到最后一个 token 的总时间。 |
| 吞吐 | Throughput | 单位时间处理的 token/请求数；需注明口径与硬件。 |
| Goodput | Goodput | 满足 SLO 的有效吞吐。 |
| SLO | Service Level Objective | 服务水平目标，如 TTFT P99 < 1s。 |
| P50/P90/P99 | Percentile | 分位数；P99 表示 99% 的请求不超过该值。 |
| 算术强度 | Arithmetic Intensity | FLOPs / Bytes；判断 kernel 受算力还是带宽限制。 |
| FLOPs | Floating Point Operations | 浮点运算次数；本讲义按“一次乘法 + 一次加法 = 2 FLOPs”计。 |
| FMA | Fused Multiply-Add | 乘加融合指令，一条硬件指令完成 `a*b+c`；按 FLOPs 计通常算 2 次运算。 |
| Roofline | Roofline Model | 用峰值算力与带宽刻画 kernel 性能上界的模型。 |
| HBM | High Bandwidth Memory | GPU 高带宽显存；decode 主要受其带宽限制。 |
| MFU | Model FLOPs Utilization | 实际算力 / 峰值算力，衡量计算效率。 |
| 有效吞吐 | Effective Throughput | 扣除失败、超时、重算后的吞吐。 |

---

## A.3 调度与内存管理

| 术语 | 英文/全称 | 解释 |
|---|---|---|
| 连续批处理 | Continuous Batching | 每个 iteration 重新组批，动态加入/移除请求。 |
| 迭代级调度 | Iteration-level Scheduling | 以一次 decode step 为调度粒度，Orca 提出。 |
| 静态批处理 | Static Batching | 一批请求固定执行到全部结束，存在 padding/尾部浪费。 |
| Chunked Prefill | Chunked Prefill | 把长 prompt 切成多个 chunk 分步 prefill，控制激活峰值。 |
| 抢占 | Preemption | 显存不足时把运行中请求踢出，释放 KV。 |
| 重算 | Recompute | 抢占后丢弃 KV，恢复时重新 prefill。 |
| 换出 | Swap | 把 KV 换到 CPU 内存，恢复时换入。 |
| PagedAttention | PagedAttention | 把 KV 切成固定块，用块表做逻辑→物理映射。 |
| 块表 | Block Table | 每条序列的逻辑块到物理块的映射表。 |
| 逻辑块/物理块 | Logical/Physical Block | 逻辑上连续的块号 vs 显存中的实际块号。 |
| 写时复制 | Copy-on-Write (COW) | 共享块被写入时才复制，支持分支/并行采样。 |
| 前缀缓存 | Prefix Caching | 复用相同前缀的 KV，避免重复 prefill。 |
| 基数树 | Radix Tree | 压缩前缀树；SGLang 用其组织 KV 前缀。 |
| 节点分裂 | Node Split | 部分匹配时把一个树节点拆成两段。 |
| LRU | Least Recently Used | 最近最少使用淘汰策略。 |
| 引用计数 | Reference Count | 记录多少请求/节点引用某段 KV，防止被误释放。 |
| 驱逐 | Eviction | 缓存不足时释放某些 KV。 |
| KV 池 | KV Pool | 预分配的 KV 存储空间。 |
| Token 池 | Token Pool | 按 token 槽位管理的 KV 池（SGLang）。 |
| 页大小 | Page Size | 每个块/页包含的 token 数；vLLM 长期默认 16，0.34.0 起默认 64；SGLang 默认 1。 |
| 内部碎片 | Internal Fragmentation | 分配单元内未使用的空间（如最后一块未满）。 |
| 外部碎片 | External Fragmentation | 空闲空间总量够但分散，无法满足连续分配。 |
| 缓存命中率 | Cache Hit Rate | 命中缓存的 token/请求比例。 |
| Cache Salt | Cache Salt | 注入前缀哈希的盐值，用于租户隔离/防侧信道。 |
| LPM | Longest Prefix Match | 最长前缀匹配调度，优先高命中请求；SGLang 的可选策略，需显式开启。 |
| FCFS | First Come First Served | 先来先服务。 |
| DFS-weight | DFS Weight | 沿基数树计算权重、优先热节点的缓存感知调度策略；权重算法随版本变化。 |
| Token Budget | Token Budget | 单步允许处理的 token 总数上限。 |
| 准入控制 | Admission Control | 决定哪些请求进入运行队列。 |
| 公平性 | Fairness | 避免某些请求长期得不到资源。 |
| 饿死 | Starvation | 低优先级/低命中请求长期排队。 |
| Aging | Aging | 等待越久优先级越高，缓解饿死。 |

---

## A.4 执行与内核

| 术语 | 英文/全称 | 解释 |
|---|---|---|
| 注意力后端 | Attention Backend | 实现 attention 的 kernel 库/路径，如 FlashAttention、FlashInfer。 |
| FlashAttention | FlashAttention | 通过 tiling + 在线 softmax 减少 HBM 读写的注意力 kernel。 |
| FlashInfer | FlashInfer | 面向推理的高性能 attention/kernel 库，支持 paged/ragged KV。 |
| Triton | Triton | OpenAI 的 GPU kernel DSL，易写易改。 |
| CUDA Graph | CUDA Graph | 把一串 kernel 捕获成图，一次 launch 重放，降低 CPU 开销。 |
| torch.compile | torch.compile | PyTorch 的图编译，减少 Python/launch 开销。 |
| Piecewise Graph | Piecewise CUDA Graph | 在动态形状处切段，其余部分用图执行。 |
| Kernel Launch | Kernel Launch | CPU 向 GPU 提交 kernel 的开销。 |
| GEMM/GEMV | GEMM/GEMV | 矩阵乘矩阵/矩阵乘向量；prefill 多为 GEMM，decode 多为 GEMV。 |
| Varlen Attention | Variable-length Attention | 一个 batch 内序列长度不同，需 ragged 索引。 |
| Slot Mapping | Slot Mapping | token → KV 物理槽位的映射（vLLM）。 |
| Out Cache Loc | Out Cache Loc | 本步新 token 的 KV 写入位置（SGLang）。 |
| 采样器 | Sampler | 从 logits 采样 token 的模块。 |
| Logits Processor | Logits Processor | 在采样前修改 logits（惩罚、约束、mask）。 |
| FSM | Finite State Machine | 有限状态机；约束解码用其表示合法 token 序列。 |
| 语法编译 | Grammar Compilation | 把 JSON Schema/正则/EBNF 编译成 FSM。 |
| Jump-forward | Jump-forward Decoding | 对确定性 token 片段一次性填入，跳过逐步采样。 |
| 增量 Detokenize | Incremental Detokenization | 流式把 token 还原为文本，处理跨 token 字符。 |

---

## A.5 分布式与部署

| 术语 | 英文/全称 | 解释 |
|---|---|---|
| TP | Tensor Parallelism | 张量并行，按头/维度切分权重。 |
| PP | Pipeline Parallelism | 流水线并行，按层切分。 |
| DP | Data Parallelism | 数据并行，多副本处理不同请求。 |
| EP | Expert Parallelism | 专家并行，MoE 专家分布到不同 GPU。 |
| EPLB | Expert Parallel Load Balancer | 动态平衡专家负载的机制。 |
| SP/CP | Sequence/Context Parallelism | 序列/上下文并行，切分长序列。 |
| PD 分离 | Prefill-Decode Disaggregation | prefill 与 decode 部署到不同实例，KV 传输。 |
| KV 传输 | KV Transfer | PD 分离/分层缓存中搬运 KV 的过程。 |
| NIXL | NVIDIA Inference Xfer Library | 高性能点对点传输库，常用于 KV 传输。 |
| Mooncake | Mooncake | 面向 KVCache 的分离式架构/传输后端。 |
| HiCache | Hierarchical Cache | SGLang 的 GPU/CPU/磁盘分层 KV 缓存。 |
| 分层缓存 | Hierarchical Cache | 把 KV 放在不同存储层级，按需预取。 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问，PD 分离常用。 |
| NVLink | NVLink | NVIDIA GPU 间高速互联。 |
| All-reduce | All-reduce | 所有 rank 归约并广播，TP 每层使用。 |
| All-to-all | All-to-all | 所有 rank 互相发送数据，EP 路由使用。 |
| Bootstrap | Bootstrap | PD 分离中建立连接/交换元数据的引导过程。 |
| 路由器 | Router | 把请求分发到实例的组件。 |
| 缓存感知路由 | Cache-aware Routing | 按前缀命中情况路由请求。 |
| 灰度 | Canary | 小比例流量验证新版本。 |
| 多租户 | Multi-tenancy | 多个用户/组织共享集群，需要隔离与配额。 |
| LoRA | Low-Rank Adaptation | 低秩适配器，低成本微调/多租户定制。 |
| 量化 | Quantization | 降低权重/激活/KV 精度，减少显存与带宽。 |
| FP8 | 8-bit Floating Point | Hopper/Blackwell 支持的低精度格式。 |
| AWQ/GPTQ | Activation-aware Weight Quantization / GPTQ | 常见的 4-bit 权重量化方法。 |
| NVFP4 | NVIDIA FP4 | Blackwell 的 4-bit 浮点格式。 |
| OpenAI API | OpenAI-compatible API | 事实标准的 HTTP 接口，便于替换引擎。 |
| SSE | Server-Sent Events | 流式返回 token 的 HTTP 机制。 |
| Prometheus | Prometheus | 指标采集与监控系统。 |
| SLO 达成率 | SLO Attainment | 满足 SLO 的请求比例。 |

---

## A.6 快速缩写表

| 缩写 | 全称 |
|---|---|
| LLM | Large Language Model |
| KV | Key-Value |
| TTFT | Time To First Token |
| TPOT | Time Per Output Token |
| ITL | Inter-Token Latency |
| E2E | End-to-End |
| SLO | Service Level Objective |
| GQA | Grouped Query Attention |
| MQA | Multi-Query Attention |
| MLA | Multi-head Latent Attention |
| MoE | Mixture of Experts |
| MTP | Multi-Token Prediction |
| COW | Copy-on-Write |
| LRU | Least Recently Used |
| LPM | Longest Prefix Match |
| FCFS | First Come First Served |
| TP | Tensor Parallelism |
| PP | Pipeline Parallelism |
| DP | Data Parallelism |
| EP | Expert Parallelism |
| EPLB | Expert Parallel Load Balancer |
| PD | Prefill-Decode |
| FSM | Finite State Machine |
| GEMM | General Matrix Multiply |
| GEMV | General Matrix-Vector Multiply |
| HBM | High Bandwidth Memory |
| MFU | Model FLOPs Utilization |

---

> **继续阅读**：[附录 B 习题参考答案与实验评分标准 →](appendix-b-exercises.md)
