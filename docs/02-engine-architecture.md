# 第 2 章 推理引擎的通用架构

> **本章目标**：在进入 vLLM 与 SGLang 的细节之前，先建立一套“引擎无关”的通用架构。之后你会发现，两个引擎的差异主要是同一组模块里的数据结构与策略不同。
>
> **先修知识**：[第 0 章](00-prerequisites.md) 的 token/GPU 直觉、[第 1 章](01-foundations.md) 的 prefill/decode 与 KV Cache。忘了也没关系，本章会用“考场/教务系统”类比重新串一遍。
>
> **一句话概括**：推理引擎 = 一个专门服务 LLM 请求的“考场”：接卷窗口收下卷子、监考排卷、草稿纸管理员管 KV、考生大脑做前向、誊写员流式交卷。
>
> **给零基础同学**：本章会出现很多组件名，不要试图一次记住。先记住“一份卷子从进考场到交卷经过哪些人”，再看每个人具体做什么。

---

## 2.0 先用考场/教务系统类比理解整个引擎

推理引擎通常由七类组件组成：API/Frontend、Processor/Tokenizer、Scheduler、KV Cache Manager、Executor/Worker、Sampler、Output Processor。把推理引擎想成一个考场/教务系统（图 2-1）：

```text
卷子（用户请求）
   │
   ▼
接卷窗口（API/Frontend）── 收下卷子、验身份、交给监考
   │
   ▼
监考/排卷（Scheduler）──── 决定先答哪份、每份答多少、什么时候加卷
   │
   ├──▶ 草稿纸管理员（KV Cache Manager）── 记录每份卷子的笔记放哪、哪些笔记可共享
   │
   ▼
考生大脑（Executor/Worker/ModelRunner）── 真正作答（GPU 前向）
   │
   ▼
誊写/收卷员（Output Processor/Detokenizer）── 把答案整理好、流式交卷
```

*图 2-1　推理引擎的考场类比*

*表 2-1　考场类比与引擎组件*

| 考场角色/环节 | 引擎组件 | 一句话职责 |
|---|---|---|
| 接卷窗口 | API / Frontend | 收下卷子、验身份、解析参数、管理连接 |
| 题目录入 | Processor / Tokenizer | 文本 → token，套答题模板 |
| 监考 / 排卷 | Scheduler | 每一步决定哪份答、哪份停、答多少 |
| 草稿纸管理员 | KV Cache Manager | 分配/共享/回收草稿纸 |
| 考生大脑 | Executor / Worker / ModelRunner | 在 GPU 上执行模型前向 |
| 落笔 | Sampler / Logits Processor | 从概率里选下一个 token |
| 誊写 / 收卷员 | Output Processor / Detokenizer | 判断停止、还原文本、流式交卷 |

这张表就是本章的“地图”。下面逐个展开。

---

## 2.1 三个核心抽象

几乎所有现代 LLM 推理引擎都可以用三个抽象来描述：

### 2.1.1 Request（请求）

一次 API 调用对应的逻辑对象，包含：

- `prompt`：输入 token（或文本、多模态输入）；
- `sampling_params`：温度、top_p、top_k、max_tokens、停止条件；
- `arrival_time`：到达时间，用于计算排队延迟；
- `priority` / `slo`：优先级或延迟目标；
- `request_id`：用于流式返回与取消。

Request 是“用户视角”的对象，生命周期跨越整个生成过程。

### 2.1.2 Sequence（序列）

引擎内部对一条生成中序列的建模，通常包含：

- 已处理的 token ID 列表（prompt + 已生成）；
- 已分配的 KV Cache 位置（块表、token 索引或树节点）；
- 当前状态（waiting / running / finished）；
- 已生成 token 数、停止原因；
- 采样状态（随机数生成器、logits processor 状态）。

一条 Request 通常对应一条 Sequence；但在并行采样（`n>1`）、beam search、投机解码等场景下，一个 Request 可能派生多条 Sequence。

### 2.1.3 Batch（批）

调度器在一次迭代（iteration / step）中决定一起送进 GPU 的序列集合。Batch 不是固定的：

- 每一步都可能加入新序列、移除完成序列；
- 每条序列本步要处理的 token 数可能不同（chunked prefill 下尤其明显）；
- Batch 的总 token 数受 `max_num_batched_tokens` 约束；
- Batch 的序列数受 `max_num_seqs` 约束。

**关键洞察**：continuous batching 的本质，是把“Batch”从“一次组批、整批执行”变成“每一步重新组批”。这也迫使 KV Cache 管理必须支持动态增删。

---

## 2.2 请求的生命周期

一条请求在引擎中的典型状态迁移如图 2-2 所示：

```text
                 ┌──────────────────────────────────────────┐
                 │                                          │
   到达           ▼           被调度                         │
 ───────▶ [WAITING] ──────────────────▶ [RUNNING: PREFILL]  │
              ▲                              │              │
              │                              │ 首 token 产出 │
              │                              ▼              │
              │                        [RUNNING: DECODE]    │
              │                              │              │
              │        被抢占（显存不足）      │ 停止条件满足  │
              └──────────────────────────────┘              │
                                                             ▼
                                                        [FINISHED]
```

*图 2-2　请求的生命周期状态机*

### 2.2.0 把状态图翻译成大白话：一条请求的 12 步

| 步骤 | 发生什么 | 负责组件 | 主要硬件 |
|---|---|---|---|
| 1 | 用户发来 HTTP 请求 | API/Frontend | CPU |
| 2 | 鉴权、限流、解析参数 | API/Frontend | CPU |
| 3 | 套 chat template，文本切成 token | Processor/Tokenizer | CPU |
| 4 | 创建 Request/Sequence，放进 waiting 队列 | 引擎核心 | CPU |
| 5 | 调度器决定“这条请求能不能现在跑” | Scheduler | CPU |
| 6 | 查前缀缓存，分配 KV 块/槽位 | KV Cache Manager | CPU/GPU 元数据 |
| 7 | 把本步要跑的序列组成 Batch | Scheduler | CPU |
| 8 | 把逻辑 Batch 转成 GPU 张量 | ModelRunner | CPU + GPU |
| 9 | 执行 prefill，算出第一个 token | Model/Attention Backend | GPU |
| 10 | 采样得到第一个 token | Sampler | GPU |
| 11 | detokenize，流式返回第一个字 | Output Processor | CPU |
| 12 | 循环 decode，直到停止，释放 KV | Scheduler + Output | GPU + CPU |

> **零基础提示**：不是所有引擎都严格按这 12 步分，但“接单 → 排队 → 分配 KV → 组批 → 前向 → 采样 → 输出 → 回收”的主干是一样的。读源码时，你可以拿这张表逐项对照。

几个容易混淆的点：

- **Prefill 与 Decode 不是两个独立状态机**，而是 RUNNING 下的两个阶段。chunked prefill 会让一条序列在多个 step 中分多次完成 prefill。
- **抢占（preemption）**：当显存不足时，调度器可能把某些运行中的序列踢回 waiting。恢复方式有两种：
  - **Recompute**：丢掉 KV，重新 prefill（实现简单，浪费算力）；
  - **Swap**：把 KV 换出到 CPU 内存，恢复时再换入（省算力，占 PCIe 带宽）。
- **Abort**：客户端断开或超时，引擎需要尽快释放资源，避免“僵尸序列”。

---

## 2.3 引擎的七大组件

把任意一个推理引擎拆开，通常都能找到下面这些组件（图 2-3）。不同引擎可能合并或拆分它们，但职责基本一致。

```text
┌────────────────────────────────────────────────────────────────────┐
│                          API / Frontend                            │
│  HTTP、OpenAI 协议、鉴权、限流、请求校验、流式响应                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│                     Processor / Tokenizer                          │
│  文本→token、chat template、多模态预处理、停止条件解析               │
└───────────────────────────────┬────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│                         Scheduler                                  │
│  准入、组批、prefill/decode 混合、抢占、优先级、token budget         │
└───────────────┬────────────────────────────────────┬───────────────┘
                │                                    │
┌───────────────▼──────────────┐      ┌──────────────▼───────────────┐
│      KV Cache Manager        │      │        Executor              │
│  分配、共享、前缀匹配、驱逐    │      │  Worker / ModelRunner / TP   │
└───────────────┬──────────────┘      └──────────────┬───────────────┘
                │                                    │
                │                      ┌─────────────▼───────────────┐
                │                      │   Model + Attention Backend │
                │                      │  GEMM、Attention、MoE、采样   │
                │                      └─────────────┬───────────────┘
                │                                    │
┌───────────────▼────────────────────────────────────▼───────────────┐
│                    Output Processor / Detokenizer                  │
│  停止判断、增量 detokenize、流式返回、用量统计、指标上报             │
└────────────────────────────────────────────────────────────────────┘
```

*图 2-3　推理引擎的七大组件*

> 图里把 Sampler 合并在 “Model + Attention Backend” 框中；下面 2.3.6 会把它单独展开。实际引擎里，采样可能由 ModelRunner 调用独立的 sampler 模块完成。

### 2.3.1 API / Frontend

负责与外界交互：

- 暴露 OpenAI 兼容的 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等接口；
- 处理鉴权、限流、请求校验、取消；
- 维护流式响应的 SSE/JSON 流；
- 在多实例部署时，可能还包含路由与负载均衡。

这一层通常是 CPU 密集型，且需要高并发。vLLM 用 Python 的 FastAPI/uvicorn；SGLang 也提供 HTTP 服务器，并在生产部署中常用独立的 Rust router。

### 2.3.2 Processor / Tokenizer

- 文本 tokenization；
- chat template 套用（不同模型的模板差异很大，是常见 bug 来源）；
- 多模态输入预处理（图像 resize、patch、音频特征）；
- 约束解码的语法编译（JSON Schema → FSM）；
- 计算 prompt 的 token 数与最大生成长度，做前置校验。

在高并发下，tokenization 可能成为 CPU 瓶颈。一些引擎会把它放在独立进程，或使用快速 tokenizer（如 `tokenizers` 的 Rust 实现）。

### 2.3.3 Scheduler

调度器是引擎的“大脑”，每一步要做四类决策：

1. **准入**：waiting 队列里哪些请求可以进入 running？
2. **组批**：本步 running 集合包含谁？每条序列处理多少 token？
3. **资源**：需要多少 KV 块/token 槽？显存够不够？
4. **抢占**：不够时踢谁？recompute 还是 swap？

调度器的输出通常是一个结构化的 `SchedulerOutput` / `ScheduleBatch`，包含：

- 本步要处理的序列 ID；
- 每条序列本步的 token 数（prefill chunk 或 1 个 decode token）；
- 每条序列的 KV 位置/块表；
- 是否需要采样、采样参数；
- 本步的 token budget 使用情况。

### 2.3.4 KV Cache Manager

负责 KV 的物理管理：

- 初始化时按显存预算创建 KV 池；
- 为序列分配/追加 KV 空间；
- 维护逻辑位置到物理位置的映射；
- 识别可复用的前缀（哈希或基数树）；
- 序列结束时回收，按 LRU 等策略驱逐；
- 支持共享（引用计数、copy-on-write）；
- 在 PD 分离场景中，配合 KV 传输。

它是 vLLM 与 SGLang 差异最大的模块，第 3～5 章会重点展开。

### 2.3.5 Executor / Worker / ModelRunner

负责真正在 GPU 上执行：

- **Executor**：管理一组 Worker，决定单进程/多进程/Ray 等执行方式；
- **Worker**：每张 GPU 一个，负责加载权重、初始化 KV 池、执行 forward；
- **ModelRunner**：把调度器给的“逻辑 batch”转成 GPU 张量（`input_ids` 是本步 token ID，`positions` 是每个 token 的位置、供 RoPE 使用，`slot_mapping` 是 KV 写入槽位——这是 vLLM 的叫法，SGLang 对应 `out_cache_loc`；`attention metadata` 包含序列长度/块表等），调用模型与注意力后端；
- **Model**：具体的 Transformer 实现，通常按 HuggingFace 配置构建；
- **Attention Backend**：FlashAttention、FlashInfer、Triton、FlexAttention、MLA 专用 kernel 等。

这一层的关键工程问题：

- 如何减少 CPU→GPU 的 launch 开销（CUDA Graph）；
- 如何为不同长度的序列构造 varlen/分页注意力的元数据；
- 如何在 TP/PP/EP 下切分计算与通信；
- 如何让采样也留在 GPU 上，避免每步 D2H 拷贝。

### 2.3.6 Sampler / Logits Processor

从 logits 得到下一个 token：

- 温度缩放、top-k、top-p、min-p、repetition penalty；
- 约束解码：用 FSM/语法 mask 掉非法 token；
- 投机解码的验证与接受；
- 随机数状态管理（保证可复现性）。

现代引擎会把采样放在 GPU 上完成，只把最终 token ID（甚至只把文本）传回 CPU，避免每步传输整个词表大小的 logits（例如 128K 词表 × 4 字节 = 512 KB，每步都传会很痛）。

### 2.3.7 Output Processor / Detokenizer

- 判断停止条件（EOS、停止字符串、最大长度、工具调用结束）；
- 增量 detokenize：处理一个 token 可能对应半个 UTF-8 字符、或跨 token 的停止字符串；
- 流式返回；
- 统计 prompt/completion token 数；
- 记录指标（TTFT、TPOT、队列长度、cache hit）。

停止字符串的判断尤其棘手：一个停止串可能横跨多个 token，必须做增量匹配，不能只看当前 token。

---

## 2.4 一步（Iteration）的完整解剖

把上面的组件串起来，一次调度-执行循环大致如图 2-4 所示：

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Scheduler  │────▶│ SchedulerOutput│────▶│  ModelRunner    │
│ 选序列/算预算 │     │ 序列、token数、│     │ 构造 GPU 输入    │
└─────────────┘     │ KV 块表        │     └────────┬────────┘
                    └──────────────┘              │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Model Forward       │
                                        │ Embedding → Layers  │
                                        │ → Attention → MLP   │
                                        │ → Logits            │
                                        └──────────┬──────────┘
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Sampler             │
                                        │ 温度/top-p/约束 mask │
                                        └──────────┬──────────┘
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Output Processor    │
                                        │ 停止判断/detokenize  │
                                        │ 流式返回/释放资源    │
                                        └─────────────────────┘
```

*图 2-4　一次 iteration 的数据流*

伪代码：

```python
while True:
    # 1. 调度：决定本步跑哪些序列、各跑多少 token
    sched_out = scheduler.schedule()

    # 2. 准备 GPU 输入：input_ids、positions、slot_mapping、block_tables
    model_input = model_runner.prepare_input(sched_out)

    # 3. 前向：所有层 + 注意力 + LM head
    logits = model.forward(model_input)

    # 4. 采样：每个序列得到下一个 token
    next_tokens = sampler.sample(logits, sched_out.sampling_params)

    # 5. 后处理：停止判断、detokenize、流式输出
    outputs = output_processor.process(sched_out, next_tokens)

    # 6. 状态更新：追加 token、更新 KV 长度、释放完成序列
    scheduler.update_from_output(sched_out, outputs)
```

真实系统会把这六步中的很多环节重叠起来：

- 第 1 步的 CPU 调度与上一轮的 GPU 计算重叠；
- 第 5 步的 detokenize 与下一轮 GPU 计算重叠；
- 多卡时，第 3 步内部还有 all-reduce/all-to-all 通信。

这些重叠是“纸面算力”与“实际吞吐”之间的关键差距。

### 2.4.1 一个具体的调度例子（数字为假设）

假设某引擎的配置是：

- `max_num_batched_tokens = 8192`；
- `max_num_seqs = 4`；
- 当前有 4 条请求：

| 请求 | 状态 | 本步想做什么 |
|---|---|---|
| A | decode 中 | 生成 1 个 token |
| B | prefill 中，prompt 共 12000 token，已完成 4096 | 继续 prefill 下一块 |
| C | waiting，prompt 2000 token | 想开始 prefill |
| D | waiting，prompt 300 token | 想开始 prefill |

调度器可能这样组批：

```text
本步 token budget = 8192

A: 1 token（decode）
B: 4096 token（prefill chunk）
C: 2000 token（prefill）
D: 300 token（prefill）
----------------------------
合计: 6397 token，序列数 4，未超过两个上限
```

于是这一批里同时有 decode 和三个 prefill。GPU 一边处理 B/C/D 的大矩阵（吃算力），一边顺手帮 A 生成一个 token（吃带宽），两种负载互补，利用率更高。

下一步，B 继续剩下的 prefill，A 继续 decode；如果 C 或 D 完成了 prefill，就转入 decode。整个过程不需要等“一整批”结束，这就是 continuous batching。

> **注意**：真实引擎还要考虑 KV 块是否够、前缀是否命中、优先级、抢占等，这里的数字只是帮助理解“一步之内可以混跑不同请求”。

---

## 2.5 KV Cache 管理的通用难题与分页思想

### 2.5.1 变长序列的分配难题

假设每条序列的 KV 需要连续显存，长度动态增长。你会遇到：

- **外部碎片**：序列 A 释放后留下 3 GB 空洞，序列 B 需要 4 GB 连续空间，虽然总空闲够，但放不下；
- **内部浪费**：为了“可能生成到 max_tokens”，每条序列预留最大长度，实际只用了一小部分；
- **无法共享**：两个请求有相同 system prompt，却各存一份；
- **扩容困难**：序列变长时需要重新分配、拷贝。

这就是操作系统在 1960 年代就遇到过的问题。答案也是现成的：**分页（paging）**。

### 2.5.2 分页的核心思想

把 KV Cache 切成固定大小的 **块（block/page）**，例如每块 16 个 token（图 2-5；vLLM 0.34.0 的默认块大小是 64，这里用 16 便于手算）：

```text
逻辑视图（每条序列连续）          物理视图（GPU 显存中的块池）

序列 A: [块0][块1][块2]          物理块 7  ← 块0
序列 B: [块0][块1]               物理块 3  ← 块1
                                 物理块 9  ← 块2
                                 物理块 1  ← B 的块0
                                 物理块 5  ← B 的块1
```

*图 2-5　逻辑视图与物理视图*

每条序列维护一张 **块表（block table）**（图 2-6）：

```text
序列 A 的块表: [7, 3, 9]
序列 B 的块表: [1, 5]
```

*图 2-6　每条序列维护一张块表*

注意力 kernel 通过块表把逻辑位置翻译成物理地址，从非连续的显存块中 gather 出 K/V。这样：

- 物理块不需要连续，**消除外部碎片**；
- 按需分配，最后一块可能只用了几个 token，**内部碎片最多 block_size-1 个 token**；
- 块可以共享：两个序列的块表指向同一物理块，引用计数 +1；写入时 copy-on-write；
- 序列结束时按块回收，回收粒度小，利用率高。

**为什么“最多浪费 block_size-1 个 token”？** 因为一条序列的最后一个块可能只装了 1 个 token，剩下 15 个空位；也可能装满了 16 个，浪费 0 个。最坏情况就是“只装了 1 个”，浪费 15 个。

```text
block_size = 16，序列有 100 个 token：
100 = 6 × 16 + 4
→ 6 个满块 + 第 7 个块只用了 4 个槽位
→ 第 7 个块浪费 12 个槽位（≤ 15）
```

注意这是**每条序列**最多浪费 15 个 token，不是每个块都浪费。块越小，浪费上限越小；但块太小会让块表变长、kernel 寻址更频繁。这就是块大小需要折中的原因。

### 2.5.3 分页带来的新问题

- **kernel 复杂度**：注意力不能再假设 K/V 连续，必须支持 block table 索引；好在这一开销可以被 FlashAttention/FlashInfer 这类 kernel 吸收。
- **元数据开销**：每条序列要维护块表；块越大，内部碎片越大；块越小，元数据越多。
- **前缀共享的粒度**：块级共享只能共享完整块；如果两个 prompt 只有最后几个 token 不同，前面的完整块仍可共享，但最后一块不能。
- **驱逐策略**：块池满了要驱逐谁？LRU 是常见选择，但要考虑引用计数与正在运行的序列。

vLLM 的 PagedAttention 就是这套思想的工程化；SGLang 的 RadixAttention 则进一步把“前缀共享”从块级提升到 token 级（`page_size=1` 时），用基数树组织 KV 索引。更准确地说，PagedAttention 关注“KV 放在哪些物理页”，RadixAttention 关注“哪些前缀可以共享、如何索引”；SGLang 的 KV 仍然存在分页/槽位池中。两者并不互斥，而是同一问题空间里的不同层次。

---

## 2.6 调度策略的通用工具箱

不同引擎的调度器名字不同，但策略空间是共享的。下面这些概念在两边都会反复出现。

### 2.6.1 先来先服务（FCFS）

按到达顺序调度，简单、公平、可预测。缺点是：

- 一个超长 prompt 可能挡住后面所有短请求（队头阻塞）；
- 不感知前缀缓存，可能错过高复用机会。

vLLM 默认是 FCFS（也支持 priority）。

### 2.6.2 最长前缀匹配（LPM）

优先调度“与已缓存前缀匹配最长”的请求，以最大化 KV 复用、减少重复 prefill。适合：

- 多轮对话（同一会话共享历史）；
- Agent（同一任务的分支共享上下文）；
- RAG（多个查询共享同一批文档）；
- few-shot / system prompt 很长的场景。

代价是可能让“前缀短但等待久”的请求饿死，需要配合公平性机制。SGLang 提供 LPM、DFS-weight、HRRN 等缓存感知策略，但需要显式开启；SGLang 0.4.0 曾默认 LPM，0.4.6 起（含 0.5.23）已改为默认 FCFS。

### 2.6.3 Token Budget 与 Chunked Prefill

`max_num_batched_tokens` 限制一步内处理的 token 总数。长 prompt 会被切成多个 chunk，分步 prefill。chunked prefill 的系统化分析常引用 Sarathi-Serve（2024），它把长 prefill 切成“无停顿”的 chunk，并与 decode 混批：

```text
Step 1: [A prefill chunk 1 (2048)] [B decode] [C decode]
Step 2: [A prefill chunk 2 (2048)] [B decode] [C decode]
Step 3: [A prefill chunk 3 (剩余)]  [B decode] [C decode] [D prefill]
```

好处：

- 限制激活峰值，避免长 prompt 导致 OOM；
- 让 decode 请求不必等一个超长 prefill 完成，改善 ITL；
- 提高 GPU 利用率（prefill 的计算密集与 decode 的带宽密集可以互补）。

代价：

- 长 prompt 的 TTFT 变长（要分多步）；
- 调度更复杂，prefill 与 decode 的 token budget 分配需要策略。

### 2.6.4 抢占（Preemption）

当 KV 池不足时，调度器必须“踢人”。两种方式：

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **Recompute** | 丢弃 KV，回到 waiting，之后重新 prefill | 实现简单，不占 CPU 内存 | 浪费算力，TTFT 抖动 |
| **Swap** | KV 换出到 CPU，恢复时换入 | 不浪费算力 | 占 PCIe 带宽，CPU 内存压力 |

vLLM V1 主要使用 recompute；SGLang 也以 recompute 为主，并可通过分层缓存减少重算。选择哪种取决于 workload 与硬件。

### 2.6.5 优先级与公平性

生产环境常有优先级：

- 付费用户 > 免费用户；
- 在线请求 > 离线批处理；
- 短请求 > 长请求（为了压 P99）。

但优先级会带来饿死风险，需要 aging（等待越久优先级越高）、配额、预留槽位等机制。两家引擎都提供不同程度的优先级/配额支持，具体以版本文档为准。

---

## 2.7 分布式并行：一张卡放不下之后

### 2.7.0 先用“小组作业”理解三个通信词

分布式里经常出现 all-reduce、all-gather、all-to-all。先用小组作业类比：

| 通信 | 小组作业类比 | 技术上做什么 |
|---|---|---|
| **all-reduce** | 每个人算一个分数，把所有人的分数加起来，再把总分告诉每个人 | 所有 GPU 的向量按元素求和，结果广播给所有 GPU |
| **all-gather** | 每人写报告的一章，最后把各章拼成完整报告，发给每个人 | 每个 GPU 收集所有 GPU 的分片，拼成完整张量 |
| **all-to-all** | 每人把不同的章节发给不同同学，每个人收到自己需要的部分 | 每个 GPU 向其他 GPU 发送不同的数据块 |

TP 每层末尾通常要 all-reduce（把各卡的部分和加起来）；EP 路由专家时要 all-to-all（把 token 发给持有对应专家的 GPU）。通信不是免费的，NVLink/RDMA 的带宽和延迟会直接影响并行效率。

### 2.7.1 张量并行（Tensor Parallelism, TP）

把每一层的权重按维度切开，分到多张卡：

- 注意力头分给不同 GPU（每卡少几个 head）；
- MLP 的中间维度切开；
- 每层末尾做 all-reduce 汇总。

特点：

- 延迟低，适合单机多卡；
- 通信量大，强依赖 NVLink/NVSwitch；
- 注意力头数、KV 头数与 MLP 中间维度通常要能被 TP 整除；GQA 模型的 KV 头很少（如 8 个），对 TP 度数尤其敏感；
- KV Cache 也按头切分，每卡只存自己那部分。

### 2.7.2 流水线并行（Pipeline Parallelism, PP）

按层切分：GPU 0 放前几层，GPU 1 放后几层。微观批次流水线执行。

特点：

- 通信量小（只传激活）；
- 有流水线气泡，延迟不一定低；
- 适合跨机、层数多的超大模型；
- 推理场景不如 TP 常用，但在超大模型/多机上仍有一席之地。

### 2.7.3 数据并行（Data Parallelism, DP）

每张卡/每个副本持有完整模型，独立处理不同请求。

特点：

- 吞吐线性扩展（理想情况）；
- 单请求延迟不变；
- 需要负载均衡；
- 显存利用率低（每副本都存一份权重）。

### 2.7.4 专家并行（Expert Parallelism, EP）

针对 MoE 模型：不同专家放在不同 GPU 上，token 通过 all-to-all 被路由到对应专家。

特点：

- 是超大 MoE（如 DeepSeek-V3/R1）的关键；
- 负载不均会导致“忙的专家排队、闲的专家空转”；
- 需要 EPLB（Expert Parallel Load Balancer）之类的机制动态调整专家分布；
- 通信模式比 TP 更复杂。

### 2.7.5 序列/上下文并行（SP/CP）

把长序列的 token 维度切开，协同计算注意力。适合超长上下文（数十万 token），但实现复杂，通常与 TP/PP 组合使用。

### 2.7.6 PD 分离（Prefill-Decode Disaggregation）

把 prefill 和 decode 放到不同的 GPU 池：

```text
请求 → [Prefill 池] --KV 传输--> [Decode 池] → 输出
```

动机：

- prefill 计算密集、decode 带宽密集，混跑会互相干扰；
- 两者扩缩容比例不同（长 prompt 多 → 多配 prefill；长输出多 → 多配 decode）；
- 可以分别选择硬件与并行策略。

代价：

- KV 需要在池间传输，占用 RDMA/NVLink 带宽；
- 架构复杂，故障域更多；
- 不是所有 workload 都划算。

vLLM 与 SGLang 都支持 PD 分离，只是传输后端与成熟度不同。第 6、10 章会再讨论。

---

## 2.8 优化技术地图

把常见技术按“主要解决什么问题”归类：

| 技术 | 主要收益 | 主要代价/限制 |
|---|---|---|
| Continuous batching | 吞吐 ↑，GPU 利用率 ↑ | 调度与 KV 管理复杂 |
| Paged / varlen KV | 消除外部碎片，支持动态长度 | kernel 需支持块表，元数据开销 |
| Prefix caching | 共享前缀只算一次，TTFT ↓、吞吐 ↑ | 哈希/树维护，缓存污染风险 |
| Chunked prefill | 激活峰值 ↓，ITL ↓，prefill/decode 可混跑 | 长 prompt TTFT ↑，调度复杂 |
| CUDA Graph | 消除 kernel launch 的 CPU 开销 | 捕获固定形状，动态 shape 需 padding/分段 |
| CPU/GPU 重叠调度 | 小模型/高 QPS 下吞吐 ↑ | 实现复杂，依赖事件同步 |
| 投机解码 | TPOT ↓（接受率高时） | 额外草稿模型/头，显存与算力开销 |
| 量化（FP8/INT4） | 权重与 KV 带宽 ↓，吞吐 ↑ | 精度损失，kernel/硬件支持不一 |
| PD 分离 | 两阶段独立扩缩容，SLO 更稳 | KV 传输、架构复杂度 |
| 分层缓存（GPU/CPU/磁盘） | 缓存命中率 ↑，长会话成本 ↓ | 传输带宽、缓存一致性 |
| 结构化输出加速 | 约束解码开销 ↓ | 语法编译、FSM 维护 |

**重要**：这些技术之间会相互作用。例如 prefix caching 提高了缓存命中，但也可能让不同请求的 KV 长度分布更不均匀，进而影响调度与抢占。评估时不能只看单项。

---

## 2.9 两个引擎的共同血脉与分野

### 2.9.1 共同血脉

- **Orca（2022）**：提出 iteration-level scheduling / continuous batching，奠定现代推理引擎的调度模型。
- **PagedAttention / vLLM（2023）**：把操作系统分页思想引入 KV Cache，解决变长序列的显存碎片问题。
- **RadixAttention / SGLang（2023—2024）**：把 KV Cache 的**前缀索引**组织成基数树（KV 数据仍在分页/槽位池中），把前缀复用做到 token 级，并把“程序结构”纳入推理优化。
- **2024—2026 的收敛**：两家都支持 chunked prefill、prefix caching、CUDA Graph、投机解码、PD 分离、分层缓存、大规模 EP；差异更多体现在实现成熟度、调度策略与生态，而不是“有没有某个功能”。

### 2.9.2 主要分野

| 维度 | vLLM | SGLang |
|---|---|---|
| KV 组织 | 固定大小块 + 块表 | token 级基数树**索引** + 分页/槽位内存池（`page_size` 默认 1） |
| 前缀复用粒度 | 块级（长期默认 16 token，0.34.0 起默认 64） | 默认 token 级；`page_size>1` 时退化为页级 |
| 默认调度 | FCFS（支持 priority） | FCFS；可选 LPM/DFS-weight/HRRN 等缓存感知策略 |
| 前端 | OpenAI API 为主 | OpenAI API + Python 前端语言 |
| 强项场景 | 通用 serving、模型覆盖、硬件后端 | 前缀密集、Agent、结构化输出、大规模 MoE |
| 工程风格 | 通用、抽象、生态大 | 性能导向、贴近热点模型 |

第 3、4 章分别深入两个引擎；第 5 章做正面对比。

---

## 2.10 常见误区

**误区 1：推理引擎就是“模型 + 一层 HTTP”。**  
如果只是这样，所有请求会互相排队、显存会浪费、长 prompt 会阻塞短请求。引擎的价值在于调度、KV 管理和执行优化。

**误区 2：KV Cache Manager 只是“存 KV 的地方”。**  
它还负责前缀匹配、块/槽位分配、共享、引用计数、驱逐、抢占回收。它决定了显存利用率和缓存命中率。

**误区 3：Scheduler 就是一个 FIFO 队列。**  
调度器还要考虑 token budget、prefill/decode 混跑、chunked prefill、优先级、公平性、抢占。队列只是它最表面的部分。

**误区 4：batch 就是“一批请求”。**  
在 continuous batching 里，batch 是“本步要处理的 token 集合”，每条序列处理 1 个 token（decode）或一段 chunk（prefill），甚至混合。

**误区 5：并行一定更快。**  
TP/PP/EP 都有通信开销和气泡；batch 太小时，通信可能比计算还贵。并行是为了“放得下/跑得快”，不是无条件的加速按钮。

**误区 6：PD 分离一定更好。**  
PD 分离适合两阶段资源需求差异大、需要独立扩缩容的场景；如果 KV 传输成本高于收益，混跑反而更划算。

**误区 7：GPU 利用率高就一定是好事。**  
如果 GPU 在大量重算（recompute）或做无效 prefill，利用率高只是“忙得没意义”。要看有效吞吐和 goodput。

---

## 2.11 本章小结

- 推理引擎的核心抽象是 **Request / Sequence / Batch**；continuous batching 让 Batch 每一步都可变。
- 七大组件：API、Processor、Scheduler、KV Manager、Executor/Worker、Sampler、Output Processor。
- 一次迭代 = 调度 → 准备输入 → 前向 → 采样 → 后处理 → 状态更新；真实系统会把 CPU 侧工作与 GPU 计算重叠。
- KV 管理的通用答案是**分页**：固定块 + 块表，消除外部碎片、支持共享与按需分配。
- 调度工具箱：FCFS、LPM、token budget、chunked prefill、抢占（recompute/swap）、优先级。
- 分布式：TP/PP/DP/EP/SP/CP/PD，各有适用场景与代价。
- vLLM 与 SGLang 共享 continuous batching 与 paged/varlen attention 的血脉，主要差异在 **KV 数据结构、调度策略与前端可编程性**。

> **用大白话再说一遍**：引擎像一套考场/教务系统。接卷窗口收下卷子，监考决定先答哪份、每份答多少，草稿纸管理员记录每份卷子的笔记放哪、哪些笔记能共享，考生大脑作答，誊写员把答案交出去。PagedAttention 管“草稿纸怎么分”，RadixAttention 在分好的草稿纸池上建“前缀索引”，continuous batching 是“哪份答完哪份交卷、空出的桌面立刻补下一份”的排卷方式。

---

## 思考题

1. 为什么 continuous batching 要求 KV Cache 管理支持“动态增删”？如果 KV 必须连续分配，会遇到什么具体问题？
2. 画出请求从 WAITING 到 FINISHED 的状态图，并标出 recompute 与 swap 两种抢占路径。
3. `max_num_batched_tokens` 和 `max_num_seqs` 分别限制什么？如果前者很大、后者很小，系统会表现成什么样？反过来呢？
4. Chunked prefill 为什么能改善 ITL？它又如何影响长 prompt 的 TTFT？请给出一个具体的权衡例子。
5. 张量并行下，KV Cache 是如何切分的？为什么 GQA 模型对 TP 的约束更严格？
6. PD 分离适合什么样的流量？如果不做 PD 分离，prefill 和 decode 混跑会带来哪些干扰？
7. 用“问题地图”解释：prefix caching 同时影响了五大问题中的哪几个？

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 3 章 vLLM：以 PagedAttention 为核心的通用推理引擎 →](03-vllm.md)
