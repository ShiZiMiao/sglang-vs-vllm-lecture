# 第 4 章 SGLang：以 RadixAttention 与程序级优化为核心的推理栈

> **本章目标**：理解 SGLang 的两层含义（前端语言 + 运行时）、RadixAttention 的树结构与操作、缓存感知调度，以及它在结构化输出、大规模 MoE、PD 分离上的工程取舍。
>
> **先修知识**：[第 1 章](01-foundations.md) 的 KV Cache、[第 2 章](02-engine-architecture.md) 的分页思想、[第 3 章](03-vllm.md) vLLM 的块级前缀缓存。建议先读第 3 章，才能体会“块级 vs token 级”的差别。
>
> **一句话概括**：SGLang 把 KV Cache 的**前缀索引**组织成一棵“前缀树”（KV 数据本身仍在显存池中），相同前缀的请求共用树上的节点；如果开启缓存感知调度，还可以优先安排前缀命中更长的请求（默认策略是 FCFS）。
>
> **给零基础同学**：不要被“基数树”“引用计数”吓到。本章 4.0 节会用“笔记目录树”类比，把树的匹配、分裂、共享讲清楚。
>
> **版本与来源**：本章的默认值与源码路径按 **SGLang 0.5.23** 核对（2026-11-25）；参数逐项说明见 [SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)。

---

## 4.0 先用“笔记目录树”类比理解 RadixAttention

RadixAttention 要解决的是前缀复用：把多个请求的 token 前缀组织成基数树，相同前缀共用同一段 KV，减少重复 prefill。想象考场里有一本共享的**笔记目录**，按题目的开头组织（图 4-1）：

```text
请
├── 解释什么是
│   ├── KV Cache
│   └── PagedAttention
└── 比较
    └── vLLM 与 SGLang
```

*图 4-1　用笔记目录树类比基数树*

“请解释什么是”这段开头被两份笔记共享；“请比较”是另一条分支。RadixAttention 的树也是同样的结构：

- 树的根 = 笔记目录的总入口；
- 每条边 = 一段 token 序列（不是单个 token），像目录里的一级标题；
- 节点 = 这段 token 序列在 KV 池中的索引（从根到节点的路径拼出完整前缀）；
- 子节点 = 在这个前缀之后分叉的不同后缀；
- 多份卷子可以共享同一条路径上的笔记；
- 请求算过的前缀会“登记”到目录里：运行中先登记已算完的部分，结束后补全整条序列，供后续请求复用。

| 笔记目录 | RadixAttention |
|---|---|
| 总入口 | 树的 root |
| 一级标题 | 一段 token 前缀（如 system prompt） |
| 子标题 | 分叉后的后缀 |
| 一条完整目录路径 | 某个请求的完整 token 序列 |
| 两条路径共享开头 | 两个请求共享 KV |
| 删除空目录 | 驱逐没有引用的叶子节点 |

这样，多轮对话、Agent 分支、RAG 共享文档，都可以看成“走同一段笔记目录路径”。Radix Cache 会复用路径上的 KV；如果进一步开启缓存感知调度（如 LPM），调度器还会尽量让“路径相同”的请求一起走，减少重复 prefill。

---

## 4.1 定位与历史

SGLang 由 LMSYS Org（与 Vicuna、Chatbot Arena 同一社区）发起，核心论文是 2023 年预印、2024 年发表在 NeurIPS 的 *SGLang: Efficient Execution of Structured Language Model Programs*。

“SGLang”其实有两层含义，初学时很容易混淆：

1. **SGLang 前端语言**：一个嵌入 Python 的 DSL（domain-specific language），用来表达带分支、选择、循环、约束的 LLM 程序；
2. **SGLang Runtime（SRT）**：真正执行这些程序、也接受 OpenAI API 的推理引擎。

早期论文的主线是“用一套语言原语描述复杂 LLM 程序，再让运行时针对这些结构做优化”；后来 SRT 逐渐独立成一个通用高性能 serving 引擎，被广泛用于多轮对话、Agent、RAG、结构化输出、强化学习 rollout 和 DeepSeek 类大模型部署。

SGLang 的核心洞察是：

> 很多 LLM 应用的请求并不是彼此独立的。它们共享 system prompt、共享多轮历史、共享检索文档、共享同一棵推理树的祖先。如果推理引擎能“看见”这些共享结构，就能避免大量重复 prefill。

vLLM 的 PagedAttention 主要解决“显存怎么放”；SGLang 的 RadixAttention 进一步解决“相同前缀的 KV 怎么复用”。两者不是替代关系，而是同一问题不同层次的答案。

---

## 4.2 从程序原语到运行时

### 4.2.1 为什么需要前端语言

考虑一个常见的 Agent 流程：

```text
用户问题
  → 模型生成“计划”
  → 并行尝试 3 个工具调用
  → 根据结果选择一个分支继续
  → 生成最终答案，并要求 JSON 格式
```

用朴素的 API 调用写，会有大量重复：

- 每个分支都要把相同的 system prompt 和问题重新发一遍；
- 分支之间共享的历史无法复用；
- 约束（JSON Schema、正则）要单独处理；
- 并行分支的调度依赖用户自己写线程池。

SGLang 前端语言提供了一组原语，把这些结构显式表达出来：

| 原语 | 作用 |
|---|---|
| `sgl.gen` | 生成一段文本，可带 `max_tokens`、`stop`、`regex`、`json_schema` 等约束 |
| `sgl.select` | 从多个候选里选择一个（如分类、路由） |
| `sgl.fork` / `sgl.join` | 分叉出多个并行分支，再汇合 |
| `sgl.system` / `sgl.user` / `sgl.assistant` | 构造对话消息 |
| 变量与表达式 | 把前一步输出作为后一步输入，支持 Python 控制流 |

一个最小例子：

```python
import sglang as sgl

@sgl.function
def answer(s, question):
    s += sgl.system("你是一个严谨的 AI 助教。")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))

state = answer.run(question="什么是 KV Cache？")
print(state["answer"])
```

> **版本提示**：`@sgl.function` 这类前端 API 在 0.4 之后经历过调整，上面两段代码只用于说明“语言层表达结构”的思想，不保证在新版本逐字可运行。生产代码请以当前版本文档为准；多数用户直接用 OpenAI 兼容接口，不接触这层 DSL。

分支例子（概念版）：

```python
@sgl.function
def multi_path(s, question):
    s += sgl.user(question)
    forks = sgl.fork(["用数学解释", "用类比解释", "用代码解释"])
    for f in forks:
        f += sgl.assistant(sgl.gen("explanation", max_tokens=256))
    s += sgl.assistant("请综合以上三种解释，给出最终答案：")
    s += sgl.gen("final", max_tokens=512)
```

在运行时看来，这些 fork 出来的分支共享同一段前缀，因此可以共用 KV Cache；`select` 可以用约束解码只生成候选 token；`gen` 的 JSON Schema 可以编译成 FSM。**语言层表达结构，运行时层利用结构**，这是 SGLang 最初的设计哲学。

> 今天大多数用户直接使用 OpenAI 兼容 API，不一定接触前端语言。但理解这套原语，有助于理解 SGLang 运行时为什么如此强调“前缀共享”和“结构化约束”。

### 4.2.2 运行时（SRT）

SRT 是一个完整的 serving 引擎，主要组件包括：

- **HTTP Server**：OpenAI 兼容接口、原生 `/generate` 接口；
- **TokenizerManager**：tokenize、chat template、多模态输入；
- **Scheduler**：调度、RadixCache 管理、批处理；
- **TP Worker**：每张 GPU 一个，执行模型；
- **ModelRunner**：构造输入、调用 attention backend、采样；
- **DetokenizerManager**：增量 detokenize、流式输出；
- **Router**（独立部署）：缓存感知的负载均衡。

其架构与 vLLM 有相似之处，但调度器与 KV 管理是围绕**基数树**设计的。

---

## 4.3 RadixAttention：把 KV Cache 组织成前缀树

### 4.3.1 基数树（Radix Tree）速览

基数树是一种压缩前缀树：每条边可以携带一段 token 序列，而不是单个字符。例如三个 prompt：

```text
A: "You are a helpful assistant. Q: 什么是 KV Cache?"
B: "You are a helpful assistant. Q: 什么是 PagedAttention?"
C: "You are a helpful assistant. Q: 什么是 RadixAttention?"
```

可以组织成图 4-2 所示的结构：

```text
                    root
                     │
     "You are a helpful assistant. Q: 什么是 "
                     │
              ┌──────┴──────┐
              │             │
          "KV Cache?"   "PagedAttention?" ...
              │
           [节点 A]
```

*图 4-2　三个 prompt 共享前缀的基数树*

共享的前缀只存一份 KV；不同的后缀各自成节点。节点的边标签是 token 序列，节点上保存这段 token 对应的 KV 索引。SGLang 默认开启 Radix Cache（可用 `--disable-radix-cache` 关闭）；本节讲的就是它开启时的行为。

### 4.3.2 树节点存什么

SGLang 的 `TreeNode` 大致包含：

```text
TreeNode
├── key: List[int]              # 这条边上的 token 序列
├── value: Optional[Tensor]     # 这段 token 对应的 KV 池索引
├── children: Dict[int, Node]   # 按第一个 token 索引子节点
├── parent: Node
├── lock_ref: int               # 有多少运行中的请求/操作锁定这段 KV（旧版可能叫 ref_counter）
├── last_access_time: float     # 用于 LRU 驱逐
└── hash_value: int             # 可选的哈希，用于快速比较/去重
```

几个关键设计：

- **边标签是 token 序列**，不是单个 token；这让树更紧凑，减少节点数量；
- **value 是 KV 池的索引**，不是 KV 数据本身；KV 数据存在一个大的 token 池里（`TokenToKVPool`）。这个池仍然是预分配的显存页/槽位，RadixAttention 只是索引结构，并不替代底层的分页分配；
- **lock_ref** 保护正在被请求使用的节点，驱逐时不能动；
- **last_access_time** 支持 LRU：越久没被访问的叶子越先被驱逐。

> **引用计数的共享笔记类比**：树节点像一份共享笔记，`lock_ref` 是“当前有多少份卷子正在用它”。只要还有卷子在用（>0），就不能把笔记收走；所有卷子都交卷了（=0），才允许清理。这样能保证正在生成的请求不会读到一个被释放的 KV 节点。

### 4.3.3 匹配（match_prefix）

新请求到来时，调度器用它的 token 序列在树中做最长前缀匹配：

```python
def match_prefix(root, token_ids):
    node = root
    matched = 0
    kv_indices = []

    while matched < len(token_ids):
        child = node.children.get(token_ids[matched])
        if child is None:
            break

        # 比较 child.key 与 token_ids[matched:]
        common = longest_common_prefix(child.key, token_ids[matched:])
        if common == 0:
            break

        kv_indices.extend(child.value[:common])
        matched += common

        if common < len(child.key):
            # 只匹配了节点的一部分：分裂节点，并返回前半段节点
            node = split_node(child, common)
            break

        node = child

    return kv_indices, node, matched
```

关键点是**节点分裂**（图 4-3）：

```text
匹配前:
  [ "ABCDEFG" ]  (一个节点)
       │
     children

请求前缀: "ABCD"
匹配后:
  [ "ABCD" ]  (原节点被分裂)
       │
  [ "EFG" ]   (新节点，继承原 value 的后半段)
       │
     children
```

*图 4-3　match_prefix 命中节点内部时的分裂*

分裂只影响元数据，不复制 KV 数据：原节点的 `value` 按长度切开，前半段留在父节点，后半段给子节点。这样，任意 token 级前缀都能被精确匹配，而不像块级缓存那样必须按块大小对齐（示例 16 token，vLLM 0.34.0 默认 64；前提是 `page_size=1`，把 page size 调大后匹配粒度会变粗）。

**手把手例子：一次“半路分叉”**

假设树里已经有一条完整路径：

```text
root
 └── "You are a helpful assistant. Q: 什么是 KV Cache?"
```

现在来了一个新请求：

```text
"You are a helpful assistant. Q: 什么是 KV 的缓存?"
```

它和原路径共享到 `... 什么是 KV`，但后面不同。匹配时会在 `KV Cache?` 这个节点内部停下，于是分裂（图 4-4）：

```text
分裂前:
  [ "You are ... 什么是 KV Cache?" ]

分裂后:
  [ "You are ... 什么是 KV" ]          ← 共享部分
        ├── [ " Cache?" ]              ← 原请求的后半段
        └── [ " 的缓存?" ]             ← 新请求的后缀
```

*图 4-4　一次“半路分叉”的匹配过程*

原节点的 KV 索引按长度切开，前半段留在父节点，后半段给 `" Cache?"` 子节点；新请求只需要为 `" 的缓存?"` 计算新的 KV。**共享的 `... 什么是 KV` 一个 token 都不用重算**。

### 4.3.4 插入（insert）

请求生成结束后，把它完整的 token 序列（prompt + 输出）插入树中：

1. 从根开始找最长公共前缀；
2. 如果已有路径覆盖了全部 token，只需更新 `last_access_time`；
3. 如果只覆盖了一部分，分裂节点；
4. 把剩余后缀作为新节点挂上，`value` 指向这段 KV 的池索引；
5. 更新引用计数与访问时间。

插入的复杂度与序列长度和分支数有关，通常是 $O(L)$ 级别的指针操作，相比 GPU 前向可以忽略；但在极高 QPS、大量短请求时，Python/C++ 的树操作也可能成为 CPU 瓶颈。

> chunked prefill 期间，已经算完的前缀也会被临时插入树中（对应源码里的 `cache_unfinished_req`），这样请求被分块或被抢占后仍能复用；请求全部结束后再补全整条序列。

> **$O(L)$ 是什么意思？** 大 O 记号描述“工作量随输入规模怎么增长”。$O(L)$ 表示 token 数从 1000 增加到 2000 时，树操作的工作量大致也翻倍，而不是平方增长。匹配/插入时，每个 token 基本只被走过一次；树的深度和分支数会影响常数项，但不会改变“线性”这个量级。跟 GPU 上几十亿次浮点运算相比，这些指针操作通常很小；只有当请求非常短、QPS 非常高时，CPU 树操作才会显现出来。

### 4.3.5 驱逐（evict）

当 KV 池不足时，RadixCache 需要释放一些节点：

- 只驱逐**叶子节点**（内部节点被多个分支共享，不能单独释放）；
- 跳过 `lock_ref > 0` 的节点（正在被请求使用）；
- 按 `last_access_time` 做 LRU，最久未访问的叶子优先；
- 释放节点的 KV 索引，从父节点的 children 中移除；
- 可能递归地让父节点变成新的叶子。

伪代码：

```python
def evict(need_tokens):
    freed = 0
    while freed < need_tokens:
        leaf = find_lru_leaf(exclude_ref_count_positive=True)
        if leaf is None:
            raise OutOfMemory
        freed += len(leaf.value)
        free_kv_indices(leaf.value)
        remove_from_parent(leaf)
    return freed
```

这与操作系统的页缓存淘汰非常相似：**LRU + 引用计数 + 只回收叶子**。

> **版本补充（SGLang 0.5.23）**：默认驱逐策略仍是 LRU，但 0.5.x 已支持通过 `--radix-eviction-policy` 选择其他策略、用 `--radix-eviction-policy-config` 配置参数，细节见官方 [Radix Cache Eviction Policy](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)。本章的讨论以默认 LRU 为准。

### 4.3.6 RadixAttention 的收益从哪里来

考虑一个多轮对话：

```text
第 1 轮: [system 2000 tokens][user Q1][assistant A1]
第 2 轮: [system 2000 tokens][user Q1][assistant A1][user Q2][assistant A2]
第 3 轮: [system 2000 tokens][user Q1][assistant A1][user Q2][assistant A2][user Q3]...
```

没有前缀缓存时，每一轮都要重新 prefill 全部历史；有 RadixAttention 时，第 2 轮只需 prefill 新增的 `[user Q2]`，第 3 轮只需 prefill `[user Q3]`。随轮数增加，节省的 prefill 计算量非常可观。

类似地：

- **Agent 分支**：同一任务的不同工具调用共享前面的推理轨迹；
- **RAG**：多个 query 共享同一批检索文档；
- **Few-shot**：所有请求共享同一段示例；
- **Beam search / 并行采样**：候选共享 prompt 前缀；
- **RL rollout**：同一 prompt 的多次采样共享前缀。

在缓存容量足够、前缀确实重复、调度策略匹配的前提下，这类负载是 SGLang 最容易体现优势的场景；反过来，如果前缀几乎不重复，树维护可能只是净开销。

### 4.3.7 代价与限制

- **元数据开销**：每个节点要存 key、value、children、引用计数、时间戳；树很大时内存不可忽视；
- **CPU 操作**：match/insert/evict 都在 CPU 侧，需要与 GPU 调度重叠，否则会成为瓶颈；
- **缓存污染**：一次性、不重复的长 prompt 会占用缓存，挤掉真正热的前缀；需要 LRU/容量策略；
- **安全隔离**：多租户下，前缀缓存可能造成信息泄漏（通过命中时间推断内容），需要租户隔离或盐值；
- **不是所有负载都受益**：如果每个请求的 prompt 都完全不同，树的价值有限，反而增加开销。此时可以关闭 Radix Cache。

---

## 4.4 缓存感知调度：LPM 与 DFS-weight

### 4.4.1 为什么调度要“看见”前缀

假设 waiting 队列里有三个请求：

| 请求 | prompt 长度 | 与缓存匹配的前缀 |
|---|---|---|
| A | 4000 | 3500 |
| B | 500 | 0 |
| C | 800 | 700 |

如果按 FCFS 先跑 B，再跑 A、C，那么 A 和 C 的高命中前缀可能在等待期间被其他请求驱逐；如果先跑 A 和 C，它们只需 prefill 500 和 100 个 token，整体吞吐更高。

> **排卷类比**：考场门口堆着很多卷子，如果按到达顺序随便发，可能同一段开头要反复做很多遍笔记。LPM 相当于“先把开头相同的卷子集中起来一起发”，一次覆盖更多份，效率高。但如果有几类卷子一直大量到达，开头冷门的卷子可能一直排不上——这就是 LPM 的公平性问题。

**最长前缀匹配（Longest Prefix Match, LPM）** 的策略就是：优先调度“已匹配前缀最长”的请求，最大化 KV 复用、减少重复计算。

### 4.4.2 SGLang 的调度策略

SGLang 的 `--schedule-policy` 提供多种策略（名称与可用项以版本为准）：

- **`fcfs`**：当前版本的默认策略，先来先服务，公平、可预测，但可能错过缓存复用；
- **`lpm`**：按最长前缀匹配优先，需要显式开启；SGLang 0.4.0 曾把它作为默认，0.4.6 起及 0.5.23 均为默认 `fcfs`，所以旧资料里的“默认 LPM”要按版本理解；
- **`dfs-weight`**：沿基数树给等待请求计算权重，优先调度更靠近热节点的请求，以提升缓存局部性（具体权重算法以版本为准）；
- **`hrrn`**：Highest Response Ratio Next，兼顾等待时间与执行代价，用于缓解饿死；
- **`shortest-prefill-first`**：优先短 prefill，降低平均 TTFT；
- **`random`**：随机，通常用于对照或避免系统性偏置。

从设计上看，LPM/DFS-weight/HRRN/SPF 属于“缓存感知”策略，FCFS/random 属于“缓存无关”策略；前者追求复用与吞吐，后者追求公平与可预测。当前版本默认是 FCFS；如果追求缓存复用，要显式切换到 LPM 或 DFS-weight，并评估公平性风险。

> **核对日期 2026-11（SGLang 0.5.23）**：`schedule_policy` 默认 `fcfs`，可选值有 `lpm`、`random`、`fcfs`、`dfs-weight`、`lof`、`priority`、`routing-key`、`hrrn`、`shortest-prefill-first`；`max_prefill_tokens` 默认 16384；`chunked_prefill_size` 默认 `None`（由引擎自动决定），设为 `-1` 可关闭 chunked prefill；`page_size` 默认 1。参数会随版本变化，启动前用 `--help` 或 [官方 Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) 复核。另注：0.5.x 的参数定义已从单一的 `server_args.py` 拆分到 `sglang/srt/arg_groups/fields/` 下。

LPM 的实现大致是：

1. 对 waiting 队列中的每个请求，调用 `match_prefix` 计算可复用的 token 数；
2. 按匹配长度降序排序（或维护优先队列）；
3. 依次尝试调度，直到 token budget 或显存耗尽；
4. 未调度的请求留在队列中，下一轮重新计算（因为缓存可能变化）。

### 4.4.3 LPM 的公平性问题

LPM 有一个明显风险：**长前缀请求可能饿死短前缀请求**。如果源源不断有高命中的长请求到来，低命中请求可能永远排不上。

缓解手段：

- 队列 aging：等待越久，优先级越高；
- 配额：为不同租户/请求类型预留槽位；
- 混合策略：一部分 batch 按 LPM，一部分按 FCFS；
- 在延迟敏感的在线场景，显式使用 `fcfs` 或优先级调度。

这是一个典型的“吞吐 vs 公平/尾延迟”权衡，没有免费午餐。

### 4.4.4 与 vLLM 的对比（预告）

| 维度 | vLLM | SGLang |
|---|---|---|
| 默认顺序 | FCFS | FCFS（可切 LPM 等缓存感知策略） |
| 前缀感知 | 缓存命中，但调度不按前缀排序 | 默认不排序；开启 LPM 后主动按前缀匹配排序 |
| 优点 | 公平、可预测、实现简单 | 前缀密集负载吞吐高、TTFT 低 |
| 风险 | 可能错过复用机会 | 可能饿死低命中请求 |

第 5 章会展开分析“什么时候这个差异真正重要”。

---

## 4.5 调度与执行的重叠：零开销批调度

### 4.5.1 CPU 侧开销从哪里来

每一步调度，CPU 要做很多事：

- 遍历 running/waiting 队列；
- 匹配前缀、分配/释放 KV；
- 构造 batch 元数据；
- 准备 attention 的 indptr/indices；
- 处理采样结果、停止条件；
- detokenize、流式输出。

模型越小、batch 越小、QPS 越高，GPU 计算越快，CPU 越容易成为瓶颈。此时 GPU 会周期性“等 CPU”，利用率出现锯齿。

### 4.5.2 重叠调度的思路

SGLang 的 **overlap scheduler**（零开销批调度）把第 $i$ 步的 CPU 侧工作与第 $i-1$ 步的 GPU 计算重叠（图 4-5）：

```text
时间 →
GPU:   [ forward batch i-1 ]  [ forward batch i ]  [ forward batch i+1 ]
CPU:        [ schedule i ]        [ schedule i+1 ]      [ schedule i+2 ]
             [ process i-1 ]       [ process i ]
```

*图 4-5　overlap scheduler 的 CPU/GPU 重叠*

这样，只要 CPU 工作时间不超过 GPU 前向时间，调度开销就被“藏”起来了，GPU 不再空等。

实现上需要：

- 双缓冲的 batch 状态（当前 batch / 下一 batch）；
- CUDA event/stream 同步，保证依赖正确；
- 对未来 token 的处理（停止条件、输出长度）做延迟一拍的处理；
- detokenizer 独立进程/线程，避免阻塞调度。

可以用 `--disable-overlap-schedule` 关闭，便于调试或排查正确性问题。对大多数 decode 负载，开启重叠能提升吞吐，尤其是小模型和高并发场景。

### 4.5.3 Chunked Prefill 与 Prefill/Decode 混跑

SGLang 同样支持 chunked prefill：

- `--chunked-prefill-size` 控制单个 chunk 的最大 token 数；
- `--max-prefill-tokens` 控制一步内 prefill 的 token 预算；
- 调度器会把 prefill chunk 与 decode 请求混在同一个 batch 中，以平衡算力与带宽。

与 vLLM 的差异主要在**排序策略**：开启 LPM 等缓存感知策略后，SGLang 会结合前缀匹配与 chunk 切分，优先让高命中请求进入；默认 FCFS 下仍按到达顺序。vLLM 则主要使用 FCFS + token budget。

### 4.5.4 CUDA Graph 与 Attention Backend

SGLang 的执行层优化包括：

- **CUDA Graph**：捕获 decode 步，按 batch size 分桶；0.5.23 用 `--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill` 分别控制 decode/prefill 的最大捕获 batch，用 `--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph` 分阶段关闭（旧版为 `--cuda-graph-max-bs` 与 `--disable-cuda-graph`）；
- **FlashInfer**：常见的高性能 attention backend 之一（是否为默认取决于模型、硬件与版本），支持 paged/ragged KV、FP8、MLA 等；
- **FlashAttention 3 / FlashMLA / CUTLASS MLA**：针对 Hopper 和 DeepSeek 类模型的专用后端；
- **torch.compile**（可选）：对部分模型/层做编译优化；
- **GPU 采样**：温度、top-p、约束 mask 都在 GPU 上完成；
- **增量 detokenize**：独立的 DetokenizerManager，与调度/执行流水化。

SGLang 与 FlashInfer 社区关系密切，很多 kernel 优化会先在 SGLang 落地；这也是它在特定模型上性能突出的原因之一。但“默认后端最快”不是绝对的，仍应针对模型和硬件实测。

---

## 4.6 结构化输出：压缩 FSM 与跳转解码

### 4.6.1 约束解码的基本原理

给定 JSON Schema、正则或 EBNF（扩展巴科斯范式，一种描述语法的格式），可以编译成一个有限状态机（FSM）：

- 每个状态代表“当前已匹配到哪里”；
- 每个状态允许的 next token 集合是确定的；
- 采样时，把非法 token 的 logit 置为 $-\infty$，只从合法 token 中采样；
- 生成一个 token 后，FSM 迁移到下一个状态。

**一个 JSON 的 FSM 例子**。假设要求输出：

```json
{"name": "Alice", "age": 30}
```

FSM 可以粗略表示成：

| 当前状态 | 只允许出现 | 说明 |
|---|---|---|
| 开始 | `{` | 必须以左花括号开头 |
| 键名 | `"name"` | 本例子假设键顺序固定；真实 JSON Schema 通常不规定键序 |
| 冒号 | `:` | 键名后必须是冒号 |
| 值 | 任意字符串 | 如 `"Alice"` |
| 逗号/结尾 | `,` 或 `}` | 还有 age 就逗号，否则结束 |
| 第二个键 | `"age"` | 本例子假设 age 紧随其后 |
| 值 | 整数 | 如 `30` |
| 结尾 | `}` | 必须闭合 |

> **注意**：JSON Schema 本身不规定对象键的顺序。这里为了演示压缩 FSM，额外假设“先 name 后 age”。真实实现会允许任意合法键序，状态数更多，但 jump-forward 的原理相同。

在“键名”状态，模型其实不需要“自由发挥”：它必须输出 `"name"`。如果这些 token 是确定的，SGLang 就可以一次性填入，而不是让模型逐个猜。这就是 jump-forward 的用武之地。

问题在于：词表可能有 10 万～20 万 token，每一步都计算“哪些 token 合法”很贵；FSM 状态也可能很多。

### 4.6.2 SGLang 的优化

SGLang 以 XGrammar 作为主要语法后端（也可选 llguidance/outlines 等，视版本），并做了两层优化：

1. **压缩 FSM**：把大量“只有一个合法转移”的连续状态压缩成一个，减少状态数与查表开销；
2. **Jump-forward decoding**：如果当前状态后面有一段**确定的 token 序列**（例如 JSON 的 `{"name": "`），就一次性把这些 token 直接填入，不必逐步采样。

```text
普通约束解码:
  step 1: 采样 '{'
  step 2: 采样 '"'
  step 3: 采样 'name'
  step 4: 采样 '"'
  step 5: 采样 ':'
  ...

Jump-forward:
  当前 FSM 确定接下来必然是 '{"name": "'
  → 一次性填入这几个 token（具体数量取决于 tokenizer）
  → 直接从值的部分开始采样
```

此外，SGLang 会把语法编译与前一步的 GPU 计算重叠，避免“收到请求才开始编译语法”的延迟。

### 4.6.3 使用方式

OpenAI 兼容接口：

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "生成一个用户信息"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

原生接口还支持 `regex`、`ebnf` 等更强约束。对于“必须输出合法 JSON/DSL”的场景，约束解码不仅提升成功率，还能减少无效 token 的生成，间接提升吞吐。

---

## 4.7 大规模并行：DP Attention、EP、EPLB 与 PD 分离

### 4.7.1 张量并行（TP）

与 vLLM 类似：按注意力头和 MLP 中间维度切分，每层 all-reduce。SGLang 的 TP 实现强调与 FlashInfer/FlashMLA 的配合，并支持多机 TP。

### 4.7.2 DP Attention

对 DeepSeek 这类使用 MLA 的模型，SGLang 提供了 **DP attention**：

- 把请求按数据并行（DP）分到不同 rank；
- 每个 rank 独立计算自己那部分请求的注意力；
- MoE 部分再用专家并行（EP）做 all-to-all；
- 避免 TP 下每张卡都复制完整的 KV/注意力计算。

配置通常是：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --enable-dp-attention \
  --dp-size 8 \
  --ep-size 8 \
  ...
```

DP attention 能显著提升大批量 decode 的吞吐，但要求模型结构、backend 和并行配置匹配；不是所有模型都适用。

### 4.7.3 专家并行与 EPLB

MoE 模型的专家分布在不同 GPU 上，token 通过 all-to-all 路由。问题是：不同专家的负载可能严重不均（热门专家排队，冷门专家空转）。

**EPLB（Expert Parallel Load Balancer）** 会统计专家负载，动态调整专家到 GPU 的映射，把热门专家复制/迁移到更多卡上，缓解长尾。SGLang 提供 `--enable-eplb` 及相关算法选项，是 DeepSeek 大规模部署的关键能力之一。

### 4.7.4 PD 分离

SGLang 支持把 prefill 和 decode 部署到不同实例（图 4-6）：

```text
┌─────────────┐      KV 传输       ┌─────────────┐
│ Prefill 实例 │ ─────────────────▶ │ Decode 实例  │
│ 计算 prompt  │  RDMA/NVLink       │ 逐 token 生成 │
└─────────────┘                    └─────────────┘
```

*图 4-6　PD 分离的部署形态*

典型配置：

```bash
# Prefill 节点
python -m sglang.launch_server --model-path ... \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --port 30000

# Decode 节点
python -m sglang.launch_server --model-path ... \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --port 30001
```

收益：

- prefill 与 decode 独立扩缩容；
- 避免长 prompt 的 prefill 阻塞 decode 的 ITL；
- 可以为两个阶段选择不同的并行策略/硬件。

代价：

- KV 传输占用高速网络；
- 需要 bootstrap 服务、连接管理、故障恢复；
- 小模型/短 prompt 场景可能不划算。

### 4.7.5 HiCache：把缓存延伸到 CPU 和磁盘

GPU 显存有限，Radix Cache 只能保留最近使用的前缀。**HiCache** 把缓存分层（图 4-7）：

```text
GPU HBM（最快，容量最小）
   ↕ PCIe/NVLink
CPU DRAM（较大，较快）
   ↕ 网络/磁盘
SSD/分布式存储（最大，最慢）
```

*图 4-7　HiCache 的 GPU/CPU/磁盘分层*

Radix 树仍然是索引结构，但节点的 KV 可以存放在不同层级；命中时按需预取到 GPU。对长会话、Agent、RAG 等“前缀复用跨请求、跨时间”的负载，HiCache 能显著提高缓存命中率，降低重复 prefill。

代价是传输延迟与缓存一致性管理；是否开启取决于 workload 的复用距离与网络带宽。

---

## 4.8 投机解码与前端生态

### 4.8.1 投机解码

SGLang 支持多种投机解码：

- **EAGLE / EAGLE-2 / EAGLE-3**：用轻量草稿头预测多个 token，树状验证；
- **Draft model**：小模型起草，大模型验证；
- **MTP（Multi-Token Prediction）**：DeepSeek 系列自带的多 token 预测头；
- **N-gram**：从上下文匹配候选（版本相关）。

配置示例：

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path ... \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 8
```

与所有引擎一样，收益取决于接受率；代码、结构化输出、重复性高的文本更容易加速。

### 4.8.2 缓存感知路由

单机多实例部署时，请求应该发到“缓存命中最高”的实例。SGLang 提供独立的 **router**（Rust 实现），支持：

- **cache-aware**：维护每个 worker 的前缀缓存近似视图，优先路由到匹配最长的实例；
- **power-of-two**：随机选两个，按负载/缓存选较好的；
- **round-robin / random**：简单基线。

这对多轮对话、Agent 集群很重要：如果请求被随机打到没有缓存的实例，前缀复用就白费了。

### 4.8.3 前端生态

除了 OpenAI API，SGLang 还提供：

- 原生 `/generate` 接口，支持 `regex`、`json_schema`、`ebnf` 等参数；
- Python `Engine` 类，用于离线批量推理；
- 与强化学习框架（如 verl）的 rollout 集成；
- 多模态模型支持（Qwen-VL、LLaVA、InternVL 等）；
- embedding/rerank 模型支持；
- 2025—2026 年还扩展出 Diffusion、Omni（TTS/ASR）等方向。

---

## 4.9 源码导读（概念版）

> 路径按 **SGLang 0.5.23** 的源码树核对（2026-11-25）；0.5.x 重构较频繁，找不到时用 `grep` 按类名定位。

| 模块 | 典型路径 | 看什么 |
|---|---|---|
| 启动入口 | `python/sglang/srt/entrypoints/engine.py` | Engine 初始化、参数 |
| HTTP 服务 | `python/sglang/srt/entrypoints/http_server.py` | OpenAI 路由、请求解析 |
| Tokenizer 管理 | `python/sglang/srt/managers/tokenizer_manager.py` | 输入处理、流式输出 |
| 调度器 | `python/sglang/srt/managers/scheduler.py` | 主循环、overlap、批处理 |
| 调度策略 | `python/sglang/srt/managers/schedule_policy.py` | LPM/FCFS/DFS-weight |
| Batch 结构 | `python/sglang/srt/managers/schedule_batch.py` | PrefillAdder、batch 元数据 |
| Radix 缓存 | `python/sglang/srt/mem_cache/radix_cache.py` | match/insert/evict/split |
| 内存池 | `python/sglang/srt/mem_cache/memory_pool.py` | TokenToKVPool、ReqToTokenPool |
| 分层缓存 | `python/sglang/srt/mem_cache/unified_radix_cache.py` | 统一 Radix 缓存与 HiCache 分层/回写（0.5.23 路径；旧版为 `hiradix_cache.py`） |
| ModelRunner | `python/sglang/srt/model_executor/model_runner.py` | 输入构造、forward、采样 |
| Attention 后端 | `python/sglang/srt/layers/attention/` | FlashInfer/FA3/FlashMLA |
| 投机解码 | `python/sglang/srt/speculative/` | EAGLE、draft、验证 |
| PD 分离 | `python/sglang/srt/disaggregation/` | prefill/decode 角色、KV 传输 |

阅读建议：

1. 先看 `scheduler.py` 的 `event_loop_overlap` / `event_loop_normal`，理解两种主循环；
2. 再看 `schedule_policy.py` 如何实现 FCFS、LPM 等策略；
3. 然后看 `radix_cache.py` 的 `match_prefix`、`insert`、`evict`；
4. 最后看 `model_runner.py` 如何把 radix 匹配到的 KV 索引喂给 attention backend。

---

## 4.10 优势与局限

### 优势

- **前缀复用做到 token 级**：当 `page_size=1` 时，任意长度的共享前缀都能精确命中，不受块对齐限制；`page_size>1` 时退化为页级；
- **缓存感知调度（可选）**：LPM/DFS-weight 主动提高命中率，前缀密集负载吞吐和 TTFT 表现好；需要显式开启，并注意公平性；
- **程序级优化**：前端语言表达 fork/select/约束，运行时可以针对性优化；
- **结构化输出强**：压缩 FSM + jump-forward，JSON/正则/EBNF 场景成熟；
- **大规模 MoE 经验**：DP attention、EP、EPLB、PD 分离，DeepSeek 类模型部署案例多；
- **CPU 侧优化**：overlap scheduler、增量 detokenize，高 QPS 小模型场景有优势；
- **性能迭代快**：与 FlashInfer、EAGLE 等社区结合紧密。

### 局限

- **模型/硬件覆盖不如 vLLM 广**：长尾模型、非 NVIDIA 后端、特殊量化格式可能支持滞后；
- **树维护有开销**：前缀完全不同的负载下，Radix Cache 可能只增加成本；
- **LPM 公平性风险**：低命中请求可能饿死，需要额外策略；
- **前端语言使用门槛**：大多数用户只用 OpenAI API，DSL 的生态与文档相对小众；
- **版本/参数变化快**：新特性多，默认值和 flag 在版本间可能变化，生产升级需要回归测试；
- **多租户隔离**：前缀缓存需要额外的安全考虑。

---

## 4.11 常见误区

**误区 1：有 RadixAttention 就一定比 vLLM 快。**  
前缀完全不共享、输出很长、缓存压力小时，树维护可能是负收益。性能取决于 workload，见第 5 章。

**误区 2：Radix 树的节点里存的是 KV 数据本身。**  
节点存的是“边上的 token 序列 + KV 池索引”，真正的 K/V 张量在统一的 token 池里。树只是索引结构。

**误区 3：前缀缓存可以共享任意相同的后缀。**  
只能共享“从开头开始”的前缀。两个请求如果中间不同、后面又相同，后面那段不能直接复用，因为注意力的上下文不同。

**误区 4：LPM 总是更好。**  
LPM 提高缓存命中，但可能饿死低命中请求，伤害 P99 和公平性。SGLang 提供 FCFS/HRRN 等策略就是为此。

**误区 5：SGLang 只是一个前端 DSL。**  
SGLang 现在主要指 SRT 运行时（serving 引擎）；前端语言是它的一部分历史和特色，但大多数用户直接用 OpenAI API。

**误区 6：结构化输出是“免费”的。**  
语法编译、FSM 状态、token mask 都有开销；复杂 schema 可能降低吞吐。压缩 FSM 和 jump-forward 是在降低这些开销，不是消除。

**误区 7：DP attention/EPLB 对所有模型都适用。**  
它们主要针对 DeepSeek 类 MLA + MoE 模型；普通稠密模型用不上，配置错了反而出问题。

---

## 4.12 本章小结

- SGLang = 前端语言（表达程序结构）+ 运行时 SRT（高性能 serving）。
- **RadixAttention** 把 KV Cache 的索引组织成压缩前缀树：token 级匹配、节点分裂、LRU 驱逐、引用计数；真正的 K/V 张量仍在内存池中。
- **缓存感知调度（可选）**：开启 LPM 后优先高命中请求，DFS-weight 进一步提升局部性；代价是公平性风险。
- **Overlap scheduler** 把 CPU 调度与 GPU 计算重叠，降低小模型/高 QPS 下的 CPU 瓶颈。
- **结构化输出**：XGrammar + 压缩 FSM + jump-forward，显著降低约束解码开销。
- **大规模能力**：DP attention、EP/EPLB、PD 分离、HiCache，支撑 DeepSeek 类模型与长会话场景。
- 优势在前缀密集、Agent/RAG、结构化输出、大规模 MoE；局限在覆盖广度、树开销与公平性。

> **用大白话再说一遍**：SGLang 把 KV 组织成一棵“笔记目录树”。相同前缀的请求走同一段路径，不用重新 prefill；哪份卷子要分叉，就在分叉点把那段笔记拆开。若开启缓存感知调度，还会优先安排“路径更长、命中更多”的请求，像先处理开头相同的卷子。结构化输出则是给模型一张“答题卡模板”，只允许填合法的格子，能确定的部分直接快进。

> **延伸阅读**：SGLang 论文（NeurIPS 2024）、LMSYS 的 RadixAttention 博客、SGLang 文档中的 [HiCache](https://docs.sglang.io/docs/advanced_features/hicache) / PD Disaggregation 章节，以及 `radix_cache.py`、`scheduler.py` 的源码。

---

## 思考题

1. 画出三个 prompt 共享 2000 token 前缀、各自有 100 token 后缀的基数树。插入第二个 prompt 时会发生节点分裂吗？为什么？
2. RadixAttention 为什么只驱逐叶子节点？如果允许驱逐内部节点，会破坏什么不变量？
3. LPM 调度在什么流量下会伤害尾延迟？请设计一个既能提高缓存命中、又避免饿死的调度策略。
4. Overlap scheduler 为什么需要“双缓冲”的 batch 状态？如果直接在当前 batch 上调度下一步，会有什么竞争条件？
5. 压缩 FSM 与 jump-forward 分别优化了约束解码的哪一部分开销？它们能叠加吗？
6. DP attention 与 TP 在 KV Cache 存储和通信模式上有什么本质区别？为什么 DeepSeek 类模型更适合 DP attention？
7. HiCache 的命中率提升在什么 workload 下最明显？如果网络带宽很低，开启 HiCache 可能反而变慢，为什么？
8. 用自己的话解释：RadixAttention 与 PagedAttention 是“互斥”还是“可组合”的？请给出一种组合方式。

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 5 章 核心机制正面对比：页表 vs 基数树，通用调度 vs 缓存感知 →](05-mechanism-comparison.md)
