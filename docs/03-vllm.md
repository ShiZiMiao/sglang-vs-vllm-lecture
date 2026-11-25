# 第 3 章 vLLM：以 PagedAttention 为核心的通用推理引擎

> **本章目标**：理解 vLLM 的核心数据结构（块表）、前缀缓存（块哈希）、调度器（token budget + chunked prefill）以及 V1 引擎的进程架构。读完后，你应该能画出一次请求在 vLLM 内部的完整路径。
>
> **先修知识**：[第 1 章](01-foundations.md) 的 KV Cache 公式、[第 2 章](02-engine-architecture.md) 的分页思想。如果忘了“块表”，本章 3.0 节会用储物柜类比重新讲。
>
> **一句话概括**：vLLM 把 KV Cache 切成固定大小的块，用一张“块表”把逻辑位置翻译成物理位置，从而实现按需分配、共享和前缀复用。
>
> **给零基础同学**：本章的关键不是背参数，而是理解“块表”和“块哈希”两个数据结构。看到 `block_table`、`slot_mapping` 时，回到 3.0 节的储物柜类比。
>
> **版本与来源**：本章的默认值与源码路径按 **vLLM 0.34.0** 核对（2026-11-25）；参数逐项说明见 [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.34.0/configuration/engine_args.html)。

---

## 3.0 先用储物柜类比理解 PagedAttention

假设你经营一个储物柜仓库：

- 每个储物柜有 16 个小格子（block_size = 16）；
- 客人来了，不一定能拿到连号的柜子，可能拿到 7 号、3 号、9 号；
- 你给每位客人一张**取物卡**，上面写着“第 0 段去 7 号柜、第 1 段去 3 号柜……”；
- 这张卡就是 **block table（块表）**；
- 柜子不需要连号，所以不会有“明明总空格够、却找不到连续 3 个柜子”的问题；
- 两个客人如果前几段东西完全一样，可以暂时共用柜子；谁要往里写新东西，再复制一份，这就是 **copy-on-write**。

PagedAttention 做的就是这件事：

| 储物柜 | vLLM |
|---|---|
| 柜子 | 物理块（physical block） |
| 16 个小格子 | block_size = 16 个 token 的 KV |
| 取物卡 | 块表（block table） |
| 客人的逻辑第几段 | 逻辑块号（logical block） |
| 两客人共用柜子 | 块共享 + 引用计数 |
| 写入时复制 | copy-on-write |
| 柜子回收 | 引用计数归零后回到空闲队列 |

有了这个类比，后面的“块表寻址”“前缀缓存”“COW”就不难了。

---

## 3.1 定位与历史

vLLM 诞生于 2023 年，由 UC Berkeley 的 Sky Computing Lab 发起，核心论文是 SOSP 2023 的 *Efficient Memory Management for Large Language Model Serving with PagedAttention*。

它要解决的核心问题是：

> 论文作者观察到：现有 LLM 服务系统把 KV Cache 存在连续显存里，为了应对未知的输出长度，往往要预留最大长度，在他们测试的负载上造成 60%～80% 的显存浪费；同时变长请求让批处理效率很低。

vLLM 的答案是把操作系统虚拟内存的分页机制搬到 KV Cache 上：

- KV 切成固定大小的 **block**；
- 每条序列维护 **block table**，逻辑块号映射到物理块号；
- 按需分配，块可以不连续；
- 支持块级共享与 copy-on-write；
- 注意力 kernel 通过块表寻址。

论文（SOSP 2023）报告，在 NVIDIA A100、OPT/LLaMA 系列模型的实验中，相对 FasterTransformer/Orca 有 2～4 倍吞吐提升（具体模型、长度与并发见论文第 6 节）；这是特定实验条件下的结果，不是跨版本的性能承诺。更重要的是，它把“PagedAttention + continuous batching”变成了开源社区的事实标准，后续几乎所有推理引擎都借鉴了这套思路。

今天的 vLLM 已经不只是“一个注意力优化”，而是一个覆盖文本、多模态、embedding、reward、分类模型，支持多种硬件、量化、并行与分离式部署的通用推理平台。2025 年起，**V1 引擎**逐步成为默认架构（vLLM 0.8 起默认启用），重构了调度、KV 管理和多进程模型。

---

## 3.2 设计总览

先看一张简化的全局图（图 3-1）：

```text
┌──────────────────────────────────────────────────────────────┐
│                    API Server Process                        │
│  FastAPI / OpenAI 协议 / 鉴权 / 限流 / tokenize / detokenize  │
└───────────────────────────┬──────────────────────────────────┘
                            │ ZMQ（请求、输出、控制）
┌───────────────────────────▼──────────────────────────────────┐
│                    EngineCore Process                        │
│  ┌─────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │ Scheduler   │──▶│ KVCacheManager   │   │ OutputProcessor│  │
│  │ 组批/抢占    │   │ 块池/前缀缓存     │   │ 停止/流式      │  │
│  └──────┬──────┘   └──────────────────┘   └───────────────┘  │
└─────────┼────────────────────────────────────────────────────┘
          │ 调度结果（SchedulerOutput）
┌─────────▼────────────────────────────────────────────────────┐
│                    GPU Worker Processes                      │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐   │
│  │ ModelRunner    │  │ Attention      │  │ Sampler       │   │
│  │ 构造输入/执行   │  │ Backend        │  │ 采样/约束      │   │
│  └────────────────┘  └────────────────┘  └───────────────┘   │
│  每张 GPU 一个 Worker；TP/PP/EP 由 Executor 组织              │
└──────────────────────────────────────────────────────────────┘
```

*图 3-1　vLLM V1 的进程架构*

几个关键点：

- **前后端分离**：API Server 负责协议与文本处理，EngineCore 负责调度与 KV 管理，GPU Worker 负责模型执行。
- **ZMQ 通信**：请求、调度输出、采样结果通过进程间消息传递。ZMQ 是一个消息队列库，可以把它想成“进程之间的快递员”；这带来隔离性与可扩展性，也带来 IPC（进程间通信）开销。
- **DP Coordinator**：当数据并行度 >1 时，额外的协调进程负责在多个 DP rank 之间分配请求、协调 MoE 的同步前向。
- **异步调度**：V1 的调度与执行可以流水化，减少 CPU 侧等待。

---

## 3.3 PagedAttention：把 KV Cache 变成“虚拟内存”

### 3.3.1 块与块表

设 `block_size = 16`（本章教学示例，便于与论文和旧资料对照）。vLLM 0.34.0 的默认值已是 64，历史版本长期为 16；实际以启动日志或 `--block-size` 为准。一条序列的第 $i$ 个 token 属于逻辑块：

$$
\text{logical\_block} = \left\lfloor \frac{i}{16} \right\rfloor
$$

先解释这个公式：

- $\lfloor \cdot \rfloor$ 是**向下取整**，例如 $\lfloor 2.7 \rfloor = 2$、$\lfloor 3.0 \rfloor = 3$；
- 这里 token 位置 $i$ 从 0 开始数（vLLM 内部用 0-based 索引）；
- 每 16 个 token 算一块：
  - $i=0 \sim 15$ → $\lfloor i/16 \rfloor = 0$，属于逻辑块 0；
  - $i=16 \sim 31$ → $\lfloor i/16 \rfloor = 1$，属于逻辑块 1；
  - $i=32 \sim 47$ → 逻辑块 2；
  - 依此类推。

如果 `block_size` 不是 16，把公式里的 16 换成 `block_size` 即可：

$$
\text{logical\_block} = \left\lfloor \frac{i}{\text{block\_size}} \right\rfloor
$$

所以“逻辑块号”回答的是：**这个 token 在序列的第几个块里**；“块表”再回答：**这个逻辑块实际放在显存的哪个物理块里**。

序列维护一张块表（图 3-2）：

```text
逻辑块:   0      1      2      3
         ┌────┐ ┌────┐ ┌────┐ ┌────┐
块表:    │ 7  │ │ 3  │ │ 9  │ │ 12 │
         └────┘ └────┘ └────┘ └────┘
           │      │      │      │
物理块池:  7      3      9      12
```

*图 3-2　逻辑块到物理块的块表映射*

物理块可以任意排列，甚至被多条序列共享。注意力 kernel 在计算第 $t$ 个 query 时：

1. 计算它属于哪个逻辑块；
2. 通过块表找到物理块；
3. 从物理块中读取对应的 K/V tile；
4. 对所有历史块做 attention。

**手把手例子（为了好算，假设 block_size = 4，见图 3-3）**：

一条序列已经生成了 10 个 token。逻辑上它需要 3 个块：

```text
token 位置:  0  1  2  3 | 4  5  6  7 | 8  9
逻辑块:      [   块 0   ] [   块 1   ] [块 2]
             4 个 token    4 个 token   2 个 token（还有 2 个空位）
```

*图 3-3　block_size=4 时的块内布局*

块表把它映射到物理块 7、3、9：

```text
逻辑块 0 → 物理块 7：存 token 0～3 的 K/V
逻辑块 1 → 物理块 3：存 token 4～7 的 K/V
逻辑块 2 → 物理块 9：存 token 8～9 的 K/V（9 号块还有 2 个空位）
```

如果下一步生成第 11 个 token（位置 10），它属于逻辑块 2，直接写入物理块 9 的空位；再生成第 12 个 token（位置 11）时，块 2 的 4 个槽位刚好填满；生成第 13 个 token（位置 12）时就要分配一个新物理块（比如 12 号），块表变成 `[7, 3, 9, 12]`。

**共享与 COW 的例子**（图 3-4）：

```text
序列 A 的块表: [7, 3, 9]
序列 B 的块表: [7, 3, 9]     ← A、B 的前 3 个块完全共享
```

*图 3-4　共享块与 copy-on-write*

现在 B 要往逻辑块 2 写入新 token，但物理块 9 还被 A 引用：

1. 分配一个新物理块 15；
2. 把 9 号块的内容复制到 15 号块；
3. B 的块表改成 `[7, 3, 15]`；
4. 9 号块的引用计数减 1（只剩 A 用）；
5. B 往 15 号块写新 token，不影响 A。

这就是“写时复制”：**只有真正要写的时候才复制，读的时候可以放心共享。**

伪代码：

```python
def paged_attention(query, block_table, kv_cache, block_size):
    output = 0
    for logical_block in range(num_blocks_for_query):
        physical_block = block_table[logical_block]
        k = kv_cache.k[physical_block]   # [block_size, num_kv_heads, head_dim]
        v = kv_cache.v[physical_block]
        output += attention(query, k, v)
    return output
```

真实 kernel 会做 tile 化、在线 softmax、向量化加载，但逻辑就是这样。

**别把 `block_table` 和 `slot_mapping` 混为一谈**：

- `block_table` 是“长期地图”：每条序列一张，记录逻辑块 → 物理块，decode 每一步都要用它读取历史 KV；
- `slot_mapping` 是“本步施工单”：只包含本步新写入的 token，记录每个新 token 的 KV 应该写到哪个物理槽位（通常是 `物理块号 × block_size + 块内偏移`）。

用上面的例子：序列块表是 `[7, 3, 9]`，本步要写入位置 10 的 token；它落在逻辑块 2、块内偏移 2，所以 `slot_mapping = 9 × 4 + 2 = 38`。**读历史靠 block_table，写新 KV 靠 slot_mapping**。第 9 章还会再看到这个例子。

### 3.3.2 显存浪费分析

两种方案的浪费对比如表 3-1 所示。

*表 3-1　连续分配与 PagedAttention 的显存浪费对比*

| 浪费类型 | 连续分配 | PagedAttention |
|---|---|---|
| 外部碎片 | 严重（释放后留下空洞） | 几乎没有（块大小固定） |
| 内部碎片 | 取决于预留策略，可能很大 | 每条序列最后一块最多浪费 15 个 token |
| 预留浪费 | 按 max_tokens 预留 | 按实际增长分配 |
| 共享 | 困难 | 块级共享 + COW |

论文（SOSP 2023，A100 等实验环境）的核心数据是：PagedAttention 把 KV 浪费从 60%～80% 降到 4% 以下。这个数字依赖具体负载与块大小，但方向是明确的。

**为什么能降到 4% 以下？** 因为分页之后，浪费主要只剩“每条序列最后一个块没装满”。块大小 16 时，最坏浪费 15 个 token，浪费比例大致是：

$$
\frac{15}{\text{平均上下文长度}}
$$

| 平均上下文长度 | 最坏内部碎片比例 | 说明 |
|---|---|---|
| 32 token | 约 47% | 短序列反而浪费严重 |
| 128 token | 约 12% | 仍然不可忽视 |
| 512 token | 约 2.9% | 接近论文量级 |
| 4096 token | 约 0.37% | 可以忽略 |

所以“<4%”不是魔法，而是**长上下文负载 + 16 token 块**下的结果。如果你的业务全是几十个 token 的短请求，块内碎片比例会更高；这时可以考虑更小的块，或者接受这部分开销。反过来，块太小会增加块表与 kernel 寻址开销，所以仍是折中。把块大小换成 0.34.0 默认的 64，趋势不变：长上下文的碎片比例更低，但几十 token 的短请求最坏会浪费大半个块，短请求场景反而更不友好。

### 3.3.3 共享与 Copy-on-Write

PagedAttention 天然支持“同一前缀，多条序列”：

```text
序列 A 块表: [7, 3, 9]
序列 B 块表: [7, 3, 9]   ← 前三个块共享，引用计数为 2
```

当 B 要继续生成 token 时，分三种情况：

- 块 9 还有空位，且只有 B 在用 → 直接写进块 9 的空位；
- 块 9 还有空位，但 A、B 共享 → 分配新块 15，把块 9 的已有内容复制过去，B 的块表改为 `[7, 3, 15]`，再往 15 的空位写；
- 块 9 已满 → **不需要复制块 9**：直接分配新块 15，B 的块表变为 `[7, 3, 9, 15]`，新 token 写入 15。

只有“共享块里还有空位、某条序列要往这个空位追加”时才需要复制；块已经满了就只是开新块，不涉及复制。

这就是 **copy-on-write（COW）**，与进程 fork 后的内存语义一致。它支撑了：

- `n > 1` 的并行采样（同一 prompt 生成多个回答）；
- beam search（多个候选共享前缀）；
- 前缀缓存命中（多个请求共享相同 system prompt 的块）。

### 3.3.4 块大小的权衡

块大小的取舍如表 3-2 所示。

*表 3-2　块大小的权衡*

| block_size | 优点 | 缺点 |
|---|---|---|
| 小（如 8/16） | 内部碎片小，前缀匹配粒度细 | 块表长，元数据多，kernel 寻址频繁 |
| 大（如 32/64） | 元数据少，kernel 效率高 | 内部碎片大，前缀复用粒度粗 |

vLLM 长期默认 16，从 0.34.0 起默认 64；两者都是工程折中。块大小通常不是首要调参项，但在前缀复用极度敏感的场景值得关注：0.34 还引入了 `prefix_match_unit`，允许前缀命中的粒度细于物理块。

---

## 3.4 KV Cache Manager：块池、哈希与驱逐

### 3.4.1 数据结构

vLLM 的 KV 管理器（V1 中为 `KVCacheManager`）大致维护：

```text
KVCacheManager
├── block_pool: list[KVCacheBlock]
│   └── KVCacheBlock { block_id, ref_cnt, block_hash, prev_free, next_free }
├── free_block_queue: 双向链表（LRU 顺序）
├── cached_block_hash_to_block: dict[hash → block]
└── req_to_blocks: dict[request_id → list[block]]
```

初始化时，根据 `gpu_memory_utilization` 与模型/激活开销算出可用显存，除以单块大小，得到块总数。

### 3.4.2 分配路径

调度器为新请求或 decode 追加 token 时，调用类似 `allocate_slots()` 的逻辑：

1. 计算需要的新块数；
2. 从空闲队列头部取块（头部是最久未使用的，适合驱逐）；
3. 如果空闲块不足，驱逐带缓存哈希的块（从哈希表移除）；
4. 如果仍不足，触发抢占；
5. 更新块表与引用计数。

### 3.4.3 前缀缓存的哈希

先不用看公式，先理解“哈希”是什么：

> 哈希（hash）就像给一段数据算一个“指纹”。同样的数据得到同样的指纹；不同的数据通常得到不同的指纹。指纹很短，方便快速比较。

vLLM 的前缀缓存默认开启（0.34.0 中 `CacheConfig.enable_prefix_caching` 为 `True`；模型/后端不支持时引擎会自动关闭并提示）。开启后，每个完整块都会参与下面的哈希。

vLLM 要给每个“完整块”算指纹，才能判断两个请求的块是不是一样、能不能复用。但只算本块内容还不够，因为：

```text
请求 A: [系统提示][问题 X]
请求 B: [系统提示][问题 Y]
```

如果只看“问题 X”和“问题 Y”所在的块，可能误以为它们相同；实际上前缀不同，不能复用。所以哈希必须把**父块的指纹**也带上（图 3-5）：

```text
块 0 指纹 = hash(块 0 的内容)
块 1 指纹 = hash(块 0 指纹, 块 1 的内容)
块 2 指纹 = hash(块 1 指纹, 块 2 的内容)
```

*图 3-5　块哈希沿父块链式累积*

这样，只有“从开头到本块为止，所有 token 都完全相同”，指纹才会相同。公式就是：

$$
h_{\text{block}} = \text{hash}\left(h_{\text{parent}},\; \text{tuple}(\text{block tokens}),\; \text{extra hashes}\right)
$$

- $h_{\text{parent}}$：前一个块的哈希，保证“只有前缀完全相同，块哈希才相同”；
- `block tokens`：本块的 token ID 元组；
- `extra hashes`：LoRA ID、多模态输入哈希、`cache_salt` 等。

几个重要细节：

- **只缓存完整块**：最后一个没填满的块不进入前缀缓存，因为它的内容还会继续增长，指纹不稳定（示例中块大小为 16，0.34.0 默认为 64）；
- **默认哈希算法**：`sha256` over pickle，也有 `sha256_cbor`、`xxhash`、`xxhash_cbor` 等更快/更稳定的选项；
- **`cache_salt`**：注入第一个块的哈希，用于租户隔离，防止通过缓存命中时间侧信道推断他人内容；
- **LRU 驱逐**：空闲队列头部是最久未使用的块；被命中时从队列中摘下（引用计数 +1），不再可驱逐；释放后按相反顺序放回尾部（后面的块更不可能被复用）。

> **LRU 的冰箱类比**：把空闲块想成冰箱里的食材。最久没被用到的放在最外面，空间不够时先扔它；刚被拿去做菜的，先放到里面，不会马上被扔。LRU = Least Recently Used，最近最少使用。

### 3.4.4 一次前缀命中的过程

请求的 prompt 有 2000 个 token，前 1984 个 token（124 个完整块）与缓存中的某个前缀相同：

1. 调度器调用 `get_computed_blocks()`，逐块计算哈希；
2. 在 `cached_block_hash_to_block` 中查找；
3. 命中的块引用计数 +1，从空闲队列移除；
4. 返回“已计算 token 数”= 1984；
5. 只需为剩余 16 个 token 做 prefill（以及后续 decode）；
6. 新生成的完整块在写入后注册哈希，供后续请求复用。

注意：vLLM V1 的块表是 append-only 的，某些情况下同一哈希可能暂时对应多个块，直到请求结束才清理。这是实现细节，不影响“块级前缀缓存”的核心语义。

---

## 3.5 调度器：Token Budget 与 Chunked Prefill

### 3.5.1 核心约束

vLLM 调度器每步受两个主要约束：

- `max_num_batched_tokens`：本步所有序列处理的 token 总数上限（0.34.0 字段默认 2048）；
- `max_num_seqs`：本步同时运行的序列数上限（0.34.0 字段默认 128）。

前者决定“计算量/激活峰值”，后者决定“并发宽度”。两者共同影响吞吐与延迟。注意默认值只是字段值，真实服务的有效并发还会被显存与 KV 池限制。

### 3.5.2 调度顺序（概念版）

V1 的调度循环可以简化为：

```python
def schedule(self):
    scheduled = []
    budget = self.max_num_batched_tokens

    # 1. 先处理 running 队列：decode 或继续 chunked prefill
    for req in self.running:
        if budget <= 0:
            break
        n = 1 if req.is_decode else min(req.remaining_prefill, chunk_size, budget)
        if self.kv_manager.can_append(req, n):
            scheduled.append((req, n))
            budget -= n
        else:
            self.preempt(req)   # 显存不足，踢回 waiting

    # 2. 再处理 waiting 队列：准入新的 prefill
    for req in self.waiting:
        if budget <= 0:
            break
        n = min(req.num_prompt_tokens, budget, chunk_size)
        blocks = self.kv_manager.allocate(req, n)
        if blocks is None:
            break
        scheduled.append((req, n))
        budget -= n

    return SchedulerOutput(scheduled, budget_used=...)
```

真实实现要复杂得多，但决策顺序基本是：**running 优先，waiting 补充，token budget 兜底，显存不足则抢占**。

> 伪代码里的 `chunk_size` 是示意变量；真实 vLLM 由 `max_num_batched_tokens` 与内部 chunking 配置共同决定每次 prefill 处理多少 token，并没有一个同名开关需要你去设置。

### 3.5.3 Chunked Prefill

当一个 prompt 很长（例如 32K token），而 `max_num_batched_tokens` 只有 8K 时，vLLM 会把它切成多个 chunk（图 3-6）：

```text
Step 1: A[0:8192]    + B decode + C decode
Step 2: A[8192:16384] + B decode + C decode
Step 3: A[16384:24576]+ B decode + C decode
Step 4: A[24576:32768]+ B decode + C decode + D prefill
```

*图 3-6　chunked prefill 的分步执行*

这样做的收益：

- 单步激活峰值受控，不会因为一个超长 prompt OOM；
- decode 请求不会被一个长 prefill 阻塞太久，ITL 更平稳；
- prefill 的计算密集与 decode 的带宽密集可以在同一 batch 内互补，提高 GPU 利用率。

代价：

- A 的 TTFT 变长（要分四步）；
- 调度器需要跟踪每条序列“prefill 到哪了”，并在 chunk 之间保持 KV 一致；
- chunk 大小需要调参：太小 → 调度开销大、TTFT 长；太大 → ITL 抖动、激活峰值高。

### 3.5.4 抢占：Recompute

当 KV 池不足时，vLLM V1 主要采用 **recompute**：

1. 选中要抢占的序列（V1 通常从 running 队列尾部选择，即优先级最低或最晚被调度的序列）；
2. 释放它的 KV 块；
3. 把它放回 waiting 队列，标记“需要重新 prefill”；
4. 之后重新调度时，从 prompt 开头重新计算（包括已经生成的部分）。

代价是浪费算力，但实现简单，且不需要 CPU↔GPU 的 KV swap。对于“短请求多、长请求少”的负载，recompute 通常可以接受；对于超长上下文，频繁抢占会很痛。

### 3.5.5 调度策略

- **FCFS**：默认，按到达顺序；
- **Priority**：按请求优先级；适合多租户、在线/离线混跑；
- 未来/版本相关的策略可能还包括更细的配额与公平性机制。

vLLM 没有内置“按前缀匹配长度重排 waiting 队列”的策略；前缀缓存仍然生效（块级哈希），但调度顺序本身不主动追求缓存命中率。SGLang 提供 LPM 等**可选**的缓存感知策略（注意：SGLang 当前默认同样是 FCFS，需要显式开启），这是第 5 章要重点对比的能力差异。

---

## 3.6 V1 引擎架构

### 3.6.1 进程模型

V1 把引擎拆成四类进程：

| 进程 | 职责 |
|---|---|
| **API Server Process** | HTTP/OpenAI 协议、tokenize、多模态预处理、detokenize、流式响应 |
| **EngineCore Process** | 调度器、KV 管理器、请求生命周期 |
| **GPU Worker Process** | 每张 GPU 一个，加载权重、执行 forward、采样 |
| **DP Coordinator Process** | DP>1 时协调请求分配与 MoE 同步前向 |

进程间通过 ZMQ 传递消息。这样做的好处：

- 前端崩溃不影响引擎核心；
- 前端可以独立扩缩容；
- 调度与执行解耦，便于异步流水；
- 多 DP rank 时可以独立扩展。

代价是 IPC 序列化/反序列化开销，尤其是高 QPS 小请求场景。V1 通过批量消息、异步发送、减少拷贝等方式缓解。

### 3.6.2 一次请求的调用链

调用链如图 3-7 所示：

```text
HTTP POST /v1/chat/completions
  → api_server.py：解析 OpenAI 请求
  → Processor：套 chat template、tokenize、多模态预处理
  → AsyncLLMEngine.add_request()
  → ZMQ 发送到 EngineCore
  → Scheduler.schedule()：分配 KV、组批
  → Executor.execute_model(SchedulerOutput)
  → GPU Worker：ModelRunner.execute_model()
      → 构造 input_ids/positions/slot_mapping/block_tables
      → Model.forward()
      → Attention Backend
      → Sampler
  → EngineCoreOutput → ZMQ 回传
  → OutputProcessor：停止判断、detokenize
  → SSE 流式返回
```

*图 3-7　vLLM 一次请求的调用链*

### 3.6.3 异步与重叠

V1 的调度器是异步的：API Server 可以持续接收新请求，EngineCore 在每个 step 开始时把新请求纳入 waiting 队列；输出也可以在前一步 GPU 计算尚未完全结束时就开始处理。

不过，vLLM 的“CPU/GPU 重叠”程度与具体版本、后端有关。SGLang 把“零开销批调度”作为核心卖点之一，第 4 章会详细讲。

---

## 3.7 执行层：Attention Backend、CUDA Graph 与编译

### 3.7.1 Attention Backend

vLLM 支持多种注意力后端，按模型、硬件、dtype、功能选择：

| 后端 | 特点 |
|---|---|
| **FlashAttention**（FA2/FA3） | 成熟、通用，支持变长与 causal；Hopper 上 FA3 更快 |
| **FlashInfer** | 高性能分页注意力 kernel，支持多种 mask/量化；与 SGLang 同源社区 |
| **Triton** | 可移植、易改，性能通常略逊于专用 kernel |
| **FlexAttention** | 用 PyTorch 组合表达注意力变体，灵活 |
| **MLA 专用** | FlashMLA / CUTLASS MLA，针对 DeepSeek 类压缩 KV |
| **Mamba/线性注意力** | 混合架构模型 |

选择逻辑通常考虑：

- 是否支持 paged KV；
- 是否支持 sliding window / ALiBi / 自定义 mask；
- 是否支持 FP8 KV；
- head_dim、GQA 比例、是否 MLA；
- 当前 GPU 架构（Ampere/Hopper/Blackwell）。

可以用 `--attention-backend` 强制指定，但大多数时候让引擎自动选择更好。

### 3.7.2 CUDA Graph

Decode 每步要启动几十到上百个 kernel。每个 kernel launch 都有 CPU 开销（几微秒），在 batch 小、模型小时，CPU 可能成为瓶颈。**CUDA Graph** 把一串 kernel 调用捕获成图，之后一次 launch 重放，大幅降低 CPU 开销。

vLLM 的做法：

- 为一批固定的 batch size 捕获 decode 图（常见序列为 1, 2, 4, 8, …；旧 runner 默认把捕获上限压在 512，数据中心 Blackwell 可到 1024）；
- 实际 batch 不匹配时，padding 到最近的捕获尺寸；
- 图捕获需要固定显存地址，因此 KV 池与输入缓冲区要预分配；
- `--enforce-eager` 可关闭 CUDA Graph，便于调试，但会损失性能。

> **版本提醒**：0.34.0 的代码中已经包含 Model Runner V2；在 V2 下，捕获形状改为按 token 数管理，`max_cudagraph_capture_size` 属于旧 runner 的参数（已标记为 deprecated）。如果你在日志里看到 token 级捕获配置，不是配置写错了，而是走的新 runner。

V1 还使用 **piecewise CUDA Graph / torch.compile**：把模型切成若干段，在注意力等动态形状处断开，其余部分编译/捕获。这样既能享受图执行的低开销，又能处理变长输入。

### 3.7.3 采样

采样在 GPU 上完成：

- logits → 温度/top-k/top-p/penalty → 采样；
- 约束解码时，用 FSM 生成 token mask，把非法 token 的 logit 置为 $-\infty$；
- 只把选中的 token ID 传回 CPU，避免传输整个词表。

vLLM 支持多种 guided decoding 后端（当前以 xgrammar、guidance 为主，outlines 视版本），通过 `guided_json`、`guided_regex`、`guided_choice` 或 OpenAI 的 `response_format` 使用。

---

## 3.8 高级能力

### 3.8.1 投机解码（Speculative Decoding）

用一个小的 draft 模型（或 n-gram、EAGLE 头、MTP 头）先猜 $k$ 个 token，再用目标模型一次前向验证。若猜对，一次前向产出多个 token；若猜错，按拒绝采样修正，保证输出分布不变。

vLLM 支持多种方法：

- draft model；
- n-gram（从 prompt/上下文中匹配候选）；
- EAGLE / EAGLE-3；
- MTP（DeepSeek 的多 token 预测头）。

收益取决于**接受率**：草稿越准、任务越可预测（如代码补全、格式化输出），加速越明显；开放式创作可能几乎不赚。

### 3.8.2 LoRA

多租户场景下，为每个用户挂一个 LoRA adapter，共享基座模型。vLLM 支持：

- `--enable-lora`；
- `--max-loras`（同时激活的 adapter 数）；
- `--max-lora-rank`；
- 动态加载/卸载 adapter。

底层需要 batched LoRA kernel：同一个 batch 里不同序列使用不同 adapter，GEMM 要按 adapter 分组。

### 3.8.3 多模态

vLLM 支持图像、视频、音频输入，流程通常是：

1. 输入 processor 把图像切成 patch；
2. 视觉编码器（ViT）产出 embedding；
3. 通过投影层映射到语言模型维度；
4. 与文本 token 拼接；
5. 多模态 embedding 也参与前缀缓存（哈希中包含图像哈希）。

多模态的显存与计算开销主要在编码器，推理引擎需要做 encoder cache、输入去重等优化。

### 3.8.4 量化

常见格式：

- **FP8**（W8A8 / W8A16）：Hopper/Blackwell 原生支持，精度损失小；
- **AWQ / GPTQ**：4-bit 权重量化，适合显存受限；
- **INT8**：W8A8；
- **GGUF**：社区格式，CPU/混合推理常用；
- **BitsAndBytes**：易用但性能一般；
- **KV Cache 量化**：FP8 KV，直接把 KV 显存减半。

量化对 decode 尤其有效，因为它直接降低“每 token 要读的字节数”。但要注意 kernel 支持、校准数据、精度回归。

### 3.8.5 并行与分布式

- **TP**：`--tensor-parallel-size`；
- **PP**：`--pipeline-parallel-size`；
- **DP**：`--data-parallel-size`，配合 DP Coordinator；
- **EP**：MoE 专家并行，通常与 DP/TP 组合；
- **Ray**：多机执行后端；
- **PD 分离**：prefill/decode 分开部署，通过 KV connector（NIXL、LMCache 等）传输 KV。

### 3.8.6 其他

- **Embedding / reward / classification**：池化模型支持；
- **Sleep mode**：释放 GPU 显存，按需唤醒；
- **Metrics**：Prometheus `/metrics`，包含队列长度、cache 命中、TTFT/TPOT 等；
- **Profiling**：`torch.profiler`、NVIDIA Nsight 集成；
- **OpenAI 兼容**：chat/completions、completions、embeddings、models 等。

---

## 3.9 源码导读（概念版）

> 路径按 **vLLM 0.34.0** 的源码树核对（2026-11-25）；版本更新后可能移动，建议用 `grep` 按类名定位。

| 模块 | 典型路径 | 看什么 |
|---|---|---|
| HTTP 入口 | `vllm/entrypoints/openai/api_server.py` | 路由、请求解析、流式响应 |
| 异步引擎 | `vllm/v1/engine/async_llm.py` | 请求提交、输出流 |
| 引擎核心 | `vllm/v1/engine/core.py` | EngineCore 主循环、ZMQ |
| 输入处理 | `vllm/v1/engine/input_processor.py` | tokenize、多模态、采样参数 |
| 调度器 | `vllm/v1/core/sched/scheduler.py` | schedule()、抢占、token budget |
| KV 管理 | `vllm/v1/core/kv_cache_manager.py` | 分配、前缀命中、驱逐 |
| 块池 | `vllm/v1/core/block_pool.py` | 空闲队列、引用计数、哈希表 |
| 哈希工具 | `vllm/v1/core/kv_cache_utils.py` | 块哈希、cache_salt |
| Worker | `vllm/v1/worker/gpu_worker.py` | 初始化、显存分析 |
| ModelRunner | `vllm/v1/worker/gpu/model_runner.py` | 构造输入、执行、采样（0.34.0 路径；旧版为 `gpu_model_runner.py`） |
| 注意力后端 | `vllm/v1/attention/backends/` | FlashAttention/FlashInfer/Triton/MLA |
| 模型实现 | `vllm/model_executor/models/` | 各模型 forward |
| 分布式 | `vllm/distributed/` | TP/PP/EP 通信 |

阅读建议：

1. 先读 `scheduler.py` 的 `schedule()`，画出“running/waiting/preempt”三条路径；
2. 再读 `kv_cache_manager.py` 的 `get_computed_blocks()` 与 `allocate_slots()`；
3. 然后读 `gpu/model_runner.py` 的 `execute_model()`，看 `SchedulerOutput` 如何变成 GPU 张量；
4. 最后挑一个 attention backend，看块表如何参与 kernel 寻址。

---

## 3.10 优势与局限

### 优势

- **通用性强**：模型覆盖广，文本/多模态/embedding/reward 都能跑；
- **硬件与量化生态广**：NVIDIA、AMD、TPU 等，FP8/AWQ/GPTQ/INT8 等；
- **工程成熟**：V1 进程架构、异步调度、可观测性、生产部署经验丰富；
- **默认性能好**：大多数模型开箱即用，社区调优充分；
- **生态大**：文档、issue、集成（Ray、KServe、llm-d 等）丰富。

### 局限

- **前缀复用是块级的**：最后不满一块的前缀无法复用；块哈希有计算开销；
- **默认调度不感知前缀长度**：FCFS 可能错过高复用机会；SGLang 提供可选的缓存感知调度（需显式开启），前缀密集负载下可能更有优势；
- **功能碎片化**：大量特性通过 flag 组合，版本间行为可能变化；
- **Python/进程开销**：高 QPS 小模型场景需要仔细调优（CUDA Graph、异步、进程数）；
- **不同后端性能差异大**：同一模型在不同 attention backend/量化格式下可能差很多，需要实测。

---

## 3.11 常见误区

**误区 1：PagedAttention 是一种量化或压缩算法。**  
不是。它不改变 KV 的数值精度，而是改变 KV 的**存放方式**：用固定块 + 块表管理，减少碎片、支持共享。量化是另一条线。

**误区 2：块表就是操作系统页表，完全一样。**  
思想相同，细节不同：KV 块只追加、通常不换出（V1 以 recompute 为主）、块大小受 attention kernel 约束、共享语义是“前缀”而不是任意内存页。

**误区 3：前缀缓存只缓存完整块，所以没什么用。**  
长前缀、块对齐的场景命中率很高；块边界损失通常很小（块越小损失越小，块大小依版本为 16/64）。短前缀、非对齐场景才会明显吃亏。

**误区 4：Copy-on-Write 每次写入都会复制。**  
只有在“共享块需要被某条序列继续写入”时才复制；只读共享不复制。复制的是整块（示例中 16 token），不是整个序列。

**误区 5：block_size 越小越好。**  
块小 → 内部碎片小、前缀粒度细，但块表更长、kernel 寻址更频繁、元数据更多。16 是工程折中，不是越小越优。

**误区 6：vLLM 的调度就是先来先服务。**  
FCFS 是默认顺序，但调度器还要做 token budget、chunked prefill、抢占、前缀缓存命中、优先级等决策。

**误区 7：V1 只是 V0 改了个名字。**  
V1 重构了进程架构、调度循环、KV 管理和执行路径，是一个较大的架构升级；具体差异以版本发布说明为准。

---

## 3.12 本章小结

- vLLM 的核心是 **PagedAttention**：固定块 + 块表 + 按需分配 + 共享/COW。
- 前缀缓存是**块级哈希**：`hash(parent, block_tokens, extras)`，只缓存完整块，LRU 驱逐，`cache_salt` 做隔离。
- 调度器以 **token budget + max_num_seqs** 为约束，running 优先、waiting 补充，支持 chunked prefill 与 recompute 抢占。
- V1 把引擎拆成 API Server / EngineCore / GPU Worker / DP Coordinator，ZMQ 通信，异步流水。
- 执行层支持多种 attention backend、CUDA Graph、torch.compile、GPU 采样、guided decoding。
- 高级能力：投机解码、LoRA、多模态、量化、TP/PP/DP/EP、PD 分离。
- 优势是通用与成熟；局限是块级前缀复用与默认不感知前缀的调度。

> **用大白话再说一遍**：vLLM 把 KV 当成仓库里的储物柜，柜子不要求连号；每位客人有一张块表，写着“第几段去几号柜”。相同前缀的客人可以暂时共用柜子，谁要写新东西再复制。前缀缓存就是“给每段柜子算指纹”，下次遇到相同前缀直接复用。V1 则是把前台、调度室、厨房拆成不同房间，让它们能并行工作。

> **延伸阅读**：PagedAttention 论文（SOSP 2023）、vLLM 文档中的 [PagedAttention](https://docs.vllm.ai/en/v0.34.0/design/paged_attention.html) / [Prefix Caching](https://docs.vllm.ai/en/v0.34.0/design/prefix_caching.html) 章节，以及 `scheduler.py`、`kv_cache_manager.py` 的源码。

---

## 思考题

1. 为什么 vLLM 的前缀缓存只缓存“完整块”？如果允许缓存不完整块，会带来什么一致性问题？
2. 块大小从 16 改成 8，前缀复用粒度变细，但元数据和 kernel 开销增加。请定量分析一个“prompt 长度 1000、共享前缀 999”的场景，两种块大小的命中 token 数分别是多少？
3. 画出 `allocate_slots()` 在“空闲块不足”时的决策流程：驱逐、抢占、重试分别发生在什么条件？
4. Chunked prefill 的 chunk 大小如何影响 TTFT、ITL 和激活峰值？请给出一个调参思路。
5. 为什么 V1 要把 API Server 和 EngineCore 拆成两个进程？这种拆分在高 QPS 下可能带来什么新问题？
6. CUDA Graph 为什么要求固定 batch size？vLLM 如何用 padding 和 piecewise graph 处理动态形状？
7. 投机解码在什么任务上收益最大？如果接受率只有 30%，还可能加速吗？为什么？
8. 用自己的话解释：PagedAttention 的“块表”与操作系统页表的异同。

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 4 章 SGLang：以 RadixAttention 与程序级优化为核心的推理栈 →](04-sglang.md)
