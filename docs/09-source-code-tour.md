# 第 9 章 源码地图：一次请求在两个引擎中的调用链

> **本章目标**：把前八章的概念映射到真实代码。你不必读完两个项目的全部源码，但应该知道“调度、KV 管理、模型执行、输出处理”分别在哪里，以及如何用调试工具定位问题。
>
> **先修知识**：第 2 章的七大组件、第 3/4 章的核心数据结构。没读过源码也没关系，本章会教你“按调用链读”，而不是从头读到尾。
>
> **一句话概括**：先锁定一条请求，跟着它从 HTTP 进入调度器、KV 管理、ModelRunner，再回到输出；沿途用 `grep` 找类、用日志找状态。
>
> **给零基础同学**：读源码不是“看懂每一行”，而是“回答一个具体问题”，例如“前缀命中后，block table 在哪里被更新？”带着问题读，效率最高。

> **路径提醒**：两个项目重构频繁。下面的路径按 **vLLM 0.34.0 / SGLang 0.5.23** 核对（2026-11-25），请以你锁定的版本为准，用 `grep`/`rg` 按类名定位；类名通常比文件路径稳定。

---

## 9.0 读源码前的三个准备

### 9.0.1 先选一个“小目标”

不要一上来就读完整个引擎。选一个具体问题，例如：

- 一个请求从 HTTP 到进入 waiting 队列，经过哪些函数？
- 前缀缓存命中后，`block_table` 在哪里被填？
- decode 一步的 `input_ids` 是怎么构造的？
- 停止字符串跨 token 时，在哪里判断？

有了问题，读代码就有“主线”，不会在工具函数里迷路。

### 9.0.2 用 `grep`/`rg` 按类名定位

```bash
# 找类定义
rg "class Scheduler" vllm/
rg "class RadixCache" python/sglang/

# 找函数调用
rg "match_prefix\(" python/sglang/
rg "allocate_slots\(" vllm/

# 找日志/指标
rg "prefix.*hit|hit.*prefix" vllm/ python/sglang/
```

先看类定义和公开方法，再看调用点；不要一开始就钻进 kernel。

### 9.0.3 画出调用链，而不是抄目录

读到一个函数时，问三个问题：

1. 谁调用它？（输入从哪来）
2. 它调用了谁？（输出到哪去）
3. 它修改了哪些状态？（请求、KV、队列）

把答案画成箭头图，比抄目录有用得多。本章 9.2/9.3 的调用链就是这种图。

---

## 9.1 读源码的方法

面对几十万行的推理引擎，直接从头读会迷路。建议按下面的顺序：

1. **先锁版本**：`git checkout <tag>`，记录 commit；不要读 main 的最新代码却跑旧版本。
2. **从小模型开始**：Qwen2.5-0.5B/1.5B 这类模型结构简单，调度与 KV 逻辑与大模型一致。
3. **先读调度器**：它决定了“每一步做什么”，是理解全局的入口。
4. **再读 KV 管理**：它是两个引擎差异最大的部分。
5. **然后读 ModelRunner**：看逻辑 batch 如何变成 GPU 张量。
6. **最后读 attention backend**：不必一开始就陷入 kernel 细节。
7. **配合测试**：单元测试往往比文档更准确地描述不变量。
8. **加日志/断点**：用 request_id 追踪一条请求的完整路径。
9. **用 profiler**：`py-spy` 看 CPU 热点，`nsys`/`torch.profiler` 看 GPU kernel。

---

## 9.2 vLLM 的目录地图

```text
vllm/
├── entrypoints/
│   ├── llm.py                     # 离线 LLM 类
│   └── openai/
│       └── api_server.py          # OpenAI HTTP 服务
├── v1/
│   ├── engine/
│   │   ├── async_llm.py           # AsyncLLM：在线引擎入口
│   │   ├── core.py                # EngineCore：调度主循环
│   │   ├── core_client.py         # 前端到 EngineCore 的客户端
│   │   ├── input_processor.py     # 输入处理：tokenize/多模态
│   │   └── output_processor.py    # 输出处理：停止/detokenize
│   ├── core/
│   │   ├── sched/
│   │   │   └── scheduler.py       # 调度器
│   │   ├── kv_cache_manager.py    # KV 分配/前缀命中/驱逐
│   │   ├── block_pool.py          # 块池、空闲队列、引用计数
│   │   └── kv_cache_utils.py      # 块哈希、cache_salt
│   ├── worker/
│   │   ├── gpu_worker.py          # 每卡 Worker：初始化/显存
│   │   └── gpu/model_runner.py    # 构造输入、执行、采样（0.34 路径）
│   ├── attention/
│   │   └── backends/              # FlashAttention/FlashInfer/Triton/MLA
│   ├── spec_decode/               # 投机解码
│   └── request.py                 # Request 状态
├── model_executor/
│   ├── models/                    # 各模型 forward
│   └── layers/                    # 线性层、注意力层、量化层
├── distributed/                   # TP/PP/EP 通信
└── config/                        # 配置与参数校验
```

### 9.2.1 一次请求的调用链

完整调用链如图 9-1 所示：

```text
POST /v1/chat/completions
  │
  ▼
vllm/entrypoints/openai/api_server.py
  │  解析 OpenAI 请求，构造 ChatCompletionRequest
  ▼
OpenAIServingChat.create_chat_completion()
  │  套 chat template、校验参数
  ▼
AsyncLLM.generate()
  │  生成 request_id，注册输出队列
  ▼
EngineCore.add_request()          ← 跨进程 ZMQ
  │  请求进入 waiting 队列
  ▼
Scheduler.schedule()              ← 每个 iteration 调用
  │  产出 SchedulerOutput：
  │    - 本步要跑的请求
  │    - 每条请求的 num_scheduled_tokens
  │    - 新分配的 KV blocks
  ▼
Executor.execute_model(SchedulerOutput)
  │
  ▼
GPUWorker.execute_model()
  │
  ▼
GPUModelRunner.execute_model()
  │  构造 input_ids / positions / slot_mapping / block_tables
  │  调用 model.forward()
  ▼
Model.forward() → AttentionBackend → Sampler
  │
  ▼
EngineCoreOutput
  │  ZMQ 回传
  ▼
OutputProcessor.process_outputs()
  │  停止判断、增量 detokenize
  ▼
AsyncLLM 输出队列 → SSE 流
```

*图 9-1　vLLM 一次请求的完整调用链*

### 9.2.2 关键数据结构

**Request**（`vllm/v1/request.py` 附近）：

```python
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    num_computed_tokens: int      # 已经算过 prefill 的 token 数
    num_tokens: int               # prompt + 已生成
    block_ids: list[int]          # 逻辑块 → 物理块
    status: RequestStatus         # WAITING/RUNNING/FINISHED
```

**SchedulerOutput**：调度器与执行器之间的契约，包含本步的请求、token 数、KV 块分配、已完成请求等。不同版本字段会变，但核心语义稳定。

**KVCacheBlock**：

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int
    block_hash: int | None
    prev_free: KVCacheBlock | None
    next_free: KVCacheBlock | None
```

### 9.2.3 调度器伪代码

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_batched_tokens
    scheduled_new = []
    scheduled_running = []

    # 1. running 优先：decode 或继续 chunked prefill
    for request in list(self.running):
        if token_budget <= 0:
            break
        num_new_tokens = request.num_tokens - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens
        )
        if new_blocks is None:
            self._preempt(request)      # 显存不足：抢占
            continue

        scheduled_running.append((request, num_new_tokens))
        token_budget -= num_new_tokens

    # 2. waiting 补充：按 FCFS/priority 准入
    while self.waiting and token_budget > 0:
        request = self.waiting[0]
        num_new_tokens = min(
            request.num_prompt_tokens - request.num_computed_tokens,
            token_budget,
        )
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens
        )
        if new_blocks is None:
            break

        self.waiting.popleft()
        self.running.append(request)
        scheduled_new.append((request, num_new_tokens))
        token_budget -= num_new_tokens

    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new,
        scheduled_running_reqs=scheduled_running,
        num_scheduled_tokens={...},
        ...
    )
```

真实代码里还有：

- prefix caching 的 `get_computed_blocks()`；
- speculative decoding 的额外 token；
- encoder inputs（多模态）；
- `finished_req_ids`、`free_encoder_input_ids`；
- 结构化输出的 grammar 状态。

但“running 优先、waiting 补充、budget 兜底、不足则抢占”的主干不变。

### 9.2.4 KV 管理的关键函数

| 函数 | 作用 |
|---|---|
| `get_computed_blocks(request)` | 计算块哈希，查前缀缓存，返回命中的块 |
| `allocate_slots(request, num_tokens)` | 为新 token 分配块；不足时驱逐/抢占 |
| `free(request)` | 释放请求占用的块，引用计数减 1 |
| `cache_full_blocks(request)` | 把写满的块注册到哈希表 |
| `evict_blocks(num_blocks)` | 从空闲队列头部驱逐带哈希的块 |

阅读时可以重点看：

- 命中块如何从空闲队列摘下（避免被驱逐）；
- 分配失败时如何驱逐；
- 引用计数何时加减；
- `cache_salt` 如何进入哈希。

### 9.2.5 ModelRunner 的关键步骤

```python
def execute_model(scheduler_output):
    # 1. 更新请求状态（token、块表）
    self._update_states(scheduler_output)

    # 2. 构造输入张量
    input_ids = [...]          # 本步所有 token
    positions = [...]          # 每条序列的位置
    slot_mapping = [...]       # token → KV 物理槽位
    block_tables = [...]       # 每条序列的块表

    # 3. 构造 attention metadata
    attn_metadata = self.attn_backend.make_metadata(...)

    # 4. 前向
    hidden = self.model(input_ids, positions, attn_metadata)

    # 5. 采样
    logits = self.model.compute_logits(hidden)
    sampled = self.sampler(logits, sampling_metadata)

    return sampled
```

`slot_mapping` 是理解 PagedAttention 的关键：它把每个 token 映射到 KV 池中的具体槽位，写入时用；`block_tables` 则用于读取历史 KV。

**一个小例子**。假设 `block_size = 4`，序列 A 的块表是 `[7, 3, 9]`，对应逻辑块 0、1、2：

```text
逻辑块 0 → 物理块 7：token 0～3
逻辑块 1 → 物理块 3：token 4～7
逻辑块 2 → 物理块 9：token 8～11（当前只用到 8、9）
```

现在 decode 要写入位置 10 的新 token（也就是第 11 个 token）：

$$
\begin{aligned}
\text{logical block} &= \left\lfloor \frac{10}{4} \right\rfloor = 2 \\
\text{offset in block} &= 10 \bmod 4 = 2 \\
\text{physical block} &= \text{block table}[2] = 9 \\
\text{physical slot} &= 9 \times 4 + 2 = 38
\end{aligned}
$$

所以这个 token 的 `slot_mapping = 38`：K/V 写入 KV 池的第 38 号槽位。`block_tables` 则告诉注意力 kernel：“这条序列的历史块是 7、3、9”，读取时按块表逐块取 K/V。

> 注意：不同后端可能把 KV 池组织成 `[num_blocks, block_size, ...]` 或扁平化的 `[num_slots, ...]`，槽位编号的具体算法可能不同；但“逻辑位置 → 物理块 → 块内偏移 → 物理槽位”这条链是一样的。

---

## 9.3 SGLang 的目录地图

```text
python/sglang/
├── srt/
│   ├── entrypoints/
│   │   ├── engine.py              # 离线 Engine
│   │   └── http_server.py         # OpenAI/原生 HTTP
│   ├── managers/
│   │   ├── scheduler.py           # 调度主循环
│   │   ├── schedule_batch.py      # Req / ScheduleBatch / PrefillAdder
│   │   ├── schedule_policy.py     # LPM/FCFS/DFS-weight
│   │   ├── tokenizer_manager.py   # 输入处理
│   │   ├── detokenizer_manager.py # 增量 detokenize
│   │   └── tp_worker.py           # 每卡 Worker
│   ├── mem_cache/
│   │   ├── radix_cache.py         # RadixCache/TreeNode
│   │   ├── memory_pool.py         # TokenToKVPool/ReqToTokenPool
│   │   └── unified_radix_cache.py # 统一 Radix 缓存/HiCache（0.5.23 路径）
│   ├── model_executor/
│   │   └── model_runner.py        # 输入构造/forward/采样
│   ├── layers/
│   │   └── attention/             # FlashInfer/FA3/FlashMLA
│   ├── models/                    # 各模型实现
│   ├── speculative/               # EAGLE/draft/MTP
│   ├── disaggregation/            # PD 分离
│   └── distributed/               # TP/DP/EP
└── lang/                          # 前端 DSL
```

### 9.3.1 一次请求的调用链

完整调用链如图 9-2 所示：

```text
POST /v1/chat/completions
  │
  ▼
srt/entrypoints/http_server.py
  │  解析 OpenAI 请求
  ▼
TokenizerManager.generate_request()
  │  tokenize、chat template、多模态预处理
  ▼
Scheduler（TP rank 0）
  │  event_loop_overlap / event_loop_normal
  ▼
get_next_batch_to_run()
  │  - 处理 running batch
  │  - 对 waiting 做 match_prefix
  │  - 按调度策略排序（默认 FCFS；可选 LPM/DFS-weight）
  │  - PrefillAdder 组批
  ▼
ScheduleBatch
  │  包含 reqs、seq_lens、KV 索引、采样参数
  ▼
TPWorker.forward_batch_generation()
  │
  ▼
ModelRunner.forward()
  │  构造 input_ids/positions/out_cache_loc
  │  调用 attention backend
  ▼
Model.forward() → AttentionBackend → Sampler
  │
  ▼
process_batch_result()
  │  追加 token、判断停止、更新 radix cache
  ▼
DetokenizerManager
  │  增量 detokenize
  ▼
流式返回
```

*图 9-2　SGLang 一次请求的完整调用链*

### 9.3.2 关键数据结构

**Req**（`schedule_batch.py`）：

```python
class Req:
    rid: str
    origin_input_ids: list[int]
    output_ids: list[int]
    prefix_indices: torch.Tensor | None   # 命中的 KV 索引
    prefix_len: int                       # 命中长度
    kv_committed_len: int                 # 已写入 KV 的长度
    finished: bool
    ...
```

**TreeNode**（`radix_cache.py`）：

```python
class TreeNode:
    key: list[int]              # 边上的 token 序列
    value: torch.Tensor | None  # KV 池索引
    children: dict[int, TreeNode]
    parent: TreeNode
    lock_ref: int               # 旧版可能叫 ref_counter
    last_access_time: float
    hash_value: int
```

**ScheduleBatch**：

```python
class ScheduleBatch:
    reqs: list[Req]
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor    # 本步新 token 的 KV 槽位
    prefix_lens: torch.Tensor
    sampling_params: ...
```

### 9.3.3 调度器伪代码

```python
def event_loop_overlap(self):
    while True:
        # 1. 处理上一步的结果（与当前 GPU 计算重叠）
        self.process_batch_result(self.running_batch)

        # 2. 调度下一步
        batch = self.get_next_batch_to_run()
        self.running_batch = batch

        # 3. 启动 GPU 前向（异步）
        self.run_batch(batch)

def get_next_batch_to_run(self):
    # 优先处理 running
    if self.running_batch:
        ...

    # 再处理 waiting：match_prefix + 按策略排序（默认 FCFS）
    new_batch = self.get_new_batch_prefill()
    return new_batch or self.running_batch

def get_new_batch_prefill(self):
    adder = PrefillAdder(self.max_prefill_tokens, ...)
    for req in self.waiting_queue:
        # 先 match_prefix，得到 prefix_indices/prefix_len
        req.prefix_indices, req.prefix_len = self.tree_cache.match_prefix(req)
        # 若启用 LPM/DFS-weight，按匹配长度排序；默认 FCFS 按到达顺序
        ...
        if adder.can_add(req):
            adder.add(req)
    return adder.finalize()
```

真实实现里，FCFS/LPM 等排序在 `schedule_policy.py` 中完成；`PrefillAdder` 负责 chunk 切分、token budget、KV 分配与多模态输入。

### 9.3.4 RadixCache 的关键函数

| 函数 | 作用 |
|---|---|
| `match_prefix(key)` | 最长前缀匹配，返回 KV 索引、节点、匹配长度 |
| `insert(key, value)` | 插入新序列，必要时分裂节点 |
| `evict(num_tokens)` | 按 LRU 驱逐叶子，直到释放足够 token |
| `cache_finished_req(req)` | 请求结束后把完整序列插入树 |
| `cache_unfinished_req(req)` | chunked prefill 的中间前缀插入树 |
| `_split_node(node, split_len)` | 分裂节点，切分 KV 索引 |

阅读时重点看：

- `match_prefix` 在部分匹配时如何分裂；
- `evict` 如何跳过 `lock_ref > 0` 的节点；
- `cache_finished_req` 如何计算“已共享”和“需保留”的 KV；
- 树节点与 `TokenToKVPool` 的索引如何对应。

### 9.3.5 ModelRunner 的关键步骤

```python
def forward(self, batch):
    # 1. 构造输入
    input_ids = batch.input_ids
    positions = batch.positions
    out_cache_loc = batch.out_cache_loc   # 本步写入的 KV 槽位

    # 2. attention metadata
    attn_metadata = self.attn_backend.make_metadata(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        prefix_lens=batch.prefix_lens,
        ...
    )

    # 3. 前向
    hidden = self.model(input_ids, positions, forward_batch)

    # 4. 采样
    logits = self.model.compute_logits(hidden)
    next_token_ids = self.sampler(logits, batch.sampling_params)

    return next_token_ids
```

与 vLLM 的 `slot_mapping` 类似，SGLang 用 `out_cache_loc` 表示“本步新 token 的 KV 写到哪里”；历史 KV 通过 `req_pool_indices`、`seq_lens` 和 radix 索引访问。

---

## 9.4 概念对照表

| 概念 | vLLM | SGLang |
|---|---|---|
| 请求对象 | `Request` | `Req` |
| 调度输出 | `SchedulerOutput` | `ScheduleBatch` |
| 调度器 | `Scheduler` | `Scheduler` |
| 调度策略 | FCFS/priority（默认 FCFS） | `SchedulePolicy`（默认 FCFS；可选 LPM/DFS 等） |
| KV 管理器 | `KVCacheManager` | `RadixCache` + `MemoryPool` |
| 物理单位 | `KVCacheBlock` | `TreeNode` + token 索引 |
| 块/槽映射 | `block_tables` + `slot_mapping` | `prefix_indices` + `out_cache_loc` |
| Worker | `GPUWorker` | `TPWorker` |
| 执行器 | `GPUModelRunner` | `ModelRunner` |
| 注意力 | `AttentionBackend` | `AttentionBackend` |
| 输出处理 | `OutputProcessor` | `DetokenizerManager` |
| API 服务 | `api_server.py` | `http_server.py` |
| 进程通信 | ZMQ | ZMQ/共享内存/进程队列 |
| 前缀缓存 | 块哈希表 | 基数树 |

---

## 9.5 调试与追踪技巧

### 9.5.1 先关掉“优化黑盒”

排查正确性问题时，先把优化关掉：

```bash
# vLLM
vllm serve ... --enforce-eager

# SGLang（0.5.23 起分阶段关闭；旧版为 --disable-cuda-graph）
python -m sglang.launch_server ... --disable-decode-cuda-graph --disable-prefill-cuda-graph --disable-overlap-schedule
```

如果关闭后问题消失，说明问题在 CUDA Graph/重叠调度/编译路径；再逐个打开定位。

### 9.5.2 单进程调试

多进程架构（API Server + EngineCore + Worker）会让断点很难打。可以查一下当前版本是否支持单进程模式：

- vLLM：环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（若版本支持）；
- SGLang：部分调试模式可让 scheduler 与 worker 同进程；以版本文档为准。

单进程模式下，Python 断点/日志能覆盖完整调用链。

### 9.5.3 用 request_id 追踪

在日志中过滤同一个 `request_id`：

```bash
grep "req-123" server.log
```

关注：

- 何时进入 waiting；
- 何时被调度；
- 分配了哪些块/索引；
- 何时 prefill 完成；
- 何时被抢占/重算；
- 何时结束、缓存是否插入。

### 9.5.4 CPU/GPU profiler

```bash
# CPU 热点：看 Python 调度、tokenize、detokenize 占比
py-spy record -o profile.svg --pid <pid>

# GPU：kernel 时间线、通信、空洞（NVIDIA Nsight Systems）
nsys profile -o trace python -m sglang.launch_server ...
```

重点看：

- GPU 是否有周期性空洞（CPU 调度瓶颈）；
- attention kernel 占比；
- all-reduce/all-to-all 占比；
- CUDA Graph 是否生效；
- 是否有大量小 kernel（launch 开销）。

### 9.5.5 读测试

测试往往是最好的“设计文档”：

- vLLM：`tests/v1/core/test_scheduler.py`、`tests/v1/core/test_prefix_caching.py`、`tests/v1/core/test_single_type_kv_cache_manager.py`；
- SGLang：`test/registered/radix_cache/` 目录、`test/registered/scheduler/` 目录（0.5.23 起测试按功能分区，路径以版本为准）。

从测试里可以看到：

- 调度器在边界条件下的行为；
- 前缀缓存的命中/驱逐不变量；
- 抢占后的状态恢复；
- 多模态/投机解码的特殊路径。

---

## 9.6 推荐阅读顺序

### 9.6.1 vLLM

1. `vllm/v1/request.py`：请求状态；
2. `vllm/v1/core/sched/scheduler.py`：调度主循环；
3. `vllm/v1/core/kv_cache_manager.py` + `block_pool.py`：块分配/前缀缓存；
4. `vllm/v1/worker/gpu/model_runner.py`：输入构造/执行；
5. `vllm/v1/attention/backends/`：挑一个 backend 看块表寻址；
6. `vllm/distributed/`：TP/EP 通信；
7. `vllm/v1/spec_decode/`：投机解码。

### 9.6.2 SGLang

1. `srt/managers/schedule_batch.py`：Req/ScheduleBatch；
2. `srt/managers/scheduler.py`：事件循环；
3. `srt/managers/schedule_policy.py`：调度策略（默认 FCFS，LPM/DFS 可选）；
4. `srt/mem_cache/radix_cache.py`：树操作；
5. `srt/mem_cache/memory_pool.py`：KV 池；
6. `srt/model_executor/model_runner.py`：输入构造/执行；
7. `srt/layers/attention/`：FlashInfer/FA3/MLA；
8. `srt/speculative/`、`srt/disaggregation/`：高级路径。

---

## 9.7 常见误区

**误区 1：读源码要从第一行读到最后一行。**  
几十万行代码，读完也记不住。正确方式是“带着问题，跟一条调用链”。

**误区 2：网上文章的路径永远有效。**  
两个项目重构频繁，类名/文件可能移动。用 `rg` 按类名定位，不要死记路径。

**误区 3：看懂 Python 就看懂了整个系统。**  
推理引擎还有 C++/CUDA kernel、NCCL 通信、进程间协议。先理解 Python 调度层，再按需下沉。

**误区 4：调度器就是引擎的全部。**  
调度器是大脑，但 KV 管理、ModelRunner、attention backend、输出处理同样关键。缺一个都跑不起来。

**误区 5：日志越多越好。**  
日志太多会拖慢系统、淹没关键信息。用 request_id 过滤，只看一条请求的生命周期。

**误区 6：单进程调试的行为等于生产多进程。**  
单进程方便断点，但 IPC、共享内存、多卡通信的行为可能不同；定位并发问题要回多进程验证。

**误区 7：读一遍就能记住。**  
读源码是“用进废退”。每读一个机制，画一张图、写一段笔记，下次才能快速捡起来。

---

## 9.8 本章小结

- vLLM 的调用链：API Server → AsyncLLM → EngineCore → Scheduler → Worker/ModelRunner → Attention → Sampler → OutputProcessor。
- SGLang 的调用链：HTTP → TokenizerManager → Scheduler → SchedulePolicy/PrefillAdder → TPWorker/ModelRunner → Attention → Sampler → DetokenizerManager。
- vLLM 的核心数据结构是 `Request` + `SchedulerOutput` + `KVCacheBlock`；SGLang 是 `Req` + `ScheduleBatch` + `TreeNode`。
- 两个引擎的 `slot_mapping`/`out_cache_loc` 都是“token → KV 槽位”的映射，是理解 KV 写入的关键。
- 调试时先关优化、再单进程、用 request_id 追踪、配合 py-spy/nsys。
- 测试是读源码的捷径，尤其是调度器与前缀缓存的边界测试。

> **用大白话再说一遍**：读源码像追一份卷子的流转单：你不需要认识每个部门，只要拿着一个卷号（request_id），看它从“接卷”到“交卷”经过哪些环节。每到一个环节，问“谁交给我的、我交给谁、我改了什么状态”，一条调用链就清楚了。

---

## 思考题

1. 在 vLLM 的 `Scheduler.schedule()` 中，为什么 running 队列要优先于 waiting？如果反过来会发生什么？
2. `get_computed_blocks()` 与 `allocate_slots()` 的调用顺序能否交换？交换后可能出现什么竞态或重复分配？
3. 在 SGLang 的 `match_prefix()` 中，节点分裂后原节点的 `value` 如何切分？如果切分错了，KV 索引会指向什么？
4. 为什么 SGLang 的 `out_cache_loc` 与 vLLM 的 `slot_mapping` 本质上是同一种东西？它们的形状/语义有何差异？
5. 如果 CUDA Graph 捕获后输入地址固定，ModelRunner 如何保证每步的 `input_ids`/`positions` 写入正确的缓冲区？
6. 你怀疑“前缀缓存没有生效”，请给出从 API 日志到 radix tree/块哈希的排查路径。
7. 读 `test_scheduler.py` 中的一个测试，说明它验证了什么不变量，并尝试构造一个反例。
8. 对比两个引擎的 `process_batch_result`/`OutputProcessor`：停止字符串跨 token 时，它们分别如何处理？

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 10 章 选型、演进与前沿 →](10-selection-and-future.md)
