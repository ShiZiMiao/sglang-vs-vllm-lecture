# 第 7 章 性能评测：如何科学地比较 vLLM 与 SGLang

> **本章目标**：建立“不被 benchmark 数字忽悠”的能力。你将学会定义 workload、选择指标、控制变量、使用两家官方压测工具，并设计一个可复现的对照实验。
>
> **先修知识**：第 1 章的 TTFT/TPOT/吞吐/goodput、第 5 章的“差异何时重要”。本章会用到 P50/P99，忘了可以回第 0 章 0.5.3。
>
> **一句话概括**：benchmark 不是“跑一条命令看谁快”，而是“在明确 workload 和 SLO 下，控制变量、重复测量、看分位数”。
>
> **给零基础同学**：本章公式不多，重点是“别被数字骗”。先看 7.0 的三个词，再看 7.5 的公平比较清单；实验部分可以照着 7.8 做。

---

## 7.0 先搞懂三个词

### 7.0.1 Workload：你测的到底是什么流量

**Workload（工作负载）** 是你压测时模拟的请求形态，至少包括：

- 输入长度分布（prompt 多长）；
- 输出长度分布（回答多长）；
- 前缀共享比例（多少 token 是重复的）；
- 到达模式（均匀到达、突发、固定并发）；
- 请求类型（聊天、RAG、JSON、代码、多模态）。

类比：测餐厅出餐速度，你要说明“客人点的是快餐还是宴席、同时来几桌、有没有提前备菜”。只报“每小时出 100 道菜”没有意义。

### 7.0.2 分位数：为什么不能只看平均值

假设 100 个请求，99 个都在 100 ms 内完成，1 个卡了 10 秒：

- 平均值 = (99×100 ms + 10000 ms) / 100 ≈ 199 ms，看起来还行；
- P50 = 100 ms，大多数人的体验；
- P99 表示“99% 的请求不超过它”。按最近秩计算，P99 仍然是 100 ms；那个 10s 的请求属于最慢的 1%，要看 **P99.9** 或**最大值**才能看到；
- 所以 P99 不是“最慢请求”，也不是“最慢 1% 的平均值”，而是一个分位阈值。

**P50** 代表“典型用户”，**P99** 代表尾延迟的边界。生产系统通常同时看两者；如果极端尾部很重要，再加 P99.9 和最大值。不同工具的插值算法可能让 P99 略有差异，报告时要说明口径。

### 7.0.3 开环与闭环：两种压测方式

| 方式 | 怎么发请求 | 类比 | 适合看什么 |
|---|---|---|---|
| **闭环（closed-loop）** | 固定 N 个客户端，收到响应后再发下一个 | 固定 N 桌客人，吃完才换下一桌 | 最大吞吐、吞吐-延迟曲线 |
| **开环（open-loop）** | 按固定速率（如 10 req/s）到达，不管系统忙不忙 | 客人按门铃进来，不管店里有没有空位 | 真实排队、P99、SLO 达标率 |

闭环容易“客户端自己限速”，掩盖排队；开环更接近真实用户，但需要控制好到达率。两家压测工具的 `--request-rate` 只控制“发出请求的节奏”：具体数值表示按该速率开环发送；`inf` 表示所有请求在 t=0 一次性发出，这是**开环突发**，不是闭环。`--max-concurrency` 只是限制同时在飞的请求数，不会让客户端“收到响应再发下一个”；严格的闭环需要固定客户端数、收到响应后再发下一个的压测器（或自研脚本）。

### 7.0.4 冷缓存与热缓存

- **冷缓存**：服务刚重启，前缀缓存为空，第一条请求最慢；
- **热缓存**：缓存已经填充，后续请求可以命中。

比较两个引擎时，要么都冷、要么都热，不能一个热身一个不热身。测量前重启服务或显式清缓存，并记录 cache hit。

---

## 7.1 为什么推理引擎的 benchmark 特别容易误导

同一对引擎，在不同人手里可以测出完全相反的结论。常见原因：

1. **workload 不同**：输入/输出长度、前缀共享比例、到达模式、并发数，都会改变瓶颈位置。
2. **硬件不同**：H100 vs A100 vs L40S，NVLink 有无，PCIe 代数，都会影响 TP/PD 的结论。
3. **版本不同**：两个项目每周都在合并 PR；一个 flag 的默认值变化就可能带来 20% 差异。
4. **调参不同**：`max_num_seqs`、chunked prefill 大小、CUDA Graph、量化、attention backend，都会影响结果。
5. **缓存状态不同**：冷缓存 vs 热缓存，前缀缓存命中率可能从 0 到 90%。
6. **客户端瓶颈**：压测机的 CPU/网络先饱和，测出来的“引擎延迟”其实是客户端延迟。
7. **指标口径不同**：平均 TTFT vs P99 TTFT；输出 token/s vs 总 token/s；是否包含排队。
8. **选择性报告**：只报告对自己有利的并发点、模型或长度。

因此，看到“A 比 B 快 N 倍”时，第一反应不应该是相信或否定，而是问：**在什么条件下、用什么口径、测的是什么 workload？**

---

## 7.2 指标：测什么，怎么测

### 7.2.1 延迟指标

| 指标 | 定义 | 测量位置 | 注意事项 |
|---|---|---|---|
| **TTFT** | 请求发出 → 第一个输出 token | 客户端流式时间戳 | 包含网络与排队；服务端指标通常更小 |
| **TPOT** | (E2E - TTFT) / (输出 token 数 - 1) | 客户端/服务端 | 平均值会掩盖长尾 |
| **ITL** | 相邻 token 的时间间隔 | 客户端流式时间戳 | 关注 P50/P90/P99，不只看均值 |
| **E2E** | 请求发出 → 最后一个 token | 客户端 | 与输出长度强相关 |

**服务端 vs 客户端**：

- 服务端指标（引擎自己上报）排除了网络与客户端处理，适合定位引擎内部瓶颈；
- 客户端指标（压测工具测）才是用户真实体验；
- 两者都要报，且要说明差异来源（网络、代理、SSE 解析）。

### 7.2.2 吞吐指标

- **输出 token/s**：最常用，但受输出长度影响；
- **总 token/s**：包含 prompt，适合 prefill 重的负载；
- **请求/s**：适合固定长度的业务；
- **每 GPU token/s**：多卡时用于比较效率；
- **每美元 token/s**：TCO（总拥有成本）视角，生产决策更重要。

### 7.2.3 Goodput 与 SLO

定义 SLO，例如：

```text
TTFT P99 < 800 ms
TPOT P99 < 80 ms
错误率 < 0.1%
```

**Goodput** = 单位时间内满足上述条件的请求数（或 token 数）。

为什么重要：一个系统在并发 256 时吞吐很高，但 P99 TTFT 已经 5 秒；另一个系统并发 128 时吞吐略低，但 P99 在 SLO 内。对业务来说，后者才有价值。

### 7.2.4 测量实现

流式请求的客户端伪代码：

```python
start = time.perf_counter()
first = None
last = None
chunks = 0
for chunk in stream:
    now = time.perf_counter()
    if first is None:
        first = now
        ttft = first - start
    else:
        itls.append(now - last)   # 第一个 token 之后才开始算 ITL
    last = now
    chunks += 1

e2e = last - start
tpot_approx = (e2e - ttft) / max(chunks - 1, 1)  # 近似：假设 1 chunk = 1 token
```

注意 `last` 必须在循环前初始化；并且第一个 token 只用于计算 TTFT，不产生 ITL。很多手写脚本会在这里写出 `now - last` 的未定义变量错误，或者把“请求发出到第一个 token”误算进 ITL。上面的 `tpot_approx` 只在“一个 chunk 正好一个 token”时准确；如果 chunk 会合并多个 token，应以响应的 `usage.completion_tokens` 为准重新计算。

注意：

- 记录的是**chunk 到达时间**，一个 chunk 可能包含多个 token；如果服务端按 token 流式返回，chunk 数≈token 数；否则需要从 usage 或文本估算。
- 客户端要避免同步日志/JSON 解析阻塞事件循环。
- 多进程压测，避免 GIL（Python 全局解释器锁）成为瓶颈。

---

## 7.3 Workload 建模：benchmark 的灵魂

### 7.3.1 四个关键维度

1. **输入长度分布**：固定长度还是长尾分布？是否有超长 prompt？
2. **输出长度分布**：短输出（分类/JSON）还是长输出（创作/代码）？
3. **前缀共享**：system prompt、few-shot、多轮历史、RAG 文档的共享比例；
4. **到达模式**：固定并发（closed-loop）还是泊松到达（open-loop）？是否有突发？

### 7.3.2 常见数据集

| 数据集 | 特点 | 适用 |
|---|---|---|
| ShareGPT | 真实多轮对话，长度分布长尾 | 通用对话 |
| Alpaca | 指令数据，较短 | 指令跟随 |
| GSM8K | 数学题，输出中等 | 推理负载 |
| 随机 token | 可控长度，无前缀共享 | 基线/压力测试 |
| 自定义 | 复刻业务分布 | 生产选型 |

**重要**：ShareGPT 的对话天然有前缀共享（同一会话多轮），会放大 RadixAttention 的优势；如果你用“每条请求都独立”的随机 prompt，结论可能完全不同。

### 7.3.3 Closed-loop vs Open-loop

**Closed-loop（固定并发）**：

```text
N 个客户端，每个发请求 → 等响应 → 再发下一个
```

- 衡量“在 N 并发下，系统能跑多快”；
- 不模拟真实到达过程，排队延迟被“客户端等待”吸收；
- 适合画 throughput-latency 曲线。

**Open-loop（固定到达率）**：

```text
按泊松过程，每秒 λ 个请求到达，不管系统是否忙
```

- 衡量“在真实到达率下，SLO 是否达标”；
- 能暴露排队、尾延迟、过载；
- 更适合生产容量规划。

两家压测工具都支持 `--request-rate`：具体数值是 open-loop 定速到达；`inf` 是 t=0 全部发出的 open-loop 突发（官方帮助原文是 “all the requests are sent at time 0”），不是 closed-loop。要模拟固定并发、收到响应再发下一个，需要固定客户端数的闭环压测器。

### 7.3.4 前缀共享的构造

要比较 RadixAttention 与块级前缀缓存，必须显式控制共享：

- **共享比例**：0%、50%、90% 的 prompt token 来自共享前缀；
- **对齐**：共享前缀长度取块大小的倍数（16/64，视版本）vs 非倍数；
- **分支**：所有请求共享同一前缀 vs 树状多分支；
- **缓存压力**：共享前缀总量接近/超过 KV 池容量，观察驱逐行为。

只在“0% 共享”的 workload 上比较，会低估 SGLang 的强项；只在“90% 共享、完全对齐”的 workload 上比较，会高估其普适性。

---

## 7.4 官方压测工具

> 以下命令与参数于 2026 年 11 月核对；两个项目的 bench CLI 仍在演进，执行前先看 `--help`。

### 7.4.1 vLLM

现代 vLLM 提供 `vllm bench` 子命令（旧版是 `benchmarks/benchmark_serving.py`；官方说明见 [Benchmarking](https://docs.vllm.ai/en/v0.34.0/benchmarking/)）：

```bash
# 在线服务压测（OpenAI 兼容端点）
vllm bench serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 10 \
  --max-concurrency 64 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result
```

```bash
# 离线吞吐
vllm bench throughput \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input-len 1024 \
  --output-len 256 \
  --num-prompts 1000
```

```bash
# 固定 batch 的延迟
vllm bench latency \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input-len 1024 \
  --output-len 128 \
  --batch-size 1
```

常用参数：

- `--request-rate`：请求/秒；`inf` 表示尽可能快；
- `--max-concurrency`：最大并发；
- `--num-prompts`：总请求数；
- `--dataset-name random`：随机长度；
- `--random-input-len` / `--random-output-len`：固定长度；
- `--percentile-metrics`：要报告的指标；
- `--save-result`：保存 JSON，便于对比。

以 `vllm bench serve --help` 为准，版本间参数可能变化。

### 7.4.2 SGLang

```bash
# 在线服务压测（新版入口；旧版可能是 python -m sglang.bench_serving）
python -m sglang.benchmark.serving \
  --backend sglang \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 10 \
  --max-concurrency 64
```

```bash
# 离线吞吐（新版入口；旧版是 python -m sglang.bench_offline_throughput）
python -m sglang.benchmark.offline_throughput \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --num-prompts 1000
```

```bash
# 单 batch 延迟（新版入口；旧版是 python -m sglang.bench_one_batch）
python -m sglang.benchmark.one_batch \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --batch-size 1 \
  --input-len 1024 \
  --output-len 128
```

SGLang 的 `benchmark.serving` 还支持 `--dataset-name random`、`--random-input-len`、`--random-output-len` 等参数（以版本为准；官方说明见 [Benchmark and Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)）。缓存清理在不同版本中可能是 `--flush-cache` 之类的选项，也可能没有；如果没有，就在两次测量之间重启服务，或让每条预热请求使用唯一前缀，避免“热缓存”污染对比。

### 7.4.3 第三方工具

- **NVIDIA genai-perf**：支持 OpenAI 兼容端点，指标齐全；
- **llmperf**：Ray 生态的压测工具；
- **locust / k6**：通用 HTTP 压测，适合自定义协议；
- **自研脚本**：最灵活，但要注意客户端瓶颈与统计正确性。

---

## 7.5 公平比较清单

做 A/B 对比时，至少控制表 7-1 中的变量。

*表 7-1　公平比较的控制变量清单*

| 类别 | 必须一致/记录 |
|---|---|
| 模型 | 同一模型、同一 revision、同一 tokenizer、同一 chat template |
| 精度 | 同一 dtype/量化格式；若不同，需单独说明 |
| 硬件 | 同一 GPU 型号/数量/互联；同一驱动/CUDA |
| 并行 | 同一 TP/PP/DP/EP 配置 |
| 上下文 | 同一 `max_model_len` / `context-length` |
| 显存 | 同一 `gpu_memory_utilization` / `mem_fraction_static`（或都调到不 OOM 的合理值） |
| 批参数 | 同一 `max_num_seqs` / `max-running-requests`；chunked prefill 配置 |
| 缓存 | 都开或都关前缀缓存；测量前清空；记录 cache hit |
| 图/编译 | 都开 CUDA Graph；记录是否用 torch.compile |
| 投机 | 都关或都开同等配置 |
| 客户端 | 同一压测机、同一网络、足够多进程、无 CPU 饱和 |
| 数据 | 同一数据集、同一随机种子、同一长度分布 |
| 版本 | 记录两个引擎的精确版本/commit |
| 重复 | 至少 3 次，报告 median 与 P99；丢弃 warmup |

**最容易犯的错**：

- 用 A 的“调优后配置”对比 B 的“默认配置”；
- 一边开 prefix cache，一边没开；
- 一边用 FP8，一边用 BF16；
- 客户端只有 4 个进程，却测 256 并发；
- 测完一轮不重启服务，第二轮全是热缓存。

---

## 7.6 怎么读官方 benchmark

看到一张“vLLM vs SGLang”的图，按下面的顺序检查：

1. **模型与硬件**：是不是目标场景？TP 多少？互联是什么？
2. **输入/输出长度**：是固定长度还是真实分布？输出多长？
3. **并发/到达率**：是最大吞吐点还是某个 SLO 点？
4. **指标**：TTFT 还是吞吐？P50 还是 P99？是否包含排队？
5. **缓存**：前缀缓存开没开？workload 有多少共享前缀？
6. **量化/投机**：是否一边用了 FP8/EAGLE，另一边没用？
7. **版本**：两个引擎的版本/commit 是否同期？是否用了 nightly？
8. **脚注**：有没有“best of N”“tuned for X”“excluding Y”？

一条经验：**官方博客的绝对数字通常不可复现**，但“在什么条件下谁擅长什么”的定性结论往往有参考价值。你要做的是在自己的 workload 上验证定性结论。

---

## 7.7 典型定性结论（以及为什么）

以下是社区中较常见的观察，**不是保证**：

| 场景 | 常见胜者 | 原因 |
|---|---|---|
| 多轮对话、Agent、RAG，前缀共享高 | SGLang | RadixAttention token 级复用；开启 LPM 后可进一步减少缓存驱逐 |
| 结构化 JSON/正则输出 | SGLang（略优） | 压缩 FSM + jump-forward |
| DeepSeek-V3/R1 大规模 MoE | SGLang（常见） | DP attention、EP/EPLB、PD 分离案例 |
| 通用多模型平台、长尾模型 | vLLM | 模型/硬件/量化覆盖广 |
| 长输出创作 | 差距缩小 | decode 主导，前缀收益被稀释 |
| 小模型、高 QPS | 看实现 | CPU 调度开销关键；SGLang overlap 有优势，vLLM V1 也在优化 |
| 单请求低延迟 | 看 kernel/backend | 与引擎调度关系较小，更多取决于 attention backend 与模型 |
| 离线最大吞吐 | 看 workload | 两者都可能赢，取决于长度分布与缓存 |

**解释框架**：先判断瓶颈在 prefill 还是 decode，再看前缀共享是否高，最后看 CPU 开销是否显著。不要只记“谁赢”。另外，这些观察通常假设两边版本相近、都做了基本调参、缓存状态一致；如果一方用调优配置、另一方用默认配置，结论不成立。

---

## 7.8 一个可复现的对照实验

### 7.8.1 实验目标

验证“前缀共享比例”对两个引擎 TTFT 与吞吐的影响。

### 7.8.2 变量

- 自变量：共享前缀比例 $r \in \{0\%, 50\%, 90\%\}$；
- 控制变量：模型、硬件、TP、dtype、max context、并发、输入/输出长度分布、缓存开关；
- 因变量：TTFT P50/P99、TPOT P50/P99、输出 token/s、goodput（TTFT P99 < 1s）。

### 7.8.3 workload 构造

- 总 prompt 长度 2048 token；
- 共享前缀长度 = $\text{round}(2048 \times r)$，取块大小的倍数（16/64）与非倍数各一组；
- 每条请求的独特后缀随机生成；
- 输出长度固定 128 token（也可加一组 512）；
- 到达率：固定 10 req/s，持续 5 分钟；另做一组 closed-loop 最大并发。

> `code/bench_prefix.py` 目前只提供 `shared`（100% 共享）和 `unique`（0% 共享）两档；要跑 50%/90%，把它的前缀构造改成“共享段 + 独有段”即可，或使用自定义数据集。

### 7.8.4 步骤

1. 启动 vLLM，记录版本、启动参数；
2. 用仓库的 `code/bench_prefix.py`（见第 8 章）生成 `shared` / `unique` 两组 workload：`shared` 每条请求拼相同长前缀，`unique` 每条请求用不同前缀。分别测冷缓存与热缓存；脚本输出 TTFT/E2E 和近似的 chunks/s，goodput 需要用 `--json-out` 的原始记录按 SLO 过滤后自己算。如果坚持用 `vllm bench serve`，要准备一个“共享前缀”的自定义数据集，否则 `random` 数据集不会产生前缀复用；
3. 启动 SGLang，记录版本、启动参数（对齐显存/批参数）；
4. 用同一个脚本跑 `shared` / `unique`，同样区分冷/热缓存；如果想观察缓存感知调度，再重启 SGLang 并加 `--schedule-policy lpm`，跑同一组 `shared` workload；
5. 每组重复 3 次，取 median，记录 P99；
6. 记录 GPU 利用率、显存、cache hit（如果引擎暴露）；
7. 交换启动顺序再跑一轮，排除机器状态影响。

### 7.8.5 结果记录模板

*表 7-2　配置记录*

| 引擎 | 版本/commit | 共享比例 | 前缀是否块对齐 |
|---|---|---|---|
| vLLM | ... | 0% | - |
| vLLM | ... | 50% | 是 / 否 |
| vLLM | ... | 90% | 是 / 否 |
| SGLang | ... | 0% | - |
| SGLang | ... | 50% | 是 / 否 |
| SGLang | ... | 90% | 是 / 否 |

*表 7-3　延迟指标（ms）*

| 引擎 | 共享比例 | TTFT P50 | TTFT P99 | TPOT P99 |
|---|---|---|---|---|
| vLLM | ... | | | |
| SGLang | ... | | | |

*表 7-4　吞吐与 goodput*

| 引擎 | 共享比例 | 输出 tok/s | goodput（满足 SLO 的 req/s） |
|---|---|---|---|
| vLLM | ... | | |
| SGLang | ... | | |

### 7.8.6 预期与解释

- $r=0$：两者差距应较小，SGLang 的树维护可能带来轻微开销；
- $r=50\%$：SGLang 的 token 级命中开始占优；若同时开启 LPM，排序收益更明显；非对齐场景差距更大；
- $r=90\%$：在缓存能容纳共享前缀的前提下，SGLang 的 TTFT 优势最大；但如果输出很长，E2E 差距被稀释；
- 如果结果与预期相反，检查：缓存是否真的命中、调度是否被其他因素主导、客户端是否瓶颈、版本是否有已知问题。

---

## 7.9 结果呈现与统计

- 报告 **median + P90/P99**，不要只报均值；
- 报告 **样本量、持续时间、warmup、重复次数**；
- 报告 **错误率、超时率**；
- 画 **throughput-latency 曲线**，而不是单点；
- 标注 **SLO 线**，报告 goodput；
- 保存原始 JSON/CSV，便于复现；
- 记录 **GPU 时钟/温度/功耗**，避免 thermal throttling（过热降频）影响；
- 多机时记录网络拓扑与 NCCL 配置。

---

## 7.10 常见误区

**误区 1：吞吐越高，系统越好。**  
如果大量请求超 SLO，吞吐只是“把系统压爆后的数字”。生产看 goodput：满足 SLO 的有效吞吐。

**误区 2：跑一次就能下结论。**  
GPU 时钟、缓存状态、客户端负载、网络抖动都会影响结果。至少重复 3 次，报告 median 和 P99。

**误区 3：平均值够用了。**  
平均值会掩盖尾延迟。用户感知的“卡顿”通常在 P95/P99，不在均值。

**误区 4：闭环压测等于真实流量。**  
闭环中客户端会等响应，排队被“客户端节流”吸收；真实用户是按自己的节奏到达的，需要开环压测。

**误区 5：一个热缓存、一个冷缓存，也能公平比较。**  
缓存状态必须一致。热缓存会让前缀命中的引擎看起来快很多，但那是“热身”的功劳。

**误区 6：只测一种输入/输出长度。**  
不同长度分布会把瓶颈从 prefill 推到 decode，结论可能反转。至少测短、中、长三组。

**误区 7：官方博客的数字能直接套到我的业务。**  
官方数字是特定模型、硬件、版本、workload 下的结果。可以参考定性结论，不能直接当容量规划。

---

## 7.11 本章小结

- Benchmark 的结论高度依赖 workload、硬件、版本、调参与缓存状态。
- 指标要区分 TTFT/TPOT/ITL/E2E、均值/分位数、吞吐/goodput。
- 用 open-loop 模拟真实到达，用 closed-loop 画吞吐-延迟曲线。
- 前缀共享比例、对齐、分支、缓存压力，是比较两个引擎时最关键的 workload 变量。
- 公平比较要控制模型、精度、硬件、并行、显存、批参数、缓存、客户端与版本。
- 官方数字可参考定性结论，但必须在自己 workload 上复现。
- 一个好的 benchmark 报告 = 可复现的配置 + 完整的分布 + 诚实的局限。

> **用大白话再说一遍**：压测就像考试，先说明“考什么卷子”（workload），再说明“怎么监考”（闭环/开环），最后看“最好、最差、大多数人的成绩”（P50/P99），而不是只看平均分。换了卷子、换了考场、换了版本，分数就不能直接比。

---

## 思考题

1. 为什么 closed-loop 压测可能高估系统的 SLO 达标率？请用排队论解释。
2. 如果客户端只有 8 个进程，却要测 256 并发，会发生什么？如何判断客户端是否成为瓶颈？
3. 设计一个 workload，使得 vLLM 的块级前缀缓存命中率接近 SGLang 的 token 级命中率。关键条件是什么？
4. 为什么“输出长度从 128 增加到 2048”会缩小两个引擎的 E2E 差距，但不一定缩小 TTFT 差距？
5. 你看到一张图：SGLang 在 90% 共享前缀下 TTFT 比 vLLM 低 5 倍。请列出至少 5 个需要核查的问题。
6. 如何测量“前缀缓存命中率”？如果引擎没有直接暴露该指标，能否用 TTFT 或 prefill token 数间接估计？
7. 设计一个实验，区分“attention backend 更快”与“调度/缓存更优”对总吞吐的贡献。
8. 为什么 goodput 比 throughput 更适合作为生产选型指标？请举一个反例。

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 8 章 动手实践：启动、调用、压测与调参 →](08-hands-on.md)
