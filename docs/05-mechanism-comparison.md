# 第 5 章 核心机制正面对比：页表 vs 基数树，通用调度 vs 缓存感知

> **本章目标**：把第 3、4 章的知识放在同一张桌子上，逐项比较 KV 管理、调度、执行与分布式策略，并回答一个关键问题：**这些设计差异在什么 workload 下才真正转化为性能差异？**
>
> **先修知识**：[第 3 章](03-vllm.md) 的块表/块哈希、[第 4 章](04-sglang.md) 的基数树/缓存感知调度。如果只记得“vLLM 分块、SGLang 种树”，也能读，但建议先补完这两章。
>
> **一句话概括**：vLLM 像“固定大小的收纳盒 + 盒号清单”，SGLang 像“按前缀共享的文件夹树”；一个胜在通用和简单，一个胜在前缀复用和可选的缓存感知调度。
>
> **给零基础同学**：本章不是要你判定“谁更强”，而是教你判断“在什么条件下谁更合适”。先看 5.0 的类比，再看表格，最后用 5.5 的公式估算自己的场景。

---

## 5.0 先用衣柜类比理解两种方案

假设你要整理一整个衣柜：

**vLLM 的做法：固定收纳盒**

- 把衣服按每 16 件装进一个标准收纳盒；
- 每个盒子上贴编号，柜子里的盒子可以不连号；
- 一张清单记录“第 1 盒放 7 号格、第 2 盒放 3 号格……”；
- 两个搭配如果前几盒完全一样，可以暂时共用盒子；
- 要改其中一件，就把那一盒复制一份再改；
- 盒子可能有空位（最后没装满），但不会出现“总空间够、却找不到连续大空间”的问题。

**SGLang 的做法：按搭配挂衣杆**

- 把“套装搭配”从前往后挂在同一根杆上；
- 相同的搭配共享前面那段杆，分叉后各挂各的；
- 每条搭配记一个“从杆的哪个位置开始”的索引；
- 不要求固定长度，任意长度的共享前缀都能复用；
- 分支很多、前缀很长时，共享得非常细；
- 但要维护一棵树，找位置、分叉、回收都需要额外操作。

| 衣柜方案 | 引擎 | 强项 | 代价 |
|---|---|---|---|
| 固定收纳盒 + 盒号清单 | vLLM | 通用、简单、kernel 友好 | 共享粒度受 16 token 盒子限制 |
| 按搭配挂衣杆 + 树 | SGLang | token 级前缀复用、缓存感知调度（可选） | 树维护开销、公平性风险 |

这个类比只强调“索引与共享方式”；SGLang 底层仍然需要盒子/槽位来放 KV，只是它可以把共享粒度做到单个 token（`page_size=1` 时），而不是一整个盒子。理解了这张表，后面的对比表就只是“细节展开”。

---

## 5.1 两条技术路线的一张图

两条路线的端到端差异可以压缩成图 5-1：

```text
vLLM:
  请求 → FCFS/token budget 调度
       → 固定大小块（如 16 token）
       → block table: 逻辑块 → 物理块
       → 块级哈希前缀缓存 + LRU
       → PagedAttention kernel
       → CUDA Graph / 多种 backend

SGLang:
  请求 → FCFS（默认）/ LPM / DFS-weight（可选）
       → token 级 KV 池（page size 默认 1）
       → radix tree: token 前缀 → KV 索引
       → 节点分裂 + 引用计数 + LRU 叶子驱逐
       → FlashInfer/FA3/FlashMLA 等 backend
       → overlap scheduler + CUDA Graph
```

*图 5-1　vLLM 与 SGLang 的两条技术路线*

两者都建立在 continuous batching 之上，都支持 chunked prefill、prefix caching、CUDA Graph、投机解码、TP/PP/DP/EP、PD 分离。**差异不在“有没有”，而在“数据结构与默认策略”**。

---

## 5.2 KV Cache 管理对比

### 5.2.1 数据结构

| 维度 | vLLM PagedAttention | SGLang RadixAttention |
|---|---|---|
| 基本单位 | 固定大小 block（长期默认 16 token，0.34.0 起默认 64） | 分页/槽位池，`page_size` 默认 1（可配置） |
| 索引结构 | 每条序列一张 block table | 全局一棵基数树 + 每请求的 KV 索引列表 |
| 逻辑→物理 | 块号映射 | 树节点直接存 KV 池索引 |
| 共享粒度 | 完整块 | 任意 token 前缀（`page_size=1` 时） |
| 共享机制 | 引用计数 + copy-on-write | 引用计数 + 节点共享 |
| 回收粒度 | 块 | 叶子节点（可含多个 token） |
| 驱逐策略 | 空闲队列 LRU | 叶子 LRU + 引用计数保护 |
| 元数据 | 块表、哈希表 | 树节点、children、key/value |
| 适用负载 | 通用、变长、多租户 | 前缀密集、多轮、Agent/RAG |

### 5.2.2 粒度：块大小的边界有多重要

假设共享前缀长度为 2005 token，块大小为 16（vLLM 长期默认；0.34.0 起默认 64，这里用 16 方便与历史资料对照）：

- vLLM：能命中的完整块是 $\lfloor 2005/16 \rfloor = 125$ 块，即 2000 token；剩下 5 个 token 需要重算；
- SGLang：可以精确命中 2005 token，一个都不浪费。

5 个 token 听起来微不足道，但在以下场景会被放大：

- 共享前缀很短、请求很多（例如每条请求共享 20 token 的模板，块级只能命中 16）；
- 前缀长度经常不是块大小的整数倍（chat template 拼接后很常见）；
- 大量分支只在末尾几个 token 不同（Agent 的“选择”节点）。

不过，如果前缀很长（几千 token）且块对齐，两者的命中率差异通常很小（在常见长前缀测试中往往不到 1%，但会随 workload 变化）。**不要因为“token 级”三个字就认为 vLLM 的前缀缓存无效**。

> 上面的“精确命中 2005 token”默认 SGLang 的 `page_size=1`。如果把它设成 16/32，SGLang 的匹配与分配也会变成页级，token 级优势会缩小；此时要按实际 page size 重新计算。反过来，vLLM 0.34 的 `block_size` 默认 64，但可用 `prefix_match_unit` 把命中粒度做得比物理块更细，具体以启动配置为准。

### 5.2.3 共享与 Copy-on-Write

vLLM 的 COW 发生在“共享块还有空位、某条序列要往这个空位继续写入”时（图 5-2）：

```text
A、B 共享块 9（只填了一半）
B 要往块 9 的空位写 → 分配块 15，复制块 9 已有内容 → B 的块表改为 [..., 15]
若块 9 已满：B 直接分配新块追加，不需要复制块 9
```

*图 5-2　块级 COW 与树节点共享的写入路径*

SGLang 的共享发生在树节点上：

```text
A、B 共享节点 N 的 KV 索引
B 继续生成 → 新 token 的 KV 写入新节点/新索引，不影响 N
```

区别在于：

- vLLM 的共享单位是“整块”；在共享块内追加时要复制整块（示例中 16 token 的 K/V），块已满时则直接开新块；
- SGLang 的共享单位是“树节点覆盖的 token 段”，分叉时只需为新后缀分配 KV，不需要复制共享段。

对于“同一前缀、大量短分支”的树状生成（如 beam search、Agent 分支），SGLang 的分叉成本更低。

### 5.2.4 驱逐策略

两者都是 LRU 变体，但作用对象不同：

- vLLM：空闲块队列按释放顺序排列，头部最久未使用；命中时把块从空闲队列摘下；释放时按逆序放回尾部。
- SGLang：按树节点的 `last_access_time` 找最久未访问的**叶子**；内部节点因为被多个分支共享，不能单独释放。

这带来一个细微差异：

- 块级 LRU 可能把“某个长前缀的中间块”驱逐掉，导致整段前缀失效；
- 树级 LRU 只驱逐叶子，天然保护共享的中间前缀，但也可能保留暂时不用的整棵子树，需要额外的修剪/压缩策略。

两者都在工程上做了优化，实际命中率取决于访问模式与缓存容量，不能只凭数据结构下结论。

### 5.2.5 内存浪费

| 浪费类型 | vLLM | SGLang |
|---|---|---|
| 外部碎片 | 几乎没有 | 几乎没有（token 池 + 任意索引） |
| 内部碎片 | 最后一块最多 15 token | 几乎没有（`page_size=1` 时按 token 分配；>1 时按页产生碎片） |
| 元数据 | 块表 + 哈希表 | 树节点 + 索引列表 |
| 预留 | 按块增长 | 按 token 增长 |

SGLang 在“细粒度”上占优，但树节点本身也有内存开销；vLLM 在“元数据简单、kernel 友好”上占优。对于长上下文、大并发，两者都会把大部分显存用于 KV 数据本身，元数据占比通常较小。

### 5.2.6 能否组合？

可以。概念上：

- 用分页块管理物理 KV，降低元数据与 kernel 寻址开销；
- 用基数树管理“前缀 → 块序列”的映射，做 token 级匹配与共享；
- 块内保留部分 token 的细粒度索引，或用“最后一块特殊处理”来支持非对齐前缀。

事实上，两个引擎都在向彼此靠拢：vLLM 支持更灵活的前缀缓存与 KV connector；SGLang 支持 page size >1 的 backend 与分层缓存。**“页表 vs 基数树”是教学上的简化，工程实现是混合的。**

---

## 5.3 调度策略对比

### 5.3.1 默认策略

| 维度 | vLLM | SGLang |
|---|---|---|
| 默认顺序 | FCFS | FCFS（当前版本） |
| 前缀感知 | 缓存命中，但不按命中长度排序 | 默认不排序；开启 LPM 等策略后才按命中长度排序 |
| 可选策略 | FCFS、priority | fcfs、lpm、dfs-weight、hrrn、shortest-prefill-first、random 等（0.5.23 共 9 种） |
| 公平性 | 较好 | FCFS 较公平；LPM 可能饿死低命中请求 |
| 实现复杂度 | 较低 | FCFS 较低；缓存感知策略需匹配 + 排序 |

> **重要澄清**：很多旧文章说“SGLang 默认 LPM”，这并不完全是错——SGLang 0.4.0 的默认值确实是 `lpm`；但 0.4.6 起已把默认值改为 `fcfs`，0.5.23 也沿用 `fcfs`。要用 LPM，需要显式指定 `--schedule-policy lpm`。所以下面讨论的 LPM 优势，都指“启用缓存感知策略之后”的行为，而不是新版本的开箱默认。

### 5.3.2 调度流程的差异

vLLM（概念版）：

```text
1. 先跑 running：decode 或继续 chunked prefill
2. 再跑 waiting：按到达顺序准入
3. token budget 用完则停
4. 显存不足则抢占（recompute）
```

SGLang（概念版）：

```text
1. 对 waiting 中每个请求做 match_prefix，得到可复用 token 数
2. 若启用 LPM，按匹配长度排序；若默认 FCFS，则按到达顺序
3. 先跑 running，再用剩余预算准入 waiting
4. 高命中请求优先（仅 LPM 等策略），必要时 chunked prefill
5. 显存不足则驱逐 LRU 叶子 / 抢占
```

差异的核心是第 2 步：**vLLM 的 waiting 顺序是“谁先来”；SGLang 默认也是“谁先来”，但开启 LPM 后可以变成“谁命中多”**。

### 5.3.3 什么时候 LPM 真的有用（前提：已启用）

LPM 的收益取决于“缓存命中差异”是否足够大：

- 如果所有请求都命中相同长度（例如统一 system prompt），排序没有意义；
- 如果缓存容量足够大，所有请求最终都能命中，排序只影响谁先谁后；
- 如果缓存容量紧张、请求前缀差异大，LPM 能避免“高命中请求在排队时被驱逐”，收益明显。

典型受益场景：

- 多轮对话：老会话的前缀很长，新会话前缀短；优先老会话能避免其长前缀被驱逐；
- Agent 集群：同一任务的多个分支共享长轨迹；
- RAG：同一批文档被多个 query 共享，文档前缀很长；
- 批量评测/RL rollout：同一 prompt 的多次采样。

### 5.3.4 公平性与尾延迟

LPM 的代价是可能让低命中请求长期排队。生产系统需要：

- 等待时间 aging；
- 每租户配额；
- 在线/离线分流；
- 对延迟敏感的请求使用 FCFS 或显式优先级。

vLLM 的 FCFS 更“可预测”，但可能在高缓存压力下错过复用机会。**没有一种策略在所有流量下最优**，这也是两家都提供多种策略的原因。

### 5.3.5 Chunked Prefill 与抢占

两者都支持 chunked prefill，但细节不同：

| 维度 | vLLM | SGLang |
|---|---|---|
| 预算参数 | `max_num_batched_tokens` | `max-prefill-tokens` / `chunked-prefill-size` |
| 混跑 | prefill chunk + decode 同一 batch | 同样支持 |
| 抢占 | 以 recompute 为主 | recompute；配合 RadixCache 减少重算 |
| 缓存对抢占的影响 | 被抢占后重算，可能命中前缀缓存 | 被抢占后重算，可能命中 radix tree |

SGLang 的 RadixCache 在抢占/重算时有一个潜在优势：如果前缀还在树中，重算可以复用；vLLM 的块级缓存也能命中，但受块对齐与驱逐顺序影响。

### 5.3.6 CPU/GPU 重叠

- **vLLM V1**：API Server、EngineCore、Worker 分离，异步调度；调度与执行可以流水化。
- **SGLang**：显式的 overlap scheduler，把下一步的 CPU 调度与当前步的 GPU 计算重叠；提供 `--disable-overlap-schedule` 开关。

两家都在解决同一个问题：小模型、高 QPS 时，CPU 侧的 Python 调度可能比 GPU 前向还慢。实现方式不同，目标一致。实际收益取决于模型大小、batch、CPU 性能与实现成熟度。

---

## 5.4 一个请求的两种走法

用一个具体例子把差异落地。

**设定**：

- 模型：Llama-3-8B，block_size=16；
- 请求：prompt 3000 token，其中前 2000 token 与缓存中的某个前缀相同；输出 200 token；
- 当前 KV 池充足，无抢占。

### 5.4.1 vLLM 的走法

```text
1. Tokenize → 3000 tokens
2. 计算块哈希：
   - 前 2000 token = 125 个完整块
   - 逐块查 cached_block_hash_to_block
   - 命中 125 块，引用计数 +1
3. 剩余 1000 token 需要 prefill：
   - 分配 63 个新块（1000/16 = 62.5 → 63）
   - 最后一个块只用了 8 个 token
4. Prefill 1000 token，写入新块
5. Decode 200 步，每步追加 1 token，必要时分配新块
6. 请求结束：
   - 新生成的完整块注册哈希，供后续复用
   - 所有块引用计数减 1，回到空闲队列
```

### 5.4.2 SGLang 的走法

```text
1. Tokenize → 3000 tokens
2. match_prefix：
   - 在 radix tree 中匹配到 2000 token 的节点
   - 返回 2000 个 KV 索引，引用计数 +1
3. 剩余 1000 token 需要 prefill：
   - 从 token 池分配 1000 个槽位
   - 调用 attention backend，用已匹配的 KV 索引 + 新 token 计算
4. Prefill 1000 token
5. Decode 200 步，每步分配 1 个新槽位
6. 请求结束：
   - 把完整序列（3000 + 200）插入 radix tree
   - 未共享部分的 KV 保留在树中，引用计数归零后可被驱逐
```

### 5.4.3 差异总结

在这个“块对齐、前缀很长”的例子中，两者都复用了 2000 token，差异很小。真正拉开差距的是：

- 前缀不是 16 的整数倍（SGLang 多命中几个 token）；
- 大量分支共享前缀（SGLang 的树节点共享更细）；
- 缓存压力大、需要按命中长度排序（SGLang 开启 LPM 后）；
- 高 QPS 小模型（overlap scheduler 的收益）。

**结论**：不要用“2000 token 共享前缀”这种对两者都友好的例子去证明谁更强；要构造“非对齐、分支多、缓存压力大”的 workload。

---

## 5.5 差异何时真正重要：一个定量框架

### 5.5.1 前缀缓存能省多少

设：

- prompt 长度 $L_p$；
- 命中前缀长度 $L_h$；
- prefill 单位 token 成本 $c_p$（与模型、硬件有关）；
- decode 每 token 成本 $c_d$；
- 输出长度 $L_o$；
- 排队时间 $t_{\text{queue}}$；
- 第一个 token 的采样/传输时间 $t_{\text{first sample}}$。

则：

$$
\text{TTFT} \approx t_{\text{queue}} + c_p (L_p - L_h) + t_{\text{first sample}}
$$

$$
\text{E2E} \approx \text{TTFT} + c_d (L_o - 1)
$$

**这两个式子怎么来的？**

- 一个请求真正要 prefill 的 token 数，是“总 prompt 长度 - 已命中前缀长度” = $L_p - L_h$；
- 所以 prefill 时间是 $c_p (L_p - L_h)$；
- 加上排队和第一个 token 的采样/传输，就是 TTFT；
- 第一个 token 已经包含在 TTFT 里了，后面只剩 $L_o - 1$ 个 token 要 decode，所以 E2E 再加 $c_d (L_o - 1)$。

前缀缓存对 TTFT 的节省比例约为：

$$
\frac{\Delta \text{TTFT}}{\text{TTFT}} \approx \frac{c_p L_h}{c_p L_p + t_{\text{queue}}}
$$

**怎么理解这个比例？**

- 没缓存时，prefill 要花 $c_p L_p$；
- 有缓存时，只花 $c_p (L_p - L_h)$；
- 省下的就是 $c_p L_h$；
- 分母用“没缓存时的 TTFT”做基准，忽略 $t_{\text{first sample}}$，所以得到上面的近似。

对 E2E 的节省比例则被 decode 稀释：

$$
\frac{\Delta \text{E2E}}{\text{E2E}} \approx \frac{c_p L_h}{c_p L_p + c_d L_o}
$$

分母里多了 decode 时间 $c_d L_o$，而分子仍然是省下的 prefill 时间 $c_p L_h$。输出越长，分母越大，比例越小。

**推论**：

- 输出越长，前缀缓存对 E2E 的相对收益越小；
- 输出越短（分类、路由、JSON 抽取、工具调用），前缀缓存越值钱；
- 排队时间越长，缓存节省越容易被掩盖。

### 5.5.2 一个数值例子

假设 $c_p = 0.01$ ms/token，$c_d = 0.05$ ms/token，$L_p = 3000$，$L_h = 2000$，$L_o = 200$，并忽略排队与采样时间。

**无缓存**：

$$
\begin{aligned}
\text{TTFT} &= 0.01 \times 3000 = 30\text{ ms} \\
\text{E2E} &= 30 + 0.05 \times (200 - 1) \\
&= 30 + 9.95 \\
&= 39.95\text{ ms} \approx 40\text{ ms}
\end{aligned}
$$

**有缓存（命中 2000 token）**：

实际 prefill token 数：

$$
3000 - 2000 = 1000
$$

$$
\begin{aligned}
\text{TTFT} &= 0.01 \times 1000 = 10\text{ ms} \\
\text{E2E} &= 10 + 9.95 = 19.95\text{ ms} \approx 20\text{ ms}
\end{aligned}
$$

**节省**：

TTFT 节省比例：

$$
\frac{30 - 10}{30} = 66.7\%
$$

E2E 节省比例：

$$
\frac{39.95 - 19.95}{39.95} \approx 50.1\%
$$

如果 $L_o = 2000$：

$$
\begin{aligned}
\text{E2E without cache} &= 30 + 0.05 \times 1999 \\
&\approx 30 + 99.95 = 129.95\text{ ms} \\
\text{E2E with cache} &= 10 + 99.95 = 109.95\text{ ms} \\
\text{savings} &= \frac{129.95 - 109.95}{129.95} \approx 15.4\%
\end{aligned}
$$

可以看到：前缀缓存省下的 20 ms 是固定的，但输出从 200 增加到 2000 后，E2E 的分母变大，节省比例从 50% 降到 15%。**不是缓存效果变差了，而是 decode 占比变高了。**

这解释了为什么 SGLang 在 Agent、RAG、短输出结构化任务中声量很大，而在“长文创作”类负载中，两家差距会被 decode 稀释。

### 5.5.3 影响差异大小的因素清单

*表 5-1　影响差异大小的因素*

| 因素 | 差异变大 | 差异变小 |
|---|---|---|
| 共享前缀比例 | 高（>50%） | 低（<10%） |
| 前缀长度对齐 | 不对齐（非 16 倍数） | 对齐 |
| 分支数量 | 多（树状/并行采样） | 少（单链） |
| 缓存压力 | 大（频繁驱逐） | 小（缓存充足） |
| 输出长度 | 短 | 长 |
| 请求到达模式 | 突发、同时到达 | 均匀、串行 |
| 模型大小 | 小（prefill 占比高、CPU 开销敏感） | 大（decode 主导） |
| SLO 关注点 | TTFT | 长输出 E2E |

---

## 5.6 执行层对比

### 5.6.1 Attention Backend

| 能力 | vLLM | SGLang |
|---|---|---|
| FlashAttention | 支持（FA2/FA3） | 支持 |
| FlashInfer | 支持 | 深度使用，许多模型默认 |
| Triton | 支持 | 支持 |
| FlexAttention | 支持 | 视版本 |
| MLA（DeepSeek） | FlashMLA/CUTLASS 等 | FlashMLA/CUTLASS，DP attention 配套 |
| 自定义 mask/滑动窗口 | 支持较广 | 支持主流模型 |
| 后端选择 | `--attention-backend` | `--attention-backend` |

SGLang 与 FlashInfer 社区关系更紧，部分 kernel 会先在其落地；vLLM 的后端覆盖更广，长尾模型/硬件更稳。**具体模型上谁快，必须实测。**

### 5.6.2 CUDA Graph 与编译

| 维度 | vLLM | SGLang |
|---|---|---|
| Decode CUDA Graph | 支持，按 batch size 分桶 | 支持，按 batch size 分桶 |
| Prefill 图/编译 | piecewise CUDA Graph + torch.compile | 可选 torch.compile；主要靠 CUDA Graph |
| 动态形状处理 | padding + 分段 | padding + 分桶 |
| 调试开关 | `--enforce-eager` | `--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph` |

两家都能把 decode 的 kernel launch 开销压到很低；差异更多在“哪些模型/后端组合被充分调优”。

### 5.6.3 采样与结构化输出

| 能力 | vLLM | SGLang |
|---|---|---|
| GPU 采样 | 支持 | 支持 |
| 约束解码后端 | xgrammar / guidance（outlines 视版本） | XGrammar 为主 |
| 压缩 FSM | 视后端 | 支持 |
| Jump-forward | 视后端 | 支持 |
| JSON Schema / Regex / EBNF | 支持 | 支持，原生接口更完整 |
| 语法编译重叠 | 部分支持 | 强调与 GPU 计算重叠 |

结构化输出是 SGLang 的传统强项，但 vLLM 通过 xgrammar 也能获得很好的性能。对于“输出必须合法 JSON”的业务，两者都可用，差异在细节与成熟度。

### 5.6.4 Detokenization 与流式

- vLLM：API Server 进程负责 detokenize；V1 做了增量处理与异步。
- SGLang：独立的 DetokenizerManager，与调度/执行流水化；对高 QPS 流式场景更友好。

中文/emoji 的增量 detokenize 是常见坑：一个 UTF-8 字符可能跨 token，必须缓存不完整字节，不能直接 `decode` 每个 token。两家都处理了，但自研服务时容易踩。

---

## 5.7 分布式能力对比

| 能力 | vLLM | SGLang |
|---|---|---|
| TP | 支持，成熟 | 支持，成熟 |
| PP | 支持 | 支持 |
| DP | 支持，DP Coordinator | 支持 |
| EP（MoE） | 支持 | 支持，DeepSeek 大规模经验 |
| DP Attention | 视模型/版本 | 明确支持（DeepSeek 类） |
| EPLB | 支持（较新版本，`--enable-eplb`） | 支持，`--enable-eplb` |
| PD 分离 | 支持，多种 KV connector | 支持，mooncake/nixl 等 |
| 分层缓存 | 有 CPU offload / connector 生态 | HiCache（GPU/CPU/磁盘） |
| 多机 | Ray 等 | 多机 TP/EP、PD |
| 路由 | 外部网关/llm-d | 自带 cache-aware router |

vLLM 的分布式生态更“通用”，与 Ray、KServe、llm-d 等集成多；SGLang 在大规模 MoE、PD 分离、缓存感知路由上有更完整的“开箱”方案。选型时要看你的集群规模与模型类型。

---

## 5.8 总对比表

表 5-2 把前面几节的差异收拢到一张表里，便于横向查阅。

*表 5-2　vLLM 与 SGLang 总对比*

| 维度 | vLLM | SGLang |
|---|---|---|
| 核心论文 | PagedAttention (SOSP'23) | RadixAttention / SGLang (NeurIPS'24) |
| KV 单位 | 固定块（长期默认 16，0.34.0 起默认 64） | 分页/槽位，`page_size` 默认 1 |
| 索引 | block table | radix tree |
| 前缀复用 | 块级哈希 | token 级树匹配 |
| 共享 | 引用计数 + COW | 引用计数 + 节点共享 |
| 驱逐 | 块 LRU | 叶子 LRU |
| 默认调度 | FCFS | FCFS（可切换 LPM 等缓存感知策略） |
| 前缀感知调度 | 否（缓存命中仍生效） | 默认否；开启 LPM/DFS-weight 后是 |
| Chunked prefill | 支持 | 支持 |
| 抢占 | recompute 为主 | recompute + 缓存复用 |
| CPU/GPU 重叠 | V1 异步架构 | overlap scheduler |
| 结构化输出 | xgrammar/guidance（outlines 视版本） | XGrammar + 压缩 FSM + jump-forward |
| 投机解码 | draft/ngram/EAGLE/MTP | EAGLE/draft/MTP 等 |
| 多模态 | 覆盖广 | 主流模型覆盖 |
| 量化 | 覆盖广 | 主流格式 |
| PD 分离 | 支持 | 支持，大规模经验 |
| 分层缓存 | connector/offload 生态 | HiCache |
| 路由 | 外部 | 自带 cache-aware router |
| 生态 | 大而全 | 性能/场景导向 |
| 典型强项 | 通用 serving、长尾模型、硬件后端 | 前缀密集、Agent/RAG、结构化输出、大规模 MoE |

---

## 5.9 什么时候不用纠结

不是所有场景都值得做“vLLM vs SGLang”的深度对比。以下情况，差异可能小到可以忽略：

| 情况 | 为什么不用纠结 |
|---|---|
| 前缀共享低于 10% | RadixAttention 的细粒度复用没有用武之地，两者都接近“无缓存” |
| 输出很长（如 2000+ token） | decode 占主导，前缀缓存节省的 prefill 被稀释 |
| 缓存容量充足、请求基本串行 | 块级缓存也能命中，LPM 的排序影响小 |
| 单请求、batch=1 的低延迟场景 | 瓶颈在 kernel/backend 与模型本身，调度差异很小 |
| 流量很小（每天几百请求） | 省下的几毫秒不值得增加运维复杂度 |
| 目标模型/硬件只被一个引擎支持 | 没得选，先解决“能不能跑” |
| 团队只熟悉一个引擎 | 运维、排错、升级成本通常高于 10% 的性能差异 |
| 已有稳定生产系统 | 迁移风险 > 潜在收益，除非瓶颈明确 |

如果以上都不符合，再回到 5.5 的公式，估算：

1. 前缀命中能省多少 prefill；
2. prefill 占 TTFT/E2E 的比例；
3. 缓存压力是否会导致驱逐；
4. 你的 SLO 更关注 TTFT 还是 TPOT。

**先算账，再选型；不要因为“token 级”三个字就迁移。**

---

## 5.10 常见误解澄清

1. **“SGLang 有 RadixAttention，所以一定比 vLLM 快。”**  
   错。前缀完全不共享、输出很长、缓存压力小时，RadixAttention 的树维护甚至可能是负收益。性能取决于 workload。

2. **“vLLM 的前缀缓存是块级的，所以基本没用。”**  
   错。块级缓存对长前缀、块对齐的负载命中率很高；块边界损失通常很小（块越小越小，16/64 依版本而定）。

3. **“SGLang 不用分页，所以没有碎片。”**  
   不准确。SGLang 底层同样要按页或槽位管理 KV；当 `page_size=1` 时确实没有块内碎片，但仍有树节点元数据开销。它可以配置更大的 `page_size`，此时也会出现页内碎片。

4. **“vLLM 是 FCFS，所以完全不考虑缓存。”**  
   错。vLLM 会做前缀缓存命中，只是 waiting 队列不按命中长度重排。

5. **“某个 benchmark 说 A 快 5 倍，所以 A 全面更好。”**  
   错。benchmark 的模型、输入输出长度、并发、前缀分布、版本、硬件都会影响结论。第 7 章专门讲这个。

6. **“两个引擎只能二选一。”**  
   错。可以用网关路由：前缀密集/结构化请求走 SGLang，通用/长尾模型走 vLLM；也可以在 PD 分离中混用不同实现（需接口适配）。

---

## 5.11 本章小结

- vLLM：块级分页 + 块哈希前缀缓存 + FCFS/token budget；优势是通用、简单、kernel 友好。
- SGLang：token 级基数树索引 + 底层分页/槽位池 + 默认 FCFS（可选 LPM/DFS-weight 等缓存感知调度）+ overlap；优势是细粒度前缀复用、前缀密集负载、结构化输出。
- 两者都支持 continuous batching、chunked prefill、CUDA Graph、投机解码、并行与 PD 分离；差异在数据结构、可选调度策略与工程实现。
- 差异是否重要，取决于共享前缀比例、对齐、分支数、缓存压力、输出长度、SLO。
- 选型不要背结论，要用自己的 workload 做对照实验（第 7、8 章）。

> **用大白话再说一遍**：vLLM 像固定大小的收纳盒，整齐、通用、好管理；SGLang 像按搭配挂衣杆，相同前缀共享得细、找衣服快，但树枝要维护。衣服（请求）搭配越相似、共享越多，挂衣杆越有优势；如果每件衣服都完全不一样，收纳盒反而更省心。

---

## 思考题

1. 共享前缀 2005 token、block_size=16 时，vLLM 与 SGLang 各能复用多少 token？如果共享前缀是 2016 token，结论会变吗？
2. 为什么 vLLM 的块级 COW 在“大量短分支”场景下比 SGLang 的节点共享更贵？请从复制粒度角度解释。
3. LPM 在什么条件下会提高吞吐，却在什么条件下会伤害 P99 TTFT？请构造两个流量场景。
4. 用第 5.5 节的公式估算：$L_p=8000$、$L_h=7000$、$L_o=50$、$c_p=0.008$ ms/token、$c_d=0.04$ ms/token 时，前缀缓存对 TTFT 和 E2E 的节省比例。
5. 如果两个引擎都支持 chunked prefill，为什么 SGLang 的 LPM 与 chunk 切分组合会更复杂？需要额外考虑哪些约束？
6. “vLLM 的 FCFS 更公平”是否绝对？在什么情况下 FCFS 也会造成不公平？
7. 设计一个实验，能够分离“RadixAttention 的细粒度复用”与“LPM 的调度排序”各自的贡献。提示：控制前缀对齐与到达顺序。
8. 你认为未来两个引擎会继续分化还是收敛？请从 KV 数据结构、调度策略、生态三个角度论证。

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 6 章 功能、模型与生态对比 →](06-features-ecosystem.md)