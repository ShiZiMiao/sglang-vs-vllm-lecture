# 第 8 章 动手实践：启动、调用、压测与调参

> **本章目标**：把前七章的原理落到命令行。你将启动两个引擎、用 OpenAI 兼容接口调用、做一次共享前缀对照实验，并学会排查最常见的 OOM、吞吐低、延迟高问题。
>
> **先修知识**：会开终端、装 Python 包即可。没写过命令行工具也没关系，8.0 节会解释 `--flag`、`python -m`、端口、base_url。
>
> **一句话概括**：先让服务跑起来，再用 OpenAI API 调通，最后用压测工具观察 TTFT/TPOT/吞吐，按瓶颈调参。
>
> **给零基础同学**：命令不要死记，先看 `--help`；每个参数都问一句“它控制什么资源”。本章的排错表可以当操作手册。

> **版本提醒**：本章命令按 **vLLM 0.34.0 / SGLang 0.5.23** 核对（2026-11-25）。参数名可能随版本变化，执行前先看 `--help`。示例模型使用 `Qwen/Qwen2.5-7B-Instruct`，你可以换成自己有权访问的模型。

---

## 8.0 命令行零基础

如果你平时只用过 Jupyter Notebook，这一节先补几个概念。

### 8.0.1 `python -m` 是什么

```bash
python -m sglang.launch_server --model-path ...
```

意思是“把 `sglang.launch_server` 这个模块当成脚本运行”。等价于找到这个模块的入口并执行它。`python -m sglang.benchmark.serving` 同理。

### 8.0.2 `--flag value` 是什么

命令行参数通常长这样：

```bash
--port 8000
--model Qwen/Qwen2.5-7B-Instruct
--tensor-parallel-size 1
```

- `--port` 是参数名，`8000` 是值；
- 有的参数是开关，不需要值，如 `--enable-prefix-caching`；
- 有的参数可以写成 `--port=8000`；
- 记不住就看帮助：

```bash
vllm serve --help
python -m sglang.launch_server --help
```

### 8.0.3 端口、host 与 base_url

- **端口（port）**：一台机器上不同服务的“门牌号”。vLLM 默认 8000，SGLang 默认 30000；
- **host**：服务监听哪个网卡。`127.0.0.1` 只允许本机，`0.0.0.0` 允许局域网；
- **base_url**：客户端访问服务的地址，例如 `http://localhost:8000/v1`。

如果 8000 被占用，可以换 `--port 8001`，客户端 base_url 也要跟着改。

### 8.0.4 虚拟环境

建议每个引擎一个独立环境，避免依赖冲突：

```bash
python -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
```

Windows PowerShell 则是：

```powershell
python -m venv $HOME\venvs\vllm
& "$HOME\venvs\vllm\Scripts\Activate.ps1"
```

激活后，`python`、`pip`、`vllm` 都指向这个环境。

### 8.0.5 Docker 参数速查

```bash
docker run --gpus all --ipc=host -p 8000:8000 vllm/vllm-openai:latest --model ...
```

| 参数 | 作用 |
|---|---|
| `--gpus all` | 把 GPU 暴露给容器 |
| `-p 8000:8000` | 宿主机 8000 → 容器 8000 |
| `--ipc=host` | 共享内存，多进程/多卡通信需要 |
| `--shm-size 32g` | 不用 host IPC 时，给足共享内存 |
| `-v /path:/path` | 挂载模型/数据目录 |

> **零基础提示**：不要看到命令就复制粘贴。先看 `--model`/`--model-path` 指向的模型是否存在、端口是否被占用、显存是否够。大部分“跑不起来”都是这三件事。

---

## 8.1 环境准备

### 8.1.1 硬件与系统

| 项目 | 建议 |
|---|---|
| GPU | 1 × 24 GB 以上（7B BF16 权重约 14 GB，加运行时开销建议 24 GB） |
| 系统 | Ubuntu 22.04/24.04，或 Windows + WSL2 |
| 驱动 | 支持所用 CUDA 版本的 NVIDIA 驱动 |
| 内存 | 32 GB 以上 |
| 磁盘 | 100 GB 以上（模型 + 编译缓存） |

先确认 GPU 可见：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 8.1.2 Python 环境

建议每个引擎一个独立虚拟环境，避免依赖冲突：

```bash
# vLLM
python -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
pip install -U pip
pip install "vllm==0.34.0"          # 讲义基线；省略版本号则安装最新版

# SGLang
python -m venv ~/venvs/sglang
source ~/venvs/sglang/bin/activate
pip install -U pip
pip install "sglang[all]==0.5.23"   # 讲义基线；省略版本号则安装最新版
```

> 本讲义按 **vLLM 0.34.0** 与 **SGLang 0.5.23** 讲解（2026-11-25 核对）。锁版本是为了让默认值与行为可复现；不锁版本时，`pip` 会装到更新的版本，参数默认值可能已经变化。

如果使用 `uv`（两个项目都推荐）：

```bash
uv venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
uv pip install "vllm==0.34.0"

uv venv ~/venvs/sglang && source ~/venvs/sglang/bin/activate
uv pip install "sglang[all]==0.5.23"
```

> SGLang 的官方安装方式迭代较快；若上述命令失败，以 [官方安装文档](https://docs.sglang.io/docs/get-started/install) 或 Docker 镜像为准。

### 8.1.3 Docker（可选，推荐生产）

```bash
# vLLM
# 镜像 tag 用 latest 只是示例；要复现讲义基线，请换成 v0.34.0（以 Docker Hub 实际 tag 为准）
docker run --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct

# SGLang（同理，可把 latest 换成 v0.5.23）
docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 30000
```

`--ipc=host` 或足够的 `--shm-size` 很重要：多进程/多卡通信会用到共享内存，默认 64 MB 常导致挂起或 NCCL（NVIDIA 多卡通信库）错误。

---

## 8.2 启动服务

> 下面两组命令按 **24 GB 卡 + 7B 模型 + 8K 上下文**给出保守值。直接照搬到更大显存可以再把并发调高；反过来，把 128 并发复制到 24 GB 卡通常会 OOM。如果 8 并发仍然 OOM，把 `max-model-len`/`context-length` 降到 4096，或把并发降到 4。
>
> 命令里的 `--host 0.0.0.0` 方便局域网访问，但也会扩大暴露面。本机实验可以改成 `127.0.0.1`；对外服务要加鉴权、限流或网关，不要直接暴露到公网。

### 8.2.1 vLLM

```bash
# 24 GB 卡 + 7B BF16 + 8K 上下文的保守配置；参数含义见下表
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
```

> vLLM 0.34.0 中 `enable_prefix_caching` 默认就是 `True`，这里的 `--enable-prefix-caching` 只是显式声明；要关闭用 `--no-enable-prefix-caching`。SGLang 的 Radix Cache 默认开启，无需额外 flag。

参数解释：

| 参数 | 作用 | 调参直觉 |
|---|---|---|
| `--tensor-parallel-size` | 张量并行度 | 单卡放不下/想提速时增大；需要 NVLink |
| `--gpu-memory-utilization` | 允许使用的显存比例 | 0.85～0.92 常见；太高易 OOM |
| `--max-model-len` | 最大上下文长度 | 越小越省 KV；按业务设置 |
| `--max-num-seqs` | 单步最大序列数 | 越大吞吐越高、TPOT 可能变差 |
| `--max-num-batched-tokens` | 单步 token 预算 | 控制 prefill 峰值与 chunk 大小 |
| `--enable-prefix-caching` | 开启块级前缀缓存 | V1 中可能默认开启；按版本确认 |
| `--enforce-eager` | 关闭 CUDA Graph | 仅调试用，会损失性能 |
| `--quantization` | 量化格式 | fp8/awq/gptq 等，需模型支持 |

**默认值与常见坑（核对日期 2026-11-25，vLLM 0.34.0）**：`--tensor-parallel-size` 默认 1；`--max-model-len` 未指定时由模型 config 推导（不能超过模型自带上限）；`--gpu-memory-utilization` 的字段默认是 `None`——在默认启用 extensible KV cache 的 CUDA/ROCm 上解析为 1.0，否则为 0.92；`--max-num-seqs` 字段默认 128，`--max-num-batched-tokens` 字段默认 2048；`--scheduling-policy` 默认 `fcfs`。默认值只是字段值，真实并发还会被显存与 KV 池限制。`--gpu-memory-utilization` 不是“权重占比”，它约束引擎可用的总显存；设得过高会挤压激活峰值与 KV 池，设得过低则 KV 池变小、并发上不去。`--tensor-parallel-size` 需要能整除注意力头数与 MLP 中间维度，GQA 模型还要看 KV 头数。逐项默认值可查 [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.34.0/configuration/engine_args.html)。

验证：

```bash
curl http://localhost:8000/v1/models
```

### 8.2.2 SGLang

```bash
# 与上面 vLLM 配置对齐的保守配置；参数含义见下表
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 1 \
  --mem-fraction-static 0.88 \
  --context-length 8192 \
  --max-running-requests 8 \
  --max-prefill-tokens 8192
```

参数解释：

| 参数 | 作用 | 调参直觉 |
|---|---|---|
| `--tp` | 张量并行度 | 同 vLLM |
| `--mem-fraction-static` | 静态显存比例 | 0.85～0.90 常见；越高 KV 池越大 |
| `--context-length` | 最大上下文 | 按业务设置 |
| `--max-running-requests` | 最大并发序列 | 越大吞吐越高、TPOT 可能变差 |
| `--max-prefill-tokens` | 单步 prefill token 预算 | 控制 chunk 与激活峰值 |
| `--schedule-policy` | 调度策略 | 默认 `fcfs`；前缀密集可试 `lpm`/`dfs-weight`，注意公平性 |
| `--chunked-prefill-size` | 单 chunk 最大 token | 长 prompt 时限制峰值 |
| `--disable-radix-cache` | 关闭前缀树缓存 | 前缀完全不共享时可关 |
| `--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph` | 分阶段关闭 CUDA Graph | 仅调试用（0.5.23 起；旧版为 `--disable-cuda-graph`） |
| `--disable-overlap-schedule` | 关闭重叠调度 | 通常默认开启；仅调试/排查时关闭 |
| `--quantization` | 量化格式 | fp8/awq/gptq 等 |

**默认值与常见坑（核对日期 2026-11-25，SGLang 0.5.23）**：`--tp` 默认 1；`--context-length` 未指定时读模型 `config.json`；`--mem-fraction-static` 默认 `None`，由引擎按模型与显存自动推导（官方文档注明显存无法探测时用 0.88）；`--max-running-requests` 默认自动；`--max-prefill-tokens` 默认 16384；`--chunked-prefill-size` 默认 `None`（自动），设为 `-1` 关闭；`--schedule-policy` 默认 `fcfs`；`--page-size` 默认 1。`--mem-fraction-static` 只控制静态池，激活峰值仍需额外余量，不要按 1.0 设置；`--context-length` 也不能超过模型自带的最大上下文。逐项默认值可查 [SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)。

验证：

```bash
curl http://localhost:30000/v1/models
```

### 8.2.3 参数映射表

两个引擎的参数对应关系见表 8-1。

*表 8-1　vLLM 与 SGLang 常用参数映射*

| 含义 | vLLM | SGLang |
|---|---|---|
| 模型路径 | `--model` | `--model-path` |
| 对外模型名 | `--served-model-name` | `--served-model-name` |
| 张量并行 | `--tensor-parallel-size` | `--tp` |
| 显存比例 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 最大上下文 | `--max-model-len` | `--context-length` |
| 最大并发序列 | `--max-num-seqs` | `--max-running-requests` |
| 单步 token 预算 | `--max-num-batched-tokens` | `--max-prefill-tokens` |
| 前缀缓存 | 默认开启（`--no-enable-prefix-caching` 关闭，视版本） | 默认开启（`--disable-radix-cache` 关闭） |
| CUDA Graph | 默认开启（`--enforce-eager` 关闭） | 默认开启（`--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph` 关闭） |
| 端口 | `--port`（默认 8000） | `--port`（默认 30000） |
| API Key | `--api-key` | `--api-key` |

> **注意**：`max-num-batched-tokens` 与 `max-prefill-tokens` 只是近似对应。vLLM 的预算通常同时覆盖 prefill 与 decode token；SGLang 的 `max-prefill-tokens` 主要限制单步 prefill。对齐参数时要看实际语义，不能只按名字映射。

---

## 8.3 用 OpenAI 兼容接口调用

### 8.3.1 Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # SGLang 改成 :30000
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "你是一个严谨的 AI 助教。"},
        {"role": "user", "content": "用三句话解释什么是 KV Cache。"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
print(resp.usage)
```

### 8.3.2 流式调用

```python
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "写一个快速排序的 Python 实现。"}],
    temperature=0.2,
    max_tokens=512,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### 8.3.3 curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己。"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### 8.3.4 结构化输出

```python
resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "生成一个用户信息"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "age", "tags"]
            }
        }
    },
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

两家都支持 OpenAI 的 `response_format`；SGLang 原生接口还支持 `regex`、`ebnf` 等参数。若使用 vLLM 的 `guided_json` 等旧接口，可能通过 `extra_body` 传入，具体以版本文档为准。

---

## 8.4 压测：从一条命令开始

### 8.4.1 vLLM

```bash
vllm bench serve \
  --model qwen \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 10 \
  --max-concurrency 8
```

- `--request-rate 10`：每秒 10 个请求（open-loop）；
- `--request-rate inf`：t=0 全部发出（open-loop 突发），用于看极限吞吐；它不是闭环，排队会被放大；
- `--num-prompts 200`：先小规模验证，再放大；
- 结果会输出 TTFT/TPOT/ITL/E2E 的均值与分位数。

### 8.4.2 SGLang

```bash
# 新版入口；旧版可能是 python -m sglang.bench_serving
python -m sglang.benchmark.serving \
  --backend sglang \
  --model qwen \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 10 \
  --max-concurrency 8
```

如果模块路径或参数名不一致，执行 `python -m sglang.benchmark.serving --help`（或旧入口 `python -m sglang.bench_serving --help`）查看当前版本支持项。

> 这里 `--model qwen` 对应服务端的 `--served-model-name`；`--tokenizer` 指向真实模型路径，避免客户端拿 `qwen` 去本地找 tokenizer 时失败。vLLM 的 `vllm bench serve` 同理。

### 8.4.3 观察 GPU 与指标

```bash
# 每 0.5 秒刷新 GPU 利用率/显存
watch -n 0.5 nvidia-smi

# vLLM 指标
curl -s http://localhost:8000/metrics | head -50

# SGLang 指标
curl -s http://localhost:30000/metrics | head -50
```

重点看：

- GPU 利用率低、同时 CPU 占用高 → 可能是 CPU/调度瓶颈或 batch 太小；
- GPU 利用率高但吞吐仍然低 → 可能是显存带宽或 kernel 效率瓶颈；
- 不要只用“GPU 利用率 < 80%”判断瓶颈：decode 阶段本来就可能是带宽受限，SM 利用率低并不自动等于 CPU 瓶颈；
- 显存是否接近上限（OOM 前兆）；
- 队列长度、cache hit、TTFT/TPOT 指标；
- 是否出现大量重算/抢占。

---

## 8.5 共享前缀对照实验

仓库中的 `code/bench_prefix.py` 是一个最小压测脚本，用来观察前缀共享对 TTFT 的影响。

### 8.5.1 运行

```bash
# 对 vLLM
python code/bench_prefix.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --mode shared \
  --shared-words 1500 \
  --suffix-words 200 \
  --num-requests 64 \
  --concurrency 8 \
  --max-tokens 64

# 对 SGLang（若想观察缓存感知调度，启动服务时加 --schedule-policy lpm）
python code/bench_prefix.py \
  --base-url http://localhost:30000/v1 \
  --model qwen \
  --mode shared \
  --shared-words 1500 \
  --suffix-words 200 \
  --num-requests 64 \
  --concurrency 8 \
  --max-tokens 64
```

对照实验：

```bash
# 无共享前缀（每条请求前缀都不同）
python code/bench_prefix.py --mode unique ...

# 共享前缀
python code/bench_prefix.py --mode shared ...
```

### 8.5.2 看什么

- `shared` 模式下，warmup（或第一条请求）会填充缓存，后续请求的 TTFT 应显著下降；
- `unique` 模式下，缓存无法复用，TTFT 应接近“无前缀命中”的稳态；
- 对 vLLM，若共享前缀长度不是 16 的倍数，观察是否有少量额外 prefill；
- 对 SGLang，默认 FCFS 下主要观察 token 级命中；若启动时加了 `--schedule-policy lpm`，再观察高命中请求是否优先、P99 TTFT 是否更稳；
- 输出长度越长，TTFT 的差异越容易被 E2E 稀释。

### 8.5.3 注意事项

- 脚本用“词数”近似 token 数，不同 tokenizer 会有偏差；
- `shared` 模式的 warmup 会预热共享前缀；`unique` 模式使用未参与测量的随机前缀预热，避免污染对照；`--warmup 0` 可关闭；
- 测量的是客户端 TTFT，包含网络与 Python 开销；
- 并发是“一次性提交 + 线程池”，更接近突发流量，不是严格泊松到达；
- 要得到可信结论，至少重复 3 次，交换引擎启动顺序。

---

## 8.6 调参与排错

### 8.6.1 OOM（显存不足）

**症状**：启动时 `CUDA out of memory`，或运行中 OOM。

**排查顺序**：

1. 降低 `--gpu-memory-utilization` / `--mem-fraction-static`（如 0.9 → 0.85 → 0.8）；
2. 降低 `--max-model-len` / `--context-length`；
3. 降低 `--max-num-seqs` / `--max-running-requests`；
4. 降低 `--max-num-batched-tokens` / `--max-prefill-tokens`，开启/调小 chunked prefill；
5. 使用量化（FP8/AWQ/GPTQ）；
6. 增大 TP；
7. 关闭 CUDA Graph 或降低最大捕获 batch（图捕获本身占显存）；
8. 检查是否有其他进程占用 GPU。

> 如果是在“加载权重”阶段就 OOM（还没开始分配 KV），降低显存比例通常没用；需要量化、换更小的模型或增大 TP。

### 8.6.2 吞吐低

| 可能原因 | 检查 | 处理 |
|---|---|---|
| 并发不足 | 请求率/并发数太低 | 提高并发或 request-rate |
| batch 太小 | `max_num_seqs` 太小 | 适当增大 |
| CPU 瓶颈 | GPU 利用率锯齿、CPU 100% | 开 CUDA Graph、overlap、减少 Python 开销 |
| 前缀缓存未命中 | cache hit 指标低 | 检查前缀是否共享、缓存是否被驱逐 |
| Attention backend 不合适 | 对比不同 backend | 换 FlashInfer/FA3/MLA |
| 量化未启用 | dtype/显存 | 用 FP8/AWQ |
| 网络/客户端瓶颈 | 客户端 CPU、带宽 | 多进程压测、换机器 |
| TP 通信瓶颈 | NCCL 时间占比 | 检查 NVLink/拓扑，减少 TP |

### 8.6.3 TTFT 高

- 排队太长：提高并发容量或降低到达率；
- prefill 太长：chunked prefill 能让长 prefill 不再长时间独占 GPU，改善整体 ITL 和公平性，但**单条长请求的 TTFT 可能更长**；要直接降低单条 TTFT，应缩短 prompt、提高前缀命中或增加算力；
- 缓存未命中：检查前缀是否稳定、是否被驱逐、`cache_salt` 是否隔离过度；
- 冷启动：模型首次请求要编译/捕获图，先 warmup；
- 客户端/网络：测量服务端 TTFT 对比。

### 8.6.4 TPOT/ITL 高或抖动

- batch 太大：decode 每步要读更多 KV，TPOT 上升；降低 `max_num_seqs`；
- KV 太长：长上下文挤占带宽；考虑 KV FP8、缩短上下文；
- chunked prefill 太大：prefill chunk 阻塞 decode；调小 chunk；
- TP 通信：all-reduce 延迟；检查 NVLink；
- 投机解码接受率低：关闭或换草稿；
- 调度抢占：频繁 recompute；增加显存/降低并发。

### 8.6.5 常见错误

| 错误 | 原因 | 处理 |
|---|---|---|
| `The model's max seq len is larger than max tokens` | 上下文/显存不够 | 降 `max-model-len` 或加显存 |
| `CUDA error: invalid argument` | kernel/backend 不兼容 | 换 backend、升级驱动 |
| NCCL timeout / hang | 网络/共享内存/防火墙 | `--ipc=host`、`--shm-size`、检查端口 |
| 输出乱码/重复 | chat template 不匹配 | 用模型自带 template，检查 tokenizer |
| 结构化输出不合法 | schema/grammar 不支持 | 简化 schema，换后端 |
| 启动卡在编译/CUDA Graph | 首次编译慢 | 等待；或 `--enforce-eager` 排查 |
| Docker 内共享内存不足 | `/dev/shm` 太小 | `--shm-size=32g` |
| WSL2 显存不足 | 默认内存限制 | 调整 `.wslconfig`，关闭其他进程 |

---

## 8.7 一个最小生产检查清单

上线前至少确认：

- [ ] 固定引擎版本/镜像 tag，记录 commit；
- [ ] 健康检查：`/health` 或 `/v1/models`；
- [ ] 鉴权：`--api-key`，网关层限流；
- [ ] 超时与取消：客户端断开后服务端能释放资源；
- [ ] 监控：TTFT/TPOT/吞吐/队列/显存/cache hit/错误率；
- [ ] 日志：请求 ID、token 数、停止原因、异常栈；
- [ ] 压测：目标并发下 P99 满足 SLO；
- [ ] 降级：量化/小模型/关闭投机解码的预案；
- [ ] 滚动升级：新旧版本共存，灰度切流；
- [ ] 安全：多租户前缀缓存隔离、`cache_salt`、输入长度限制；
- [ ] 成本：每百万 token 的 GPU 成本，而不只是吞吐。

---

## 8.8 常见误区

**误区 1：服务启动成功，就代表能稳定服务。**  
首次请求要编译/捕获 CUDA Graph、填充缓存，可能特别慢；上线前要 warmup，并压测到目标并发。

**误区 2：显存利用率参数越高越好。**  
设得过高，权重 + KV + 激活峰值容易 OOM；设得过低，KV 池太小、并发上不去。要留安全余量。

**误区 3：`max_model_len` 直接设成模型最大值。**  
上下文越长，KV 池能容纳的并发越少。按业务实际长度设置，不要盲目拉满。

**误区 4：压测不 warmup，直接测第一轮。**  
第一轮包含编译、图捕获、冷缓存，TTFT 会异常高。先发 warmup 请求，再统计。

**误区 5：只看吞吐，不看 SLO。**  
高吞吐可能是靠超大 batch 堆出来的，P99 已经爆了。要同时看 TTFT/TPOT P99 和 goodput。

**误区 6：一 OOM 就加卡或量化。**  
先降并发/上下文/chunk 大小，很多 OOM 是参数问题，不是硬件不够。

**误区 7：本机能访问，局域网就能访问。**  
推理服务的 Web 框架常默认监听 `127.0.0.1`；局域网访问要显式绑定 `0.0.0.0` 并放行防火墙。注意这会扩大暴露面，生产环境应加鉴权/网关，不要直接暴露到公网。

---

## 8.9 本章小结

- 两个引擎都提供 OpenAI 兼容接口，启动命令与参数高度对应。
- 关键参数：显存比例、最大上下文、最大并发、token 预算、前缀缓存、CUDA Graph。
- 压测要区分 open-loop 与 closed-loop，先小规模验证再放大。
- 共享前缀实验能直观看到 RadixAttention/块级缓存的价值，但要用对齐/非对齐、共享/不共享多组对照。
- 调参的核心是定位瓶颈：OOM 看显存账本，吞吐低看并发/CPU/缓存，TTFT 高看排队/prefill，TPOT 高看 batch/KV/通信。
- 生产上线要锁版本、做监控、留降级预案。

> **用大白话再说一遍**：先把厨房开起来（启动服务），再点一道菜（API 调用），然后请一批客人试吃（压测），最后看是灶台不够（算力）、仓库门太窄（带宽）、还是座位不够（显存）。调参就是按瓶颈换旋钮，不是把所有旋钮拧到最大。

---

## 思考题

1. 为什么 `--gpu-memory-utilization` 设得过高会导致启动成功但运行中 OOM？请从激活峰值与 KV 增长角度解释。
2. 在 24 GB 卡上部署 7B BF16 模型，`max-model-len=8192`、`max-num-seqs=64`，估算 KV 是否够用。如果不够，优先调哪个参数？
3. 为什么 `--max-num-batched-tokens` 既影响 TTFT 又影响 ITL？请给出一个调参实验。
4. 用 `bench_prefix.py` 设计实验：共享前缀 1500 词、后缀 200 词，比较 shared/unique 两种模式。预测两条 TTFT 曲线，并解释差异。
5. 如果 vLLM 的 TTFT 比 SGLang 高，但吞吐更高，可能是什么原因？如何验证？
6. Docker 中 NCCL 挂起，为什么 `--shm-size` 可能解决问题？还有哪些排查方向？
7. 设计一个“上线前检查清单”，至少包含 10 项，并说明每项失败后的后果。

> 参考答案与提示见[附录 B](appendix-b-exercises.md)。

---

> **继续阅读**：[第 9 章 源码地图：一次请求在两个引擎中的调用链 →](09-source-code-tour.md)
