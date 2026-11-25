# 共享前缀对照实验脚本

`bench_prefix.py` 用 OpenAI 兼容接口向任一推理服务发送流式请求，测量客户端侧的 TTFT、E2E 与近似 TPOT，并对比两种模式：

- `shared`：所有请求共享同一段长前缀，各自只有不同的后缀；
- `unique`：每条请求使用独立随机前缀，几乎没有前缀复用。

它适合做第 5、7、8 章中的定性对照实验，不替代官方压测工具，也不用于生产容量规划。

## 安装

```bash
pip install -r requirements.txt
```

## 启动服务

vLLM（默认 8000 端口）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen --port 8000
```

SGLang（默认 30000 端口）：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen --port 30000
```

## 运行

以下命令在 `code/` 目录下执行；如果在仓库根目录，把 `bench_prefix.py` 换成 `code/bench_prefix.py`。

```bash
# vLLM：共享前缀
python bench_prefix.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --mode shared \
  --shared-words 1500 \
  --suffix-words 200 \
  --num-requests 64 \
  --concurrency 16 \
  --max-tokens 64

# vLLM：独立前缀（对照）
python bench_prefix.py \
  --base-url http://localhost:8000/v1 \
  --model qwen \
  --mode unique \
  --shared-words 1500 \
  --suffix-words 200 \
  --num-requests 64 \
  --concurrency 16 \
  --max-tokens 64
```

把 `--base-url` 换成 `http://localhost:30000/v1` 即可测 SGLang。

## 输出

```text
mode=shared requests=64 ok=64 failed=0 concurrency=16 wall=12.34s
TTFT ms : p50=   120.3  p90=   180.5  p99=   250.1  mean=   135.7
E2E  ms : p50=  1200.2  p90=  1500.4  p99=  1800.9
TPOT ms : p50=    16.8  p90=    18.2  p99=    19.5  (approx)
output chunks/s (approx tokens/s): 330.12
```

- `TTFT`：请求发出到第一个非空 content chunk 的时间；
- `E2E`：到最后一个非空 content chunk 的时间；
- `TPOT`：`(E2E - TTFT) / (chunks - 1)` 的近似值，假设一个 chunk 对应一个 token；
- `output chunks/s`：用 chunk 数近似输出 token 数，仅用于相对比较；
- `--json-out result.json` 可保存每次请求的原始数据。

## 实验设计建议

1. **固定变量**：同一模型、同一精度、同一 GPU、同一 `max_tokens`、同一并发；
2. **对照**：`shared` vs `unique`，分别对两个引擎运行；
3. **对齐与非对齐**：`--shared-words` 取不同值，观察块级缓存（vLLM，`block_size` 历史 16、0.34.0 起默认 64）与 token 级缓存（SGLang，`page_size=1` 时）的差异；
4. **缓存状态**：`--warmup 0` 测冷启动；默认 warmup 会填充缓存，测稳态；`unique` 模式的 warmup 使用未参与测量的随机前缀，避免预热污染对照组；
5. **重复**：至少 3 次，交换引擎启动顺序，报告中位数与 P99；
6. **客户端**：并发高时用多进程/多机，避免 Python 客户端成为瓶颈。

## 注意事项

- 用“词数”近似 token 数，不同 tokenizer 会有偏差；
- 客户端测得的 TTFT 包含网络与 Python 开销，和引擎内部指标不同；
- 突发提交（默认 `--request-rate 0`）与泊松到达不同，后者更接近真实流量；
- 共享前缀的 warmup 会预热缓存；如果要测“冷缓存下的第一条请求”，请设 `--warmup 0` 并重启服务；
- 结果受模型、硬件、版本、参数影响，**不要跨环境比较绝对数字**。
