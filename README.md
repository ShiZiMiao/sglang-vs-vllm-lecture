# LLM 推理系统：SGLang 与 vLLM 对比讲义

> 面向人工智能专业本科高年级/研究生的系统课讲义  
> 从一次文本生成讲起，逐步建立 LLM 推理引擎的完整心智模型，再深入对比 SGLang 与 vLLM 的设计取舍。

[![在线讲义](https://img.shields.io/badge/%E5%9C%A8%E7%BA%BF%E8%AE%B2%E4%B9%89-shizimiao.github.io-blue)](https://shizimiao.github.io/sglang-vs-vllm-lecture/)
[![docs](https://github.com/ShiZiMiao/sglang-vs-vllm-lecture/actions/workflows/docs.yml/badge.svg)](https://github.com/ShiZiMiao/sglang-vs-vllm-lecture/actions/workflows/docs.yml)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

---

## 1. 这门课讲什么

大语言模型（LLM）的“训练”决定了模型会不会思考，而“推理系统”决定了它能不能便宜、稳定、低延迟地把思考结果交付给用户。vLLM 与 SGLang 是当前开源社区最具代表性的两个 LLM 推理引擎：

- **vLLM**：以 PagedAttention 和通用、成熟的工程生态著称，模型覆盖面广、硬件后端多、生产部署经验丰富。
- **SGLang**：以 RadixAttention、可选的缓存感知调度和结构化程序执行著称，在多轮对话、Agent、RAG、结构化输出、大规模 MoE 部署等场景表现突出。

本讲义不满足于罗列“谁比谁快”的结论，而是希望你能回答三个层次的问题：

1. **原理层**：KV Cache 为什么存在？PagedAttention 与 RadixAttention 各自解决了什么问题？
2. **系统层**：一个请求从 HTTP 到达，到 token 流式返回，中间经过哪些模块？调度器每一步在做什么决策？
3. **工程层**：给定模型、硬件、流量形态和 SLO，应该如何选型、配置、压测和调优？

---

## 2. 目标读者与前置知识

**目标读者**：人工智能、计算机科学相关专业的高年级本科生或研究生；也适合**没有系统学过 Transformer 和推理系统**、但会一点 Python 的同学。

- 会 Python 基础（变量、函数、循环）即可；
- 深度学习背景不是必需的：第 0 章会用“文字接龙”“开会”等类比补齐 token、注意力、Q/K/V、GPU 显存等概念；
- 如果已经学过 Transformer，第 0 章可以快速浏览，把重点放在 KV Cache 与 GPU 瓶颈上；
- 计算机系统背景（虚拟内存、缓存、并发）会在第 1、2 章逐步建立，不需要提前掌握。

如果暂时不熟悉 GPU 体系结构也没关系：第 0、1、2 章会把必需的背景补齐。

---

## 3. 学习目标

完成本讲义后，你应该能够：

- 用 roofline 模型解释为什么 prefill 是计算密集型、decode 是显存带宽密集型；
- 手算给定模型的权重显存、单 token KV Cache 大小和最大并发数；
- 说清 PagedAttention 的 block table、前缀哈希、copy-on-write 与 LRU 驱逐；
- 说清 RadixAttention 的基数树匹配、插入、分裂、引用计数与 LRU 驱逐；
- 对比两种引擎在 KV 管理、调度、前缀复用、执行后端上的差异；
- 独立启动 vLLM/SGLang 服务，用 OpenAI 兼容 API 调用并做可复现压测；
- 根据 workload 特征做出有理有据的选型，而不是背“某个引擎更快”的结论。

---

## 4. 章节地图

| 章节 | 主题 | 核心问题 |
|---|---|---|
| [第 0 章 预备知识：从“文字接龙”到 GPU 服务](docs/00-prerequisites.md) | token、注意力、GPU、指标 | 没学过 Transformer，也能读懂吗？ |
| [第 1 章](docs/01-foundations.md) | 从一次文本生成到推理服务 | LLM 推理为什么难？TTFT/TPOT 是什么？KV Cache 有多大？ |
| [第 2 章](docs/02-engine-architecture.md) | 推理引擎的通用架构 | 调度器、KV 管理器、执行器如何协作？ |
| [第 3 章 vLLM：以 PagedAttention 为核心的通用推理引擎](docs/03-vllm.md) | vLLM 核心机制 | PagedAttention、块级前缀缓存、V1 引擎如何工作？ |
| [第 4 章 SGLang：以 RadixAttention 与程序级优化为核心的推理栈](docs/04-sglang.md) | SGLang 核心机制 | RadixAttention、缓存感知调度（可选 LPM）、重叠调度如何工作？ |
| [第 5 章 核心机制正面对比：页表 vs 基数树，通用调度 vs 缓存感知](docs/05-mechanism-comparison.md) | 核心机制正面对比 | 页表 vs 基数树，通用调度 vs 缓存感知，差异何时真正重要？ |
| [第 6 章 功能、模型与生态对比](docs/06-features-ecosystem.md) | 功能、模型与生态 | 量化、LoRA、多模态、投机解码、PD 分离谁更全？ |
| [第 7 章 性能评测：如何科学地比较 vLLM 与 SGLang](docs/07-benchmarking.md) | 性能评测方法论 | 怎样公平比较？官方 benchmark 有哪些坑？ |
| [第 8 章 动手实践：启动、调用、压测与调参](docs/08-hands-on.md) | 动手实践 | 安装、起服务、调用、压测、调参、排错 |
| [第 9 章 源码地图：一次请求在两个引擎中的调用链](docs/09-source-code-tour.md) | 源码地图 | 一次 forward 的调用链经过哪些文件？ |
| [第 10 章 选型、演进与前沿](docs/10-selection-and-future.md) | 选型与前沿 | 给定业务如何选？两个引擎共同的演进方向 |
| [附录 A](docs/appendix-a-glossary.md) | 术语表 | 快速查阅 |
| [附录 B](docs/appendix-b-exercises.md) | 习题与实验 | 巩固与拓展 |
| [附录 C](docs/appendix-c-references.md) | 参考资料 | 论文、文档、博客、代码 |

建议学习路径：

- **两周速成**：第 0 → 1 → 2 → 3 → 4 → 5 章，配合第 8 章跑通一个服务（第 6、7 章选读）；
- **完整课程（约 32 学时）**：全部正文 + 附录 B 的实验；
- **工程选型**：第 0、1、5、6、7、8、10 章，重点做第 8 章的对照压测。

### 零基础怎么读

1. **第一遍不要抠公式**：先读第 0 章并完成自测；再读第 1 章 1.1～1.5，知道“prefill 吃算力、decode 吃带宽、KV 很占显存”即可。
2. **第二遍建立结构**：读第 2 章 2.0～2.4，跟着“一条请求的 12 步”走一遍。
3. **第三遍再进细节**：读第 3、4 章的 3.0/4.0 类比和“手把手例子”，再回头看公式与源码。
4. **每章只记三件事**：一个核心问题、一个关键数据结构、一个代价。

---

## 5. 版本约定与免责声明

推理引擎迭代极快。本讲义以 **vLLM 0.34.0** 与 **SGLang 0.5.23** 为讲解基线，事实核对日期为 **2026-11-25**。

> **讲义基准（事实核对日期：2026-11-25）**
>
> | 项目 | 基线版本 | 官方核对来源 | 说明 |
> |---|---|---|---|
> | vLLM | 0.34.0（2026-11-16 发布） | [GitHub Release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)、[PyPI](https://pypi.org/project/vllm/) | V1 引擎自 0.8 起为默认；`block_size` 自 0.34.0 起默认 64 |
> | SGLang | 0.5.23（2026-10-30 发布） | [GitHub Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)、[PyPI](https://pypi.org/project/sglang/) | 0.4.6 起默认调度为 `fcfs`；0.5.x 的参数定义在 `sglang/srt/arg_groups/` |
> | 参数默认值 | 对应版本的 `--help` 与启动日志 | [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.34.0/configuration/engine_args.html)、[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) | 本文不承诺跨版本逐字兼容 |
>
> 教学示例中仍常用 `block_size=16`，便于与 PagedAttention 论文和历史资料对照；但 0.34.0 的实际默认值是 64，启动时以日志或 `--block-size` 为准。SGLang 的 `page_size` 官方默认值为 1，因此 Radix Cache 的共享粒度可以做到 token 级。

在这个基线下：

- **稳定的部分**：PagedAttention、RadixAttention、continuous batching、chunked prefill、CUDA Graph、张量并行等原理不会轻易改变；
- **易变的部分**：具体命令行参数、默认值、支持的模型与量化格式、内部类名和文件路径；
- **实践原则**：运行前以你安装版本的官方文档和 `--help` 为准；本讲义中的命令用于建立直觉，不保证跨版本逐字兼容。

另外，任何“A 比 B 快 N 倍”的说法都高度依赖模型、硬件、流量形态、版本与调参。第 7 章会专门讲如何做公平评测。**本讲义的目标是教你判断方法，而不是让你背结论。**

---

## 6. 建议的实验环境

| 项目 | 最低建议 | 说明 |
|---|---|---|
| GPU | 1 × 24 GB（如 RTX 4090 / L4） | 可跑 7B/8B 的 FP16 或 4-bit 量化 |
| 内存 | 32 GB 以上 | 模型下载与 CPU 侧缓存 |
| 系统 | Linux 或 Windows + WSL2 | 两个引擎的官方支持均以 Linux 为主 |
| CUDA | 12.1 以上 | 以引擎版本要求为准 |
| Python | 3.10+ | 建议使用独立虚拟环境 |
| 磁盘 | 100 GB 空闲 | 模型权重与编译缓存 |

如果只有 CPU 或很小显存，仍然可以完成第 0、1、2、5 章的推导与第 7 章的方法论学习；第 8 章可改用云端 GPU 或 Colab 类环境。

---

## 7. 本讲义的使用方式

每章采用统一结构：

1. **本章导读**：先修知识、一句话概括、给零基础同学的阅读建议；
2. **问题引入**：为什么需要这个机制，先用生活类比建立直觉；
3. **原理拆解**：数据结构、算法、公式、分步算例与图示；
4. **两个引擎的实现**：vLLM 怎么做、SGLang 怎么做；
5. **对比与取舍**：什么时候差异重要，代价是什么；
6. **常见误区 + 小结 + 思考题**：自检是否真正理解。

图表按章编号（如“图 3-1”“表 5-2”），正文用“（图 3-1）”引用；示意图用 `text` 代码块绘制，以便在终端与网页中保持对齐。

建议边读边做三件事：

- 在纸上画出调度器每一步的状态变化；
- 用第 1 章的公式估算你手头模型的显存账本；
- 每读完一章，回到第 5 章的对比表补充一行自己的理解。

---

## 8. MkDocs 站点

本讲义已经配置为 MkDocs + Material 站点，章节之间带有“继续阅读”跳转链接，导航栏按四个部分组织。

在线站点：<https://shizimiao.github.io/sglang-vs-vllm-lecture/>

```bash
# 安装文档依赖（建议在独立虚拟环境中）
pip install -r requirements-docs.txt

# 本地预览：http://127.0.0.1:8000
mkdocs serve

# 构建静态站点到 site/
mkdocs build
```

站点首页为 `docs/index.md`，配置见 `mkdocs.yml`。

> **关于 Material 的 MkDocs 2.0 警告**：`mkdocs serve` 启动时可能看到 Material 团队关于 MkDocs 2.0 不兼容的提示。这是前瞻性警告，不是构建错误，不影响当前站点。`requirements-docs.txt` 把大版本限制在 MkDocs 1.6.x + Material 9.x（本项目验证于 Material 9.8.x）；如果想让终端安静一些，可以这样启动：
>
> ```powershell
> # PowerShell
> $env:NO_MKDOCS_2_WARNING=1; mkdocs serve
> ```
>
> ```bash
> # Bash
> NO_MKDOCS_2_WARNING=1 mkdocs serve
> ```

### 局域网访问

`mkdocs serve` 默认只监听 `127.0.0.1`，局域网内其他设备访问不到。需要显式绑定到 `0.0.0.0`：

```powershell
# 先按 Ctrl+C 停掉当前服务，再执行
mkdocs serve -a 0.0.0.0:8000
```

然后在 Windows 防火墙中放行 TCP 8000（管理员 PowerShell）：

```powershell
New-NetFirewallRule -DisplayName "MkDocs 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

查看本机局域网 IP：

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '127.*' } |
  Select-Object InterfaceAlias, IPAddress
```

其他设备在同一局域网内访问：

```text
http://<本机局域网IP>:8000
```

例如 `http://192.168.1.23:8000`。若无法访问，依次检查：是否同一网段、Windows 网络配置文件是否为“专用”、防火墙规则是否生效、路由器是否开启了 AP 隔离。

> MkDocs 的 `serve` 是开发服务器，只适合在可信局域网内共享，不要直接暴露到公网。需要长期对外提供服务时，应执行 `mkdocs build`，把 `site/` 交给 Nginx、Caddy 或静态托管平台。

如果不想用开发服务器，也可以构建后用 Python 静态服务器共享：

```powershell
mkdocs build
python -m http.server 8000 --directory site --bind 0.0.0.0
```

---

## 9. 目录速览

```text
sglangVSvllm/
├── README.md                     ← 你在这里
├── mkdocs.yml                    ← MkDocs 配置
├── requirements-docs.txt         ← 文档构建依赖
├── docs/
│   ├── index.md                  ← 站点首页
│   ├── 00-prerequisites.md
│   ├── 01-foundations.md
│   ├── 02-engine-architecture.md
│   ├── 03-vllm.md
│   ├── 04-sglang.md
│   ├── 05-mechanism-comparison.md
│   ├── 06-features-ecosystem.md
│   ├── 07-benchmarking.md
│   ├── 08-hands-on.md
│   ├── 09-source-code-tour.md
│   ├── 10-selection-and-future.md
│   ├── appendix-a-glossary.md
│   ├── appendix-b-exercises.md
│   ├── appendix-c-references.md
│   ├── javascripts/mathjax.js
│   └── stylesheets/extra.css
├── site/                         ← mkdocs build 生成
└── code/
    ├── bench_prefix.py           ← 共享前缀对照实验
    ├── requirements.txt
    └── README.md
```

---

## 10. 许可证

- 讲义正文、图表与习题：**CC BY 4.0**，转载或改编请署名并注明来源；
- `code/` 目录下的示例脚本：**MIT**；
- 完整条款见 [LICENSE](LICENSE)。

---

> **一句话预告**：vLLM 把 KV Cache 当成“操作系统的虚拟内存页”来管理，SGLang 把它当成“可复用的前缀树”来管理；一个强调通用与工程成熟，一个强调前缀复用与程序级优化。理解这句话背后的每一个词，就是本讲义的全部内容。
