# LLM 推理引擎源码学习计划（PyTorch 底座 → vLLM / SGLang）

> 适用代码库：`/Users/minim4/github/pytorch`、`sglang/`、（可选）`vllm/`
> 学习方向：LLM 推理引擎优化，CUDA 为主
> 建议周期：10 周，每周 8-10 小时
> 定位：这是 [PyTorch 推理优化源码学习计划](../PyTorch推理优化源码学习计划.md) 的 **LLM 重排版**——保留通用底座，砍掉 `torch.compile` 特有内容，把重心压到 KV cache、prefill/decode、CUDA Graph、attention kernel、调度与量化。

## 〇、两条视角，一句话记住

- **编译器视角**（PyTorch `torch.compile`）：把 Python 模型自动变成 kernel。通用、自动、但对 LLM 负载不够特化。
- **引擎视角**（vLLM / SGLang）：手写模型 + 手动管 KV cache + 手写/复用高性能 kernel。为确定性、内存布局控制和 LLM 负载特化而生。

**本计划走引擎视角，但先打通 PyTorch 底座**——因为引擎跑在 PyTorch 之上，显存、dispatcher、CUDA Graph、Triton 都是共享地基。

> 配套笔记：[[project_sglang_vs_vllm_kvcache]]（KV cache / RadixCache vs BlockHash / 调度差异）。全程对照阅读。

## 一、学习目标

完成本计划后，应能够：

1. 正确 benchmark LLM 推理，分离 prefill / decode、冷启动 / 稳态，报告 TTFT、TPOT、吞吐、显存。
2. 讲清一次 `generate()` 的完整数据流：tokenize → prefill → decode 循环 → KV cache 读写 → 采样。
3. 理解 KV cache 的内存布局与管理：PagedAttention 分页、RadixCache 前缀复用、显存碎片。
4. 读懂 attention kernel（FlashAttention / PagedAttention）的输入输出与优化点，能读 Triton 实现。
5. 分析 decode 阶段的 kernel launch 开销，理解 CUDA Graph capture/replay 与固定 batch 分档。
6. 理解 continuous batching / 调度器如何在动态请求下保持 GPU 打满。
7. 分析并应用低精度与量化（W8A8 / W4A16 / KV cache 量化）到真实推理负载。
8. 独立定位一个 LLM 推理瓶颈（算力 / 访存 / launch / 调度 / 显存）并给出优化方案。

## 二、整体学习路径

```text
推理 benchmark 方法学（TTFT/TPOT）
  -> PyTorch eager 运行时（dispatcher / CUDACachingAllocator）           [底座]
  -> 最小 LLM 推理：gpt_fast 读懂 prefill/decode/KV cache               [心智模型]
  -> attention 机制：SDPA / FlashAttention / PagedAttention             [核心算子]
  -> CUDA Graph capture/replay + Triton kernel 读写                     [核心优化]
  -> 切入 SGLang/vLLM：KV cache 管理（Paged / Radix）                   [引擎主线]
  -> continuous batching 与调度器                                       [引擎主线]
  -> 低精度与量化（W8A8 / W4A16 / KV cache 量化）                        [部署刚需]
  -> 端到端综合项目 + 排障手册                                          [产出]
```

全程带一个真实模型（建议一个小 GPT，如 Llama-3.2-1B / TinyLlama），所有结论用 profiler、日志、生成代码验证。

## 三、10 周计划

### 第 1 周：可信的 LLM 推理 benchmark

目标：建立正确的 LLM benchmark 方法，避免把编译、warmup、同步误算为稳态。

阅读 / 工具：
- `torch.profiler`、`torch.utils.benchmark` 基本用法。
- `benchmarks/gpt_fast/benchmark.py`（PyTorch 侧的 LLM benchmark 参照）。
- 了解指标定义：TTFT（首 token 延迟）、TPOT / ITL（token 间延迟）、吞吐（tokens/s）、并发。

实践：
1. 对一个小模型分别测 prefill-only、decode-only、端到端。
2. 分离冷启动、warmup、稳态。
3. 记录 TTFT、TPOT、吞吐、峰值显存、GPU 利用率。

验收：能解释异步 CUDA 计时为何要同步，为什么 prefill 和 decode 必须分开报指标。

### 第 2 周：PyTorch eager 运行时（底座）

目标：建立 Tensor 从 Python → ATen → dispatcher → CUDA kernel 的心智模型，理解显存分配行为。

阅读：
- `c10/core/DispatchKey.h`、`c10/core/DispatchKeySet.h`（先理解职责）。
- `c10/cuda/CUDACachingAllocator.cpp`、`c10/cuda/CUDAStream.cpp`（理解 caching allocator 为什么存在、如何影响显存曲线）。
- `c10/core/InferenceMode.h`。

实践：
1. profiler 分析一个 eager 前向，把时间分成 Python / dispatcher / kernel / 同步。
2. 观察 CUDACachingAllocator 的显存复用行为（`torch.cuda.memory_summary()`）。

验收：看到显存问题时能判断是碎片、缓存未释放还是真实占用；看到慢时能先分层归因。

### 第 3 周：最小 LLM 推理——读懂 gpt_fast

目标：用一个"够简单能读完"的实现建立 prefill / decode / KV cache 心智模型。

阅读：
- `benchmarks/gpt_fast/model.py`（模型定义、KV cache 结构、attention）。
- `benchmarks/gpt_fast/generate.py`（prefill + decode 循环、采样）。

重点概念：
- prefill 一次处理整段 prompt vs decode 每次一个 token；
- KV cache 如何随 decode 增长、如何被复用；
- 为什么 decode 是访存 + launch 受限、小 GEMM 效率低。

实践：手动 print KV cache 形状随步数变化；分别计时 prefill 和单步 decode。

验收：能画出一次 `generate()` 的数据流图，并解释 prefill 与 decode 的负载差异根因。

### 第 4 周：attention 机制与 kernel

目标：理解 attention 的不同实现路径与优化点，为读引擎 attention 打基础。

阅读：
- `test/dynamo/test_sdpa.py`、SDPA 后端选择逻辑（flash / mem-efficient / math）。
- `torch/_inductor/kernel/flex_attention.py`（了解 flex_attention 思路）。
- FlashAttention 论文核心思想（online softmax、tiling、不落地中间矩阵）——概念即可。

实践：对比 naive attention、SDPA 各后端的速度与显存；观察 seq_len 增长时的差异。

验收：能解释 FlashAttention 为什么省显存、为什么快；能说出 PagedAttention 相比 FlashAttention 多解决了什么（KV cache 非连续存储）。

### 第 5 周：CUDA Graph + Triton（核心优化）

目标：掌握 decode 阶段消 kernel launch 开销的核心手段，以及读写 Triton 的能力。

阅读：
- `torch/_inductor/cudagraph_trees.py`（重点：static address、input mutation、动态 shape 三个限制条件）。
- `torch/_inductor/codegen/triton.py`（Triton codegen，建立读 Triton 的能力）。
- `torch/_inductor/runtime/README.md`。

实践：
1. 对小模型开 `torch.compile(mode="reduce-overhead")`，用 `TORCH_LOGS="output_code,kernel_code"` 看生成的 Triton 与 CUDA Graph 使用。
2. 手写一个最小 Triton kernel（如 fused RMSNorm 或 element-wise），跑通并 benchmark。

验收：能解释 CUDA Graph 的三个限制条件为什么逼着引擎做"固定 batch 分档 + padding + 预分配 KV cache 地址"；能独立读懂一个 Triton kernel。

### 第 6 周：切入引擎——SGLang/vLLM 的 KV cache 管理

目标：从 PyTorch 底座跨到真实引擎，理解 KV cache 分页与前缀复用。

阅读（SGLang 为主，vLLM 对照）：
- SGLang 的 KV cache / memory pool 相关模块（token→block 映射、分配释放）。
- SGLang RadixCache（前缀树复用）。
- vLLM PagedAttention / BlockManager（对照理解分页思想）。

重点概念：
- 分页为什么解决显存碎片；
- RadixCache（前缀树）vs BlockHash 的复用粒度差异（对照 [[project_sglang_vs_vllm_kvcache]]）；
- KV cache 地址如何配合 CUDA Graph 的 static address 要求。

实践：跑多轮对话/共享前缀请求，观察前缀缓存命中对 TTFT 的影响。

验收：能讲清一条请求的 KV cache 从分配到复用到释放的全过程，并对比 SGLang 与 vLLM 的策略取舍。

### 第 7 周：continuous batching 与调度器

目标：理解引擎如何在动态到达的请求下保持 GPU 打满。

阅读（SGLang 为主）：
- SGLang scheduler / batch 组织逻辑（如何把 prefill 和 decode 请求拼进同一 batch）。
- 请求生命周期：waiting → running → finished 的状态流转。
- chunked prefill / prefill-decode 混合调度（如支持）。

重点概念：
- continuous batching 相比静态 batching 的吞吐优势；
- prefill 与 decode 抢占、优先级、显存压力下的驱逐；
- 调度粒度如何影响 TTFT 与 TPOT 的权衡。

实践：变化并发数与 prompt 长度分布，观察吞吐、TTFT、TPOT 曲线与调度行为。

验收：能解释一个高并发场景下引擎的调度决策，并指出瓶颈是显存、调度还是 kernel 效率。

### 第 8 周：低精度与量化

目标：把量化应用到真实 LLM 推理，理解精度/速度/显存权衡。

阅读 / 工具：
- torchao 的 eager quantization 与 PT2E quantization（PyTorch 新量化主线，旧 `torch.ao.quantization` 正迁移，不要投入）。
- 引擎侧的量化路径：W8A8、W4A16（GPTQ/AWQ 权重量化）、KV cache 量化（FP8/INT8）。
- 复用你已有的 `quantization/` 目录积累。

重点概念：
- 权重量化 vs 激活量化 vs KV cache 量化各自省什么（显存 / 带宽 / 算力）；
- decode 访存受限场景下，KV cache 量化为什么收益大；
- 量化对数值精度的影响与验证方法。

实践：对同一模型跑 FP16 基线 + 至少一种量化，对比 TTFT/TPOT/吞吐/显存 + 输出误差。

验收：能为一个部署场景选择合适的量化方案并用数据论证。

### 第 9 周：端到端综合项目

目标：完成一次可复现的 LLM 推理优化，串起前 8 周。

综合项目（选一个小 GPT，真实负载）至少包含：
1. eager FP16/BF16 基线；
2. prefill / decode 分离的 TTFT / TPOT / 吞吐 / 显存报告；
3. KV cache 行为分析（增长、复用、碎片）；
4. attention 后端对比（SDPA 各后端或引擎 attention）；
5. CUDA Graph 开关对比；
6. 至少一个量化实验；
7. 一个高并发调度实验（continuous batching）；
8. 正确性与数值误差验证；
9. 基于 profiler / 日志 / 生成代码的瓶颈解释。

产出是一份短报告，说清"瓶颈在哪、为什么这样优化、副作用是什么"，而不只是速度数字。

### 第 10 周：沉淀排障手册 + 引擎源码深读

目标：把方法固化成可复用资产，并挑一个引擎子系统深读。

1. 整理一份《LLM 推理排障手册》：症状 → 可能原因 → 验证手段 → 优化方向。
2. 从 SGLang/vLLM 里选一个子系统（KV cache / 调度 / 一个 attention kernel）逐行读通，写一篇源码笔记。
3. 回看第 6-7 周的对照表，补全"引擎为什么手写而不交给编译器"的答案。

验收：手册能指导下一次真实调优；源码笔记能讲清一个引擎子系统的设计取舍。

## 四、固定实验模板

每个实验用同一张记录表：

| 项目 | 内容 |
| --- | --- |
| 模型与输入 | 模型、并发、prompt 长度分布、生成长度、dtype |
| 环境 | GPU、CUDA、PyTorch/引擎 commit、关键配置 |
| 正确性 | 与基线输出的误差或一致性 |
| 延迟 | TTFT（P50/P99）、TPOT / ITL |
| 吞吐 | tokens/s、requests/s |
| 资源 | 峰值显存、GPU 利用率、KV cache 占用 |
| 内核 | 主要 kernel、attention 实现、CUDA Graph 是否命中 |
| 调度 | batch 组织方式、前缀缓存命中率、驱逐次数 |
| 结论 | 瓶颈、优化理由和副作用 |

每次只改一个主要变量。

## 五、性能诊断速查

```bash
# PyTorch 侧：图捕获 / graph break（学 gpt_fast 时用）
TORCH_LOGS="graph_breaks,recompiles" python generate.py

# PyTorch 侧：生成代码与 kernel（第 5 周）
TORCH_LOGS="output_code,kernel_code,perf_hints" python your_infer.py

# 完整编译调试产物
TORCH_COMPILE_DEBUG=1 python your_infer.py
```

- `nsight` / `nsys` 抓 timeline，重点看 decode 阶段的 kernel launch 间隙与 CUDA Graph 是否命中。
- 引擎自带的 metrics / logging（SGLang/vLLM 都有）看 TTFT、TPOT、缓存命中、显存。

## 六、建议跳过 / 弱化的内容

- `torch.compile` 的 Dynamo graph break / guard / recompile 调试细节（引擎不走）。
- torch.export / AOTInductor 部署（引擎有自己的加载编译流程）。
- backward / autograd / 优化器 / 分布式训练通信（DDP/FSDP）。
- 传统 TorchScript。

但 **不要跳过 eager 运行时（dispatcher / allocator）和 attention/KV cache/CUDA Graph 的通用原理**——这些是引擎的地基。

## 七、编译器视角 vs 引擎视角对照

| 问题 | `torch.compile` | vLLM / SGLang（本计划目标） |
| --- | --- | --- |
| kernel launch 开销 | 自动 fusion + `reduce-overhead` CUDA Graph | 手动 CUDA Graph + 固定 batch 分档 |
| 动态 seq_len | 动态 shape / 重编译 | padding 分档 + PagedAttention |
| attention | flex_attention / SDPA 后端选择 | 手写 PagedAttention / FlashAttention |
| KV cache | 靠 functionalization 处理 mutation | 手动分页管理（RadixCache / BlockManager） |
| 算子实现 | Inductor 生成 Triton | 手写 Triton / CUDA 算子 |
| 批处理 | 静态或需重编译 | continuous batching 动态拼批 |

**读引擎源码时反复问**：这里手写了什么、为什么不交给编译器？答案通常是——引擎要确定性、要极致控制 KV cache 内存布局、要针对 LLM 负载特化。理解了"为什么手写"，就从"会用 PyTorch"跨到了"懂推理引擎"。

## 八、阅读原则

1. 先跑一个最小例子，再读它实际经过的源码。
2. 先看入口、数据结构、调用关系，不要一开始逐行读大文件（对照 [[feedback_code_reading_strategy]]）。
3. 所有性能判断都用 profiler、日志或生成代码验证。
4. 优先优化真实瓶颈，不以"开了更多开关"为完成标准。
5. 每周保留实验记录，最终形成自己的 LLM 推理排障手册。
