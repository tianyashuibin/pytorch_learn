# 第 8 周：低精度与量化 ★★

## 目标

把量化落到真实 LLM 推理，搞清三类量化**各省什么**、**为什么 decode 场景 KV 量化收益最大**、以及**怎么验证精度损失**。这是部署刚需。

## 三类量化，一句话各省什么

| 量化对象 | 典型方案 | 主要省 | 打在什么瓶颈上 |
| --- | --- | --- | --- |
| 权重 (Weight) | W4A16 (GPTQ/AWQ)、W8A8 | 显存 + 权重搬运带宽 | **decode 访存受限**（瘦高 matmul，瓶颈是搬权重） |
| 激活 (Activation) | W8A8 (SmoothQuant) | 算力（int8 tensor core） | **prefill 算力受限**（大矩阵乘） |
| KV cache | INT8 / FP8 KV | 显存 + 读 KV 带宽 | **decode 访存受限**（每步读全量 KV） |

关键结论：**decode 是访存/launch 受限**（第 1、2 周实测），所以省"搬运"的量化（W4A16、KV 量化）对 decode 收益最大；激活量化省算力，主要帮 compute-bound 的 prefill。

## 先跑三个 mini 实现建立骨架

```bash
cd week8
python quant_basics.py        # 量化的三个旋钮：对称/非对称、粒度、位宽
python weight_only_w4a16.py   # W4A16：int4 打包权重，4x 压缩 + decode 为何受益
python kv_cache_quant.py       # KV cache int8：显存 + 带宽双省，误差极小
```

跑出来的关键数据（在本机 randn 权重上，趋势为准）：

- `quant_basics`：int8 per-tensor ~5% → per-channel ~0.8%（**粒度是精度主来源**）；非对称对偏斜分布再降一半；int4 必须配 group-wise。
- `weight_only_w4a16`：4096×4096 权重 **32MB → 8.5MB（近 4x）**；误差在 group-wise 精度内。
- `kv_cache_quant`：KV int8 per-token，attention 输出误差仅 **~0.8%**，显存 **1.88x**（含 scale）。

> 注意：mini 实现用随机权重、且没有 GPTQ/AWQ 的误差补偿，所以误差数字偏大（真实 W4A16 常能到 ~1%）。这里看的是**旋钮之间的相对趋势**，不是绝对精度。

---

## 引擎量化源码阅读地图

真实引擎的量化都走"量化方法(QuantizationConfig) + 每层量化方案(scheme) + 融合 dequant kernel"三段式。先看 config/scheme 的分派，再看某一种（如 AWQ）的 kernel。

### vLLM（`~/github/vllm-main/vllm/model_executor/layers/quantization/`）

- `base_config.py` — `QuantizationConfig` 抽象基类：每种量化方法实现它，`get_quant_method()` 决定某个 Linear 用哪个量化实现。**入口先看这个。**
- `awq_triton.py` — AWQ 的 Triton 反量化/矩阵乘 kernel（呼应第 5 周 Triton 阅读能力）。`auto_awq.py` / `auto_gptq.py` 是加载 HuggingFace 上 AWQ/GPTQ checkpoint 的 config。
- `fp8.py` / `fbgemm_fp8.py` — FP8 权重/激活量化。
- `compressed_tensors/` — 统一的压缩张量框架（W8A8、W4A16 等多种 scheme 收敛到这里），看 `schemes/` 子目录理解"一层怎么选量化方案"。
- `bitsandbytes.py` — bnb 的 NF4/INT8 路径。

### SGLang（`~/github/sglang/python/sglang/srt/layers/quantization/`）

- `base_config.py` / `base_scheme.py` — 同样的 config + scheme 两层抽象。
- `awq/`、`compressed_tensors/` — 与 vLLM 结构对应（很多 kernel 直接复用/移植）。
- `blockwise_int8.py` — 块级 int8。
- `fp4_kv_cache_quant_method.py` / `dequantization.py` — **KV cache 量化**方法（对照本周 `kv_cache_quant.py`）。

### torchao（PyTorch 官方量化主线）

计划里提到的 torchao 本机未安装（`pip install torchao`）。它是 PyTorch 新量化主线，提供 `quantize_(model, int4_weight_only())` 这类一行 API + 高效 kernel；旧的 `torch.ao.quantization` 正在迁移，**不要再投入旧 API**。要读就读 torchao 的 `quant_api.py` 和 `dtypes/`。

---

## 实践作业（计划要求：数据论证）

对同一个模型（如第 1 周的 TinyLlama）跑 **FP16 基线 + 至少一种量化**，用第 1 周的 `benchmark_baseline.py` 采集，对比：

| 指标 | 从哪来 |
| --- | --- |
| TTFT（prefill 延迟） | week1 计时 |
| TPOT / ITL（decode 每 token） | week1 计时 |
| 吞吐（tok/s） | week1 |
| 显存峰值 | week1 `peak_mem_mb` |
| 输出误差 | 对比 FP16 与量化版的 logits / 生成文本 |

最省事的量化入口：`pip install torchao` 后 `quantize_(model, int4_weight_only())`；或用 HF `bitsandbytes`（`load_in_4bit=True`）。

## 本周验收（能答出来才算过）

1. 权重量化 / 激活量化 / KV cache 量化各省什么（显存 / 带宽 / 算力）？各自打在 prefill 还是 decode 的瓶颈上？为什么？
2. 为什么 W4A16 里权重压到 4-bit 而激活保持 16-bit 就够？（结合 decode 访存受限）
3. per-tensor / per-channel / group-wise 对精度和开销的影响？int4 为什么几乎必须 group-wise？（跑 `quant_basics.py` 佐证）
4. KV cache 量化为什么在长上下文 / 高并发 decode 场景收益最大？（结合第 3 周 KV 显存公式 + 第 2 周访存受限结论）
5. 你的量化实验里，精度损失怎么量化的？速度/显存收益和精度损失如何权衡、如何为一个部署场景选型？

## 承上启下

- 第 2 周"decode 访存/launch 受限" + 第 3 周"KV cache 有多贵" → 本周量化正是打这两个瓶颈的手段。
- 第 5 周 Triton 阅读能力 → 直接用来读 vLLM `awq_triton.py` 的融合 dequant kernel。
- 第 6-7 周引擎的 KV 分页/调度 + 本周 KV 量化 → 第 9 周端到端项目里一起用上。
