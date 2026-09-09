# 第 1 周：可信的 LLM 推理 benchmark

## 目标

建立正确的 LLM benchmark 方法，能分离 **prefill / decode**、**冷启动 / 稳态**，并用 LLM 口径的指标报告性能。

## 为什么 LLM benchmark 和普通模型不一样

普通模型报一个"前向延迟"就够了。LLM 推理是**两段完全不同的负载**，必须分开：

| 阶段 | 做什么 | 指标 | 瓶颈 |
| --- | --- | --- | --- |
| prefill | 一次处理整段 prompt | **TTFT**（首 token 延迟） | 计算受限，随 prompt_len 增长 |
| decode | 每次生成 1 个 token | **TPOT / ITL**（token 间延迟） | 访存 + kernel launch 受限，随 KV cache 缓慢上升 |

把它们混成一个平均延迟，会同时掩盖两边的真实问题。

## 文件

- `timing_utils.py` — CUDA event 计时 + 同步 + 分位数统计。**核心：GPU 计时前后必须 `torch.cuda.synchronize()`**，否则测的是 kernel 发射时间而非执行时间。
- `model.py` — 加载小 GPT（默认 TinyLlama-1.1B），自动选 device / dtype。
- `benchmark_baseline.py` — 主程序。手写解码循环，自管 KV cache，分别计时 prefill 和每一步 decode；区分冷启动 / 预热 / 稳态。
- `scaling_experiment.py` — 用数据验证两条规律：TTFT 随 prompt_len 上升、TPOT 随 KV cache 增长漂移。

## 运行

```bash
pip install torch transformers
cd week1

# 主 benchmark
python benchmark_baseline.py --prompt-len 128 --gen-len 64 --steady 5

# 规律验证实验
python scaling_experiment.py
```

无 GPU 时会自动退到 CPU / MPS 跑通流程（CUDA 显存/计时指标显示为 nan，属正常）。

## 本周验收（能答出来才算过）

1. 为什么异步 CUDA 计时前后要 `synchronize()`？不加会测到什么？
2. 为什么第一次调用（冷启动）不能算进稳态平均？它多包含了什么开销？
3. 为什么 prefill 和 decode 必须分开报指标？各自的瓶颈是什么？
4. 跑 `scaling_experiment.py`：TTFT 是否随 prompt_len 上升？TPOT 是否随 decode 步数漂移？用你的数据解释原因。

## 记录

把每次结果的 `[json]` 段贴进实验记录表（见主计划第四章模板），后续几周会不断和这个基线对比。

> 注意：这里为了看清 prefill/decode **故意不用 `model.generate()`**。真实引擎（SGLang/vLLM）也是自己管解码循环和 KV cache——本周的手写循环就是理解引擎的第一步。
