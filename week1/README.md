# 第 1 周：建立可信的推理性能基线

对应学习计划「第 1 周」。目标：掌握正确的推理 benchmark 方法，避免把编译、warmup 或同步误算为稳态推理时间。

## 文件

| 文件 | 作用 |
|---|---|
| `timing_utils.py` | 计时/统计工具：CUDA 同步计时、P50/P90/P99、峰值显存。本周核心。 |
| `model.py` | 实验模型：`TinyMLP`（最小验证）与 `TinyTransformer`（贴近 LLM 负载）。 |
| `benchmark_baseline.py` | 主脚本：对 5 种配置测冷启动 + 稳态延迟 + 吞吐 + 显存。 |
| `profiler_demo.py` | `torch.utils.benchmark` 与 `torch.profiler` 用法示例。 |

## 运行

```bash
python benchmark_baseline.py                     # 默认 transformer / fp32
python benchmark_baseline.py --model mlp
python benchmark_baseline.py --dtype float16 --iters 100
python profiler_demo.py
```

macOS 无 CUDA：脚本自动降级到 MPS 或 CPU；`synchronize` 与显存统计会相应变为 no-op / N/A。部分 `torch.compile` 模式（如 max-autotune、reduce-overhead 的 CUDA Graph）在 CPU/MPS 上可能不生效或报错，脚本会跳过并打印原因。

## 本周验收（要能口头讲清）

1. **为什么异步 CUDA 计时前后要 `synchronize`？**
   kernel launch 是异步的，Python 侧计时只测到"提交到流"的时间。不同步会把 GPU 尚未算完的耗时漏掉，得到虚低的延迟。见 `timing_utils.timed`。

2. **为什么第一次调用不能纳入稳态平均？**
   首次调用包含 `torch.compile` 编译、autotune、cudnn benchmark、allocator 首次分配等一次性开销，量级远大于稳态。必须单独作为"冷启动"记录，稳态只统计 warmup 之后的多次运行。见 `benchmark_baseline.bench_one`。

## 配合日志重跑（计划第五节速查）

```bash
TORCH_LOGS="graph_breaks,graph_code" python benchmark_baseline.py
TORCH_LOGS="recompiles,dynamic"       python benchmark_baseline.py
TORCH_LOGS="output_code,kernel_code"  python benchmark_baseline.py
```
