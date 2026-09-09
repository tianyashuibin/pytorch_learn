# 第 2 周：理解 eager 推理运行时

对应学习计划「第 2 周」。目标：建立 Tensor 操作从 Python → ATen → dispatcher → CUDA kernel 的心智模型。

## 对应源码阅读

- `c10/core/InferenceMode.h` / `.cpp`
- `c10/core/DispatchKey.h` / `DispatchKeySet.h`
- `aten/src/ATen/` 中一个熟悉算子的声明与实现（如 `linear` / `add`）
- `c10/cuda/CUDACachingAllocator.cpp`、`c10/cuda/CUDAStream.cpp`（先理解职责，不逐行读）

## 文件

| 文件 | 作用 |
|---|---|
| `_common.py` | 把 week1 的 `model.py` / `timing_utils.py` 加入 import 路径，复用模型 |
| `dispatch_explore.py` | 观察 `DispatchKeySet`：requires_grad / no_grad / inference_mode 的差异、算子的后端注册、meta 设备 |
| `op_profile.py` | profiler 分析 eager 模型，把时间拆成 CPU 侧（框架+dispatcher+launch）与 GPU 侧（kernel），找 top-3 算子 |
| `compute_vs_bandwidth.py` | 用算术强度（AI = FLOPs/Bytes）判断算子偏计算受限还是带宽受限 |

## 运行

```bash
python dispatch_explore.py
python op_profile.py --model transformer
python compute_vs_bandwidth.py
```

## 本周验收

> 看到性能问题时，能先判断它属于**框架开销 / 算子实现 / 内存分配 / GPU kernel**。

判断路径：
1. `op_profile.py` 看 Self CPU 与 Self CUDA 的比值——比值大且 GPU 利用低 → 框架/launch bound（小算子太多）。
2. 单个算子 CUDA 时间高 → 算子实现/kernel bound，再用 `compute_vs_bandwidth.py` 判断是算力还是带宽。
3. profiler 里 `aten::empty` / allocator 相关条目突出 → 内存分配 bound。
4. `cudaStreamSynchronize` / `cudaDeviceSynchronize` 占比高 → 同步等待（可能是不必要的 `.item()` / `.cpu()`）。
