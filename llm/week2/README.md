# 第 2 周：PyTorch eager 运行时（底座）

## 目标

建立 Tensor 从 **Python → ATen → dispatcher → CUDA kernel** 的心智模型，并理解 **CUDACachingAllocator** 的显存行为。这是所有推理引擎（vLLM/SGLang 也跑在 PyTorch 上）的公共地基。

## 为什么这周对 LLM 重要（★★★）

- **dispatcher**：decode 阶段一步只算一个 token、kernel 很小，此时 Python + dispatch 的框架开销可能超过 GPU 计算本身——这就是引擎要上 CUDA Graph（第 5 周）的根本原因。先在这周把"框架开销"看见、量化出来。
- **allocator**：KV cache 是大块、反复分配释放的显存。理解 caching allocator 的 `allocated` vs `reserved`、碎片、为什么"显存不降"，直接通向第 6 周 PagedAttention 的设计动机。

## 对应源码（阅读，理解职责即可，不逐行）

- `c10/core/DispatchKey.h`、`c10/core/DispatchKeySet.h` — tensor 携带一组 key，dispatcher 按优先级选实现。
- `c10/core/InferenceMode.h` — inference_mode 如何砍掉 autograd 记账层。
- `c10/cuda/CUDACachingAllocator.cpp` — 为什么不直接 cudaMalloc/Free，而是自己缓存复用。
- `c10/cuda/CUDAStream.cpp` — stream 与异步执行（配合第 1 周的同步理解）。

## 文件

- `_common.py` — 设备/dtype 选择；可选加载真实小 GPT，失败自动退回纯 Tensor microbench。
- `dispatch_explore.py` — 打印不同上下文下的 DispatchKeySet；profiler 展开一次 matmul 的算子/内核；量化 inference_mode 对小算子的开销收益。
- `profile_layers.py` — 把一次前向的时间拆成 Python/dispatch/kernel/同步，并粗判 launch 受限 vs compute 受限。
- `allocator_behavior.py` — 观察 caching allocator：释放后 reserved 不降、缓存命中、碎片、`memory_summary`。

## 运行

```bash
pip install torch transformers   # transformers 可选，缺了会自动退回 microbench
cd week2

python dispatch_explore.py
python profile_layers.py          # 加 --trace 可导出 chrome trace
python allocator_behavior.py
```

## 本周验收（能答出来才算过）

1. 一次 `torch.matmul` 从 Python 到 GPU 经过哪些层？dispatcher 靠什么选中实现？
2. `allocated` 和 `reserved` 有什么区别？为什么 `del` 一个大张量后 reserved 常常不下降？
3. profiler 里 CPU 总时间 >> CUDA 总时间说明什么？这在 LLM 的哪个阶段最常见，怎么缓解？
4. 找出你 workload 里 self CUDA time 最高的 3 个算子，分别判断 compute 受限还是 bandwidth 受限，说明依据。

> 承上启下：第 1 周你已经看到 decode 比 prefill 慢在"每步很小"。本周用 profiler 把"小在哪"落到框架开销 + 小 kernel 上；第 5 周再用 CUDA Graph 把这部分开销消掉。
