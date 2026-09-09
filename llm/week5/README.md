# 第 5 周：CUDA Graph + Triton（核心优化）★★★

## 目标

掌握 decode 阶段消 **kernel launch 开销**的核心手段（CUDA Graph），以及读写 **Triton** 的能力。这是全计划对 vLLM/SGLang 迁移价值最高的一周。

## 为什么这周最值钱

- 第 1 周测到 decode 慢，第 2 周定位到"小 kernel + launch/框架开销"——**CUDA Graph 就是消这块开销的答案**，也是引擎 decode 阶段的标配。
- vLLM/SGLang 大量算子是 **Triton 手写**的。第 6 周读引擎源码前，必须能看懂 Triton。

## CUDA Graph 的三个约束（本周验收核心）

capture 一次、replay 多次，把 launch 开销从"每 kernel 一次"降到"每次 replay 一次"。代价是三条硬约束——**它们正是引擎设计的由来**：

| 约束 | 含义 | 逼出引擎的什么做法 |
| --- | --- | --- |
| static address | 输入/输出张量地址不变 | 预分配、固定地址的 KV cache（第 3 周已这么做） |
| 原地更新输入 | 新输入 `copy_` 进固定张量 | 请求数据写进固定 buffer，不换张量 |
| static shape | 图只对特定形状有效 | batch 分档 + padding，归一到少数固定档位 |

## 文件

- `cuda_graph_demo.py` — 手动 CUDA Graph capture/replay，在 decode-like 小 workload 上对比 eager vs graph；代码里逐条标出三个约束。
- `triton_rmsnorm.py` — 手写一个融合 RMSNorm 的 Triton kernel，验证正确性 + 测有效带宽 vs torch。非 CUDA 环境会打印 kernel 源码供阅读。
- `compile_reduce_overhead.py` — `torch.compile(mode="reduce-overhead")` 自动上 CUDA Graph，和手动版对照；教你用 `TORCH_LOGS` 看生成的 Triton。

## 运行

```bash
pip install torch triton      # triton 需要 CUDA
cd week5

python cuda_graph_demo.py
python triton_rmsnorm.py
python compile_reduce_overhead.py

# 看 Inductor 生成的 Triton 代码
TORCH_LOGS="output_code,perf_hints" python compile_reduce_overhead.py
TORCH_COMPILE_DEBUG=1 python compile_reduce_overhead.py   # 导出完整产物
```

**注意：本周三个脚本基本都需要 NVIDIA GPU**（CUDA Graph 和 Triton 都是 CUDA 特性）。无 CUDA 时脚本会优雅跳过并打印要点/源码，逻辑仍可阅读；但真正的加速数字要在 CUDA 机器上跑。

## 对应阅读

- `torch/_inductor/cudagraph_trees.py` — 自动 CUDA Graph 如何处理 static address / mutation / shape。
- `torch/_inductor/codegen/triton.py` — Inductor 生成的 Triton 长什么样。
- `torch/_inductor/runtime/README.md`。

## 本周验收（能答出来才算过）

1. CUDA Graph 为什么能加速 decode？它把什么开销从"每 kernel"降到了"每次 replay"？
2. 复述三个约束，并解释每一条如何逼出引擎的一个具体做法（KV cache 固定地址 / 原地更新 / batch 分档 padding）。
3. 你写的 Triton RMSNorm 融合，收益主要来自哪里？为什么看"有效带宽"而不是 FLOPS？
4. `reduce-overhead`（编译器自动）和引擎手动管 CUDA Graph，各自的取舍是什么？

## 承上启下

- 三个约束把第 3 周（固定地址 KV cache）、第 4 周（flash 偏好固定 shape）串起来。
- Triton 读写能力 → 第 6 周直接用来读 SGLang/vLLM 的手写算子。
- "编译器自动 vs 引擎手动"的对照 → 贯穿第 6-7 周，理解引擎"为什么不交给编译器"。
