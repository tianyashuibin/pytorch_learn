# 第 4 周：attention 机制与 kernel

## 目标

理解 attention 的不同实现路径与优化点，为第 6 周读引擎 attention 打底。两条正交的优化线要分清：

- **FlashAttention**：解决"attention 计算怎么省显存/带宽"——不落地 `[seq,seq]` 中间矩阵。
- **PagedAttention**：解决"KV cache 在显存里怎么摆放"——分块 + block_table，按需分配、可共享前缀。

## 对应阅读

- `test/dynamo/test_sdpa.py` — SDPA 用法与后端行为。
- `torch/nn/attention/__init__.py` — `sdpa_kernel` 上下文管理器、`SDPBackend` 枚举。
- `torch/_inductor/kernel/flex_attention.py` — flex_attention 思路（了解）。
- FlashAttention 论文核心：online softmax + tiling + 不落地中间矩阵（概念即可）。

## 文件

- `_common.py` — 计时 / 显存工具。
- `naive_vs_flash.py` — naive attention（显式 `[seq,seq]`）vs SDPA(flash)，测速度和峰值显存随 seq_len 的变化。看 naive 的 O(seq²) 显存暴涨。
- `sdpa_backends.py` — 强制切换 flash / efficient / cudnn / math 后端对比；演示什么条件下 flash 用不了掉回 math。
- `paged_attention_demo.py` — 最小 block_table 寻址 demo：逻辑连续的序列物理上散在乱序 block，仍能正确 attention；对比连续预分配 vs 分页的显存利用。

## 运行

```bash
pip install torch
cd week4

python naive_vs_flash.py       # GPU 上差异最明显
python sdpa_backends.py
python paged_attention_demo.py
```

CPU/MPS 上 flash 后端可能不可用（看不到显存差异属正常），重点理解机制；有 CUDA 才能测出 naive 的显存暴涨和 flash 的加速。

## 本周验收（能答出来才算过）

1. naive attention 的显存为什么随 seq² 增长？那个大矩阵是什么？
2. FlashAttention 为什么又省显存又快？（关键词：tiling、online softmax、不落地 `[seq,seq]`、少 HBM 读写）
3. 什么条件下 SDPA 会从 flash 掉回 math 后端？引擎为此会做什么（对齐 head_dim/dtype）？
4. PagedAttention 相比 FlashAttention 多解决了什么问题？block_table 是干什么的？它怎么带来 prefix sharing？

## 承上启下

- 第 3 周你把 KV cache 预分配成一整块连续 `[max_seq_len, ...]`——本周 `paged_attention_demo.py` 正是指出它的浪费，并给出分页解法。
- FlashAttention（计算）+ PagedAttention（存储）这两条线，第 6 周会在 SGLang/vLLM 源码里合流。
- flash 对固定 shape/dtype 的偏好，也呼应第 5 周 CUDA Graph 对固定形态的要求。
