# 第 10 周：收尾 —— 排障手册 + 引擎子系统深读 ★★★

## 目标

把前 9 周的机制沉淀成两样**能长期用**的东西，并补全那道贯穿全程的问题：

1. 《LLM 推理排障手册》：症状 → 可能原因 → 验证手段 → 优化方向。见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。
2. 挑一个引擎子系统**逐行读通**，写成源码笔记。选 SGLang `RadixCache` 前缀缓存，见 [radix_cache_deepread.md](radix_cache_deepread.md)（行号全部核对真实源码）。
3. 回看第 6-7 周对照表，补全 **"引擎为什么手写而不交给编译器"** 的答案（下文）。

## 文件

| 文件 | 内容 |
| --- | --- |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 排障手册，6 大症状分类，每条标注对应学习周 |
| [radix_cache_deepread.md](radix_cache_deepread.md) | SGLang RadixCache 逐行深读（RadixKey/TreeNode/match/split/insert/evict/lock_ref） |

## 怎么用

- 排障手册：遇到线上问题，先按 §0 分清 prefill/decode，再去对应症状表逐条排除。**每次只改一个变量**。
- 深读笔记：配合真实源码 `~/github/sglang/python/sglang/srt/mem_cache/radix_cache.py` 对照着读，笔记里每个流程都给了行号。

---

## 补全：引擎为什么手写 kernel/调度，而不交给编译器？

> 承接第 5 周（Triton 手写 kernel）、第 6-7 周（引擎自研 KV 分页 + 调度），也是第 9 周 W4A16 "未融合就变慢" 那个教训的总纲。

先厘清：**不是二选一，而是分工**。编译器（`torch.compile`/Inductor、XLA）负责"把一段静态计算图自动优化 + 融合 + 生成 kernel"；引擎手写的是**编译器覆盖不到、或覆盖得不够好的那部分**。具体四条：

### 1. 编译器优化的是"计算"，引擎要优化的很多是"内存/调度/生命周期"

`torch.compile` 能融合逐元素算子、选好 matmul、消一部分 launch，但它**看不到也管不了**：
- KV cache 怎么**分页、复用前缀、跨请求共享**（第 6 周 PagedAttention / RadixCache）——这是数据结构与内存生命周期问题，不是计算图问题。
- 请求怎么**动态组 batch、chunked prefill、抢占**（第 7 周调度器）——这是运行时策略，编译期根本不知道有哪些请求。

这两块恰是 LLM 推理吞吐的大头，编译器天然够不着。

### 2. LLM 推理是"动态形状 + 长期驻留状态"，和编译器偏爱的静态图相反

- 编译器在**静态 shape、无副作用**的计算图上最强。但推理里 batch 大小、seq 长度每步都变，KV cache 是**跨很多次前向持续增长的可变状态**。
- 引擎的做法是自己管理这份状态（分页表、前缀树、req_to_token 映射），只把"单步的、形状归档后的"计算丢给底层 kernel/CUDA Graph。**CUDA Graph 之所以能用（第 5 周），正是引擎先把 shape 归档成静态档**，替编译期解决了动态性。

### 3. 融合边界和数值技巧，编译器给不到极致

- FlashAttention（第 4 周）不只是融合，是**改算法**：tiling + online softmax，从不物化 seq² 注意力矩阵。这类"重写数学"的优化，通用编译器不会替你做。
- W4A16 的**融合 dequant kernel**（第 8/9 周）：边解包 int4 边做 matmul，权重从不 materialize 成 fp16。第 9 周实测里 mini 实现"先解回整块 fp16"反而更慢更费显存——**收益全在这个手写融合 kernel 里**。编译器面对自定义量化布局，通常给不出这种专用 kernel。

### 4. 编译器"够用"的地方，引擎也确实交给它

反过来说：模型内部那些**逐元素/规约/norm/激活**的融合，引擎乐于用 `torch.compile`/Inductor 或 Triton 生成——因为那正是编译器的主场。所以现代引擎（vLLM/SGLang）是**混合体**：手写调度 + 分页 + attention/量化 kernel，其余计算尽量交给编译器和厂商库（cuBLAS/cuDNN）。

### 一句话总结

> **编译器擅长优化"一段静态计算"，引擎必须亲自管"动态的请求调度与长期驻留的 KV 状态"，并在 attention/量化这些需要改算法或专用内存布局的地方手写融合 kernel。** 二者是分工：能交给编译器的计算就交出去，交不出去的内存/调度/专用 kernel 才自研——这正是第 5~9 周每一课的共同主线。

---

## 十周全景回顾

| 周 | 主题 | 一句话带走 |
| --- | --- | --- |
| 1 | 基线与指标 | 分开测 TTFT/TPOT，冷热分离，一切优化先有基线 |
| 2 | profiler + 访存/算力受限 | decode 访存/launch 受限，prefill 算力受限 |
| 3 | 手写 MiniGPT + KV cache | KV 显存公式；decode 逐 token 复用 KV |
| 4 | attention 后端 | FlashAttention 改算法（tiling+online softmax），别用 MATH |
| 5 | Triton + CUDA Graph | 手写融合 kernel；CUDA Graph 消 launch（三约束） |
| 6 | 引擎 KV 管理 | PagedAttention 分页；RadixCache/block-hash 前缀复用 |
| 7 | 引擎调度 | continuous batching、chunked prefill、抢占（读 vLLM 源码） |
| 8 | 量化 | 权重/激活/KV 各省什么；decode 场景 W4A16+KV 量化收益最大 |
| 9 | 端到端项目 | 一次可复现优化 + 报告；量化收益依赖融合 kernel |
| 10 | 收尾 | 排障手册 + RadixCache 深读 + "引擎为何手写"总纲 |

## 本周验收（能答出来就算通关）

1. 遇到"TPOT 高"，你的排查顺序是什么？（先分 prefill/decode → profiler 归因 → 对应症状表）
2. RadixCache 靠什么保证"正在用的前缀不被驱逐"？（`lock_ref` + `protected_size_` + `evictable_leaves`）
3. 前缀缓存和调度器怎么衔接？（`cache_unfinished_req` 里的 dec/inc_lock_ref 锁交接）
4. "引擎为什么不全交给 `torch.compile`"——你能用上面四条自己讲一遍吗？
5. 如果让你给一个新场景做推理优化，你的第一步、归因方法、验收口径分别是什么？（回到第 1、9 周的方法论）

## 承上启下（结课）

十周从"会测"到"会读引擎源码、会排障、会讲清取舍"。接下来的自然方向：
- 挑 vLLM/SGLang 的**另一个子系统**（如 attention backend 选择、投机解码 EAGLE、PD 分离）用同样方法深读。
- 把第 9 周的 MiniGPT 换成真实模型（TinyLlama）+ torchao 融合量化，跑一遍真实收益。
- 参与一个 issue / 读一个 PR，把"读懂"变成"改得动"。
