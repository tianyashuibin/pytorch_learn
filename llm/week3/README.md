# 第 3 周：最小 LLM 推理——读懂 prefill / decode / KV cache

## 目标

用一个"够简单能读完"的实现建立 **prefill / decode / KV cache** 心智模型。这是全计划的心智地基——后面 attention、CUDA Graph、引擎 KV cache 管理、调度全都建立在它之上。

## 本周做法：先手写，再读 gpt_fast

直接读 `benchmarks/gpt_fast/model.py` 也行，但**自己写一遍带显式 KV cache 的 mini GPT**，能 print 出每一步 cache 的形状，理解会扎实得多。写完再回去读 gpt_fast，你会发现结构一一对应（KVCache / Attention / Transformer / decode 循环）。

## 一次 generate 的数据流（要能默画）

```text
tokens = tokenize(prompt)
prefill:  model(tokens[0:P], start_pos=0, causal=True)   # 一次处理整段，填满 cache 前 P 格
          -> 取最后位置 logits -> 采样 -> 第 1 个生成 token
decode 循环（每步）:
          model(token, start_pos=当前长度)                # 只处理 1 个 token，cache 追加 1 格
          -> logits -> 采样 -> 下一个 token
```

| | prefill | decode |
| --- | --- | --- |
| 每次前向的 seq | prompt_len（大） | 1（小） |
| 矩阵乘 | 大 GEMM，吃满算力 | 小 GEMM，算力用不满 |
| KV cache 动作 | 一次写入一大段 | 每步追加一格、读全部历史 |
| 瓶颈 | 计算受限 | 访存 + kernel launch 受限 |

## 文件

- `mini_gpt.py` — 从零手写的 mini GPT：RMSNorm + RoPE + 带 **KVCache** 的 MHA + SwiGLU。KV cache 是"预分配 max_seq_len + 按位置写入"的静态形态（引擎和 CUDA Graph 需要的形态）。随机权重即可。
- `generate.py` — 手写 prefill + decode 循环；`--verbose` 打印数据流；分别计时 prefill 和单步 decode。
- `kvcache_anatomy.py` — 把 KV cache 掰开：显存公式与增长、decode 中 cache 的变化、为什么预分配 max_seq_len。

## 运行

```bash
pip install torch          # 本周不需要 transformers，模型是手写的
cd week3

python generate.py --prompt-len 64 --gen-len 32 --verbose
python kvcache_anatomy.py
```

CPU 也能跑通（模型很小），有 GPU 则 prefill/decode 的时间差异更明显。

## 本周验收（能答出来才算过）

1. 默画一次 `generate()` 的数据流：prefill 和 decode 各自的输入 seq 长度、KV cache 动作是什么？
2. 为什么 prefill 摊到每 token 比 decode 快？（跑 `generate.py` 用数据支撑）
3. KV cache 显存的计算公式是什么？随 seq_len、batch、layers 怎么变？（跑 `kvcache_anatomy.py`）
4. 为什么把 KV cache 预分配成固定 max_seq_len、地址不变？这对第 5 周的 CUDA Graph 有什么意义？它的代价是什么、引擎怎么缓解？

## 承上启下

- 第 1 周你测到"decode 比 prefill 慢在每步小"，第 2 周落到"框架开销 + 小 kernel"，**本周看清了 decode 的第三个负担：读越来越长的 KV cache**。
- 预分配、固定地址的 KV cache → 第 5 周 CUDA Graph 能 capture/replay 的前提。
- KV cache 的显存浪费与碎片 → 第 6 周 PagedAttention / RadixCache 要解决的问题（对照 [[project_sglang_vs_vllm_kvcache]]）。
