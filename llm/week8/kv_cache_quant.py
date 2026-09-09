"""第 8 周（3/3）：KV cache 量化 —— decode 访存受限场景收益最大的那一个。

承接第 3 周 kvcache_anatomy.py：KV cache 显存 = 2*layers*batch*heads*seq*head_dim*dtype_bytes。
它随 seq 和 batch 线性增长，长上下文 / 高并发时往往比权重还大，而且 decode 每步都要把
**整个 KV cache 读一遍**算 attention —— 这是典型访存受限。

把 KV cache 从 fp16 压到 int8（甚至 fp8），直接带来两个收益：
  1. 显存减半 -> 能塞更长上下文 / 更大 batch（吞吐↑）。
  2. decode 每步读 KV 的带宽减半 -> TPOT↓。

代价：K/V 被量化引入误差，传导到 attention 输出。本文件量化 KV、跑 attention、测输出误差，
并算显存节省。常见做法：per-token 或 per-head 量化 K、V（粒度细，误差小）。

不依赖特定 GPU。运行：python kv_cache_quant.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quantize_per_token_int8(x: torch.Tensor):
    """对 (batch, heads, seq, head_dim) 的 KV，按最后一维（head_dim）逐 token 对称 int8 量化。

    每个 (token, head) 一个 scale —— 粒度细，贴合每个 token 自己的数值范围。
    """
    qmax = 127
    scale = (x.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_per_token_int8(q, scale):
    return q.float() * scale


def attention(q, k, v):
    """标准缩放点积注意力（因果掩码），返回输出。"""
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    seq = scores.shape[-1]
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def demo():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 1, 8, 512, 64

    Q = torch.randn(batch, heads, seq, head_dim)
    K = torch.randn(batch, heads, seq, head_dim)
    V = torch.randn(batch, heads, seq, head_dim)

    ref = attention(Q, K, V)

    # 量化 K、V（Q 是当前 step 的 query，通常不量化）
    Kq, Ks = quantize_per_token_int8(K)
    Vq, Vs = quantize_per_token_int8(V)
    Khat = dequantize_per_token_int8(Kq, Ks)
    Vhat = dequantize_per_token_int8(Vq, Vs)
    got = attention(Q, Khat, Vhat)

    print(f"KV cache 量化：K/V fp16 -> int8（per-token）")
    print(f"  attention 输出相对误差：{(ref-got).norm()/ref.norm():.4%}")

    # 显存对比（承接第 3 周公式）
    def kv_mb(seq_len, batch_, dtype_bytes):
        n_layers = 32   # 假设一个 7B 级模型
        return 2 * n_layers * batch_ * heads * seq_len * head_dim * dtype_bytes / 1024 / 1024

    print("\n=== KV cache 显存（32 层, 8 头, head_dim=64）===")
    print(f"{'场景':<22}{'fp16':>12}{'int8':>12}{'节省':>8}")
    for tag, s, b in [("seq=512  batch=1", 512, 1),
                      ("seq=4096 batch=1", 4096, 1),
                      ("seq=4096 batch=32", 4096, 32),
                      ("seq=32768 batch=8", 32768, 8)]:
        fp16 = kv_mb(s, b, 2)
        int8 = kv_mb(s, b, 1) + kv_mb(s, b, 4) / head_dim  # +scale(每token一个fp32)
        print(f"{tag:<22}{fp16:>10.0f}MB{int8:>10.0f}MB{fp16/int8:>7.2f}x")

    print("\n要点：")
    print("  - KV cache 随 seq×batch 线性涨，长上下文/高并发时是显存主角。")
    print("  - decode 每步读全量 KV -> 访存受限 -> KV 量化同时省显存 + 省带宽（TPOT↓）。")
    print("  - per-token/per-head 粒度让误差很小；fp8 常比 int8 精度更稳（动态范围大）。")
    print("  - 对照第 2 周结论：decode 是 launch/访存受限，KV 量化正打在带宽这个瓶颈上。")


if __name__ == "__main__":
    demo()
