"""第 4 周：naive attention vs SDPA(FlashAttention)，看清"为什么 flash 省显存"。

naive attention 的问题：
    scores = Q @ K^T        # 显式生成 [B, heads, seq, seq] 的大矩阵！
    attn   = softmax(scores)
    out    = attn @ V
  那个 [seq, seq] 中间矩阵是 O(seq^2) 显存。seq=4096、32 头时它就是几个 GB，
  而且要写回显存再读回来，带宽全耗在这。

FlashAttention 的核心思想（SDPA 的 flash / mem-efficient 后端）：
  - 分块(tiling)遍历 K/V，用 online softmax 增量累加，**从不把完整 [seq,seq] 落地到显存**。
  - 显存从 O(seq^2) 降到 O(seq)，且中间结果留在 SRAM/寄存器，省掉大量 HBM 读写。
  => 又省显存又快，尤其长序列。

本脚本用真实内存/时间数据把这个差异测出来。

运行：python naive_vs_flash.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _common import peak_mem_mb, pick_device, pick_dtype, reset_peak, sync, timed


def naive_attention(q, k, v):
    # 显式生成 [B, heads, seq, seq] scores —— 这就是 O(seq^2) 显存的来源
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def sdpa_attention(q, k, v):
    # 让 PyTorch 自动选后端（GPU 上通常是 flash / mem-efficient）
    return F.scaled_dot_product_attention(q, k, v)


def main():
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype}\n")

    B, heads, hd = 1, 16, 64
    print(f"{'seq_len':>8} | {'naive ms':>10} {'naive MB':>10} | {'sdpa ms':>10} {'sdpa MB':>10} | 加速")
    print("-" * 72)
    for seq in [256, 512, 1024, 2048, 4096]:
        q = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)
        k = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)
        v = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)

        # naive：可能在长序列 OOM，捕获一下
        try:
            reset_peak(device)
            naive_ms = timed(lambda: naive_attention(q, k, v), device, iters=10)
            naive_mb = peak_mem_mb(device)
        except RuntimeError as e:  # OOM
            naive_ms, naive_mb = float("nan"), float("nan")
            print(f"{seq:>8} | naive OOM: {str(e)[:40]}")
            continue

        reset_peak(device)
        sdpa_ms = timed(lambda: sdpa_attention(q, k, v), device, iters=10)
        sdpa_mb = peak_mem_mb(device)

        speedup = naive_ms / sdpa_ms if sdpa_ms > 0 else float("nan")
        print(f"{seq:>8} | {naive_ms:>10.3f} {naive_mb:>10.1f} | "
              f"{sdpa_ms:>10.3f} {sdpa_mb:>10.1f} | {speedup:>4.1f}x")

    print("\n要点：")
    print("  - naive 的 [seq,seq] scores 让显存随 seq^2 暴涨（4096 时几个 GB）。")
    print("  - SDPA(flash) 显存随 seq 线性，且更快——长序列差距越大。")
    print("  - 这就是所有 LLM 引擎都用 FlashAttention 变体的原因。")
    print("  （CPU/MPS 上 flash 后端可能不可用，看不出显存差异属正常，重点理解机制。）")


if __name__ == "__main__":
    main()
