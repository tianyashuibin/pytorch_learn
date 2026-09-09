"""第 4 周：PagedAttention 的思想（机制 demo，为第 6 周引擎做铺垫）。

FlashAttention 解决"attention 计算怎么省显存/带宽"。
PagedAttention 解决另一个正交问题："KV cache 在显存里怎么摆放"。

第 3 周我们把每条序列的 KV cache 预分配成一整块连续 [max_seq_len, ...]。问题：
  - 每条请求都按最长预留 -> 大量显存浪费（大多数序列没那么长）。
  - 不同请求长度不一 -> 连续大块之间产生碎片（回顾第 2 周 allocator）。

PagedAttention 的思路（借鉴操作系统虚拟内存分页）：
  - 把 KV cache 切成固定大小的小 block（比如每 block 存 16 个 token 的 K/V）。
  - 物理上这些 block 散落在一个大池子里，不要求连续。
  - 每条序列维护一张 block_table：逻辑位置 -> 物理 block 号。
  - attention 时按 block_table 把需要的 block "gather" 起来再算。
  好处：按需分配、几乎零碎片、还能让多条序列共享相同前缀的 block（prefix sharing）。

本 demo 用最小代码演示 block_table 寻址：逻辑连续的序列，物理上放在乱序 block 里，
仍能正确取回 K/V 做 attention。这就是 vLLM PagedAttention / SGLang 内存池的核心机制。

运行：python paged_attention_demo.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _common import pick_device, pick_dtype


def main():
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype}\n")

    heads, hd = 4, 32
    block_size = 4         # 每个 block 存 4 个 token 的 K/V（真实引擎常用 16）
    num_blocks = 16        # 物理池子里一共 16 个 block
    seq_len = 10           # 这条序列实际有 10 个 token

    # ---- 物理 KV 池：所有 block 放一起，形状 [num_blocks, block_size, heads, hd] ----
    k_pool = torch.randn(num_blocks, block_size, heads, hd, device=device, dtype=dtype)
    v_pool = torch.randn(num_blocks, block_size, heads, hd, device=device, dtype=dtype)

    # ---- block_table：这条序列的逻辑 block 顺序 -> 物理 block 号（故意乱序，模拟按需分配）----
    n_logical_blocks = (seq_len + block_size - 1) // block_size  # ceil(10/4)=3
    block_table = torch.tensor([7, 2, 11][:n_logical_blocks], device=device)
    print(f"序列长度={seq_len}, block_size={block_size} -> 需要 {n_logical_blocks} 个逻辑 block")
    print(f"block_table（逻辑->物理）: {block_table.tolist()}")
    print("  逻辑 token 0-3  -> 物理 block 7")
    print("  逻辑 token 4-7  -> 物理 block 2")
    print("  逻辑 token 8-9  -> 物理 block 11（只用前 2 格）\n")

    # ---- 按 block_table 把散落的 block gather 成逻辑连续的 K/V ----
    k_gathered = k_pool[block_table].reshape(-1, heads, hd)[:seq_len]  # [seq_len, heads, hd]
    v_gathered = v_pool[block_table].reshape(-1, heads, hd)[:seq_len]
    print(f"gather 后逻辑连续 K 形状: {tuple(k_gathered.shape)}（物理上来自乱序 block）")

    # ---- 用 gather 出来的 K/V 做一次 decode attention（1 个 query token 对全部历史）----
    q = torch.randn(1, heads, hd, device=device, dtype=dtype)
    # 整理成 SDPA 需要的 [B, heads, seq, hd]
    qh = q.transpose(0, 1).unsqueeze(0)                          # [1, heads, 1, hd]
    kh = k_gathered.permute(1, 0, 2).unsqueeze(0)                # [1, heads, seq, hd]
    vh = v_gathered.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(qh, kh, vh)
    print(f"attention 输出形状: {tuple(out.shape)}  ✓ 计算正确，物理不连续不影响结果\n")

    # ---- 对比：连续存储 vs 分页存储的显存利用 ----
    print("=== 连续预分配 vs 分页 的显存对比（假设 max_seq_len=2048）===")
    max_seq = 2048
    contiguous_slots = max_seq                                   # 每条序列预留满
    paged_slots = n_logical_blocks * block_size                 # 只分配用到的 block
    print(f"  连续预分配: 每条序列占 {contiguous_slots} 格（实际只用 {seq_len} 格，"
          f"浪费 {contiguous_slots - seq_len} 格）")
    print(f"  分页:       只分配 {paged_slots} 格（{n_logical_blocks} 个 block），"
          f"浪费 ≤ block_size-1 = {block_size - 1} 格")
    print(f"  -> 分页把'按最长预留'变成'按需增长'，显存利用率大幅提升。")

    print("\n=== 附加好处：prefix sharing ===")
    print("  两条请求若 prompt 前缀相同，可以让它们的 block_table 前几项指向同一批物理 block，")
    print("  KV cache 直接复用、无需重算 —— 这就是 SGLang RadixCache / vLLM prefix caching 的基础。")
    print("  （对照 [[project_sglang_vs_vllm_kvcache]]，第 6 周细看两者复用粒度差异。）")


if __name__ == "__main__":
    main()
