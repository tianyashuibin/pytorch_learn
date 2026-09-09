"""第 4 周：SDPA 的多个后端，以及 PyTorch 如何选。

对应阅读：
  test/dynamo/test_sdpa.py
  torch/nn/attention/__init__.py（sdpa_kernel 上下文管理器、SDPBackend 枚举）

SDPA 有多个后端实现同一个数学：
  - FLASH_ATTENTION     : FlashAttention-2，最快、省显存，但有 dtype/head_dim/对齐等约束
  - EFFICIENT_ATTENTION : memory-efficient，约束更松的省显存实现
  - MATH                : 纯 PyTorch 数学实现（就是 naive），总能跑，最慢，作 fallback
  - CUDNN_ATTENTION     : cuDNN 后端（视版本/硬件）

PyTorch 会按输入(dtype、head_dim、是否 causal、是否有 mask 等)自动挑一个能用且最快的。
理解"什么条件下 flash 用不了、掉回 math"是本周关键——引擎为了稳定命中 flash，
会刻意把 shape/dtype 对齐到 flash 的约束上。

运行：python sdpa_backends.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _common import pick_device, pick_dtype, timed


def get_backends():
    """兼容不同 torch 版本的 SDPBackend / sdpa_kernel 位置。"""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return SDPBackend, sdpa_kernel
    except Exception:
        return None, None


def main():
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype} torch={torch.__version__}\n")

    SDPBackend, sdpa_kernel = get_backends()
    if SDPBackend is None:
        print("当前 torch 版本没有 torch.nn.attention.sdpa_kernel，跳过后端强制切换演示。")
        return

    B, heads, seq, hd = 1, 16, 1024, 64
    q = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)
    k = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)
    v = torch.randn(B, heads, seq, hd, device=device, dtype=dtype)

    candidates = [
        ("FLASH_ATTENTION", getattr(SDPBackend, "FLASH_ATTENTION", None)),
        ("EFFICIENT_ATTENTION", getattr(SDPBackend, "EFFICIENT_ATTENTION", None)),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("MATH", getattr(SDPBackend, "MATH", None)),
    ]

    print("=== 强制各后端，看谁能跑、谁最快（is_causal=True）===")
    for name, backend in candidates:
        if backend is None:
            continue
        try:
            with sdpa_kernel([backend]):
                ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                           device, iters=20)
            print(f"  {name:<22} {ms:>8.3f} ms")
        except RuntimeError as e:
            print(f"  {name:<22} 不可用: {str(e)[:60]}")

    print("\n=== 什么条件会让 flash 掉回 math ===")
    # 例子：非对齐 head_dim 或某些 mask 形态可能让 flash 用不了
    weird = torch.randn(B, heads, seq, 48, device=device, dtype=dtype)  # head_dim=48 非常见对齐
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(weird, weird, weird, is_causal=True)
        print("  head_dim=48：flash 可用")
    except RuntimeError as e:
        print(f"  head_dim=48：flash 拒绝 -> {str(e)[:70]}")
        print("  这类约束就是引擎要把 head_dim 对齐到 flash 友好值(如 64/128)的原因。")

    print("\n要点：不指定后端时 PyTorch 自动挑最优可用后端；")
    print("      引擎为了稳定吃到 flash 的速度，会主动满足它的 dtype/对齐约束。")


if __name__ == "__main__":
    main()
