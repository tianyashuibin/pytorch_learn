"""第 5 周：手写一个 Triton kernel（fused RMSNorm），建立读写 Triton 的能力。

为什么练这个：vLLM/SGLang 大量算子是 Triton 手写的（fused RMSNorm、RoPE、
paged attention 等）。第 6 周读引擎源码时，你得能看懂 Triton。这里写一个最小但完整的。

RMSNorm 融合的意义：
  朴素做法 = square -> mean -> rsqrt -> mul -> mul(weight)，多个 kernel，
  每步都把中间结果写回 HBM 再读回来（带宽浪费）。
  融合成一个 Triton kernel：一行数据只读一次、只写一次，中间量留在寄存器/SRAM。
  RMSNorm 是典型 bandwidth-bound 算子，融合收益主要来自省 HBM 往返。

Triton 心智模型：
  - kernel 按 program(=block) 并行，每个 program 处理一部分数据（这里：一行）。
  - tl.load / tl.store 显式搬数据；mask 处理边界；中间计算在寄存器里。

需要 CUDA + triton（pip install triton）。非 CUDA 环境跳过，只保留可读代码。

运行：python triton_rmsnorm.py
"""

from __future__ import annotations

import torch


def torch_rmsnorm(x, weight, eps=1e-5):
    # 参考实现（会拆成多个 kernel）
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def main():
    if not torch.cuda.is_available():
        print("[skip] Triton kernel 需要 CUDA GPU。当前无 CUDA。")
        print("      可阅读下方 kernel 源码，理解 tl.load/tl.store、mask、寄存器内计算的写法。")
        _print_kernel_source()
        return
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print("[skip] 未安装 triton：pip install triton")
        return

    @triton.jit
    def rmsnorm_kernel(
        x_ptr, w_ptr, out_ptr,
        stride_row,           # 一行有多少元素（=N）
        N: tl.constexpr,      # 特征维，编译期常量
        eps: tl.constexpr,
        BLOCK: tl.constexpr,  # 每个 program 处理的列数（>=N，一次装下一整行）
    ):
        row = tl.program_id(0)                     # 每个 program 负责一行
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * stride_row + cols

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # 融合：mean(x^2) -> rsqrt -> 归一 -> 乘 weight，全在寄存器里完成
        mean_sq = tl.sum(x * x, axis=0) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * rrms * w
        tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)

    def triton_rmsnorm(x, weight, eps=1e-5):
        assert x.is_contiguous()
        M, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        rmsnorm_kernel[(M,)](x, weight, out, x.stride(0), N=N, eps=eps, BLOCK=BLOCK)
        return out

    # ---- 正确性 + 性能对比 ----
    device, dtype = "cuda", torch.float16
    M, N = 4096, 4096
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)

    ref = torch_rmsnorm(x, w)
    out = triton_rmsnorm(x, w)
    max_err = (ref.float() - out.float()).abs().max().item()
    print(f"正确性：与 torch 参考的最大误差 = {max_err:.4e}  "
          f"({'✓ 通过' if max_err < 1e-2 else '✗ 偏大，检查实现'})\n")

    def bench(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    torch_ms = bench(lambda: torch_rmsnorm(x, w))
    triton_ms = bench(lambda: triton_rmsnorm(x, w))
    # 有效带宽：读 x + 写 out ≈ 2 * M*N*2字节
    gb = 2 * M * N * 2 / 1e9
    print(f"torch RMSNorm:  {torch_ms:.4f} ms  ({gb / (torch_ms / 1e3):.0f} GB/s)")
    print(f"triton 融合:    {triton_ms:.4f} ms  ({gb / (triton_ms / 1e3):.0f} GB/s)")
    print(f"加速:           {torch_ms / triton_ms:.2f}x")
    print("\n要点：RMSNorm 是 bandwidth-bound，融合的收益来自减少 HBM 读写往返，")
    print("      看有效带宽是否接近显卡峰值来判断优化是否到位。")


def _print_kernel_source():
    print("""
    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_row, N, eps, BLOCK):
        row  = tl.program_id(0)          # 每个 program 处理一行
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row*stride_row + cols, mask=mask).to(tl.float32)
        rrms = 1.0 / tl.sqrt(tl.sum(x*x, 0)/N + eps)   # 中间量在寄存器
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        tl.store(out_ptr + row*stride_row + cols, (x*rrms*w), mask=mask)
    """)


if __name__ == "__main__":
    main()
