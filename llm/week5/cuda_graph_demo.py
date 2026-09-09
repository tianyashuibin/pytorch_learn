"""第 5 周：CUDA Graph capture/replay —— decode 消 launch overhead 的核心机制。

回顾：第 1 周发现 decode 慢，第 2 周定位到"小 kernel + 框架/launch 开销"。
decode 每步要发射几十上百个小 kernel，每个 kernel 从 CPU 发射到 GPU 都有固定开销。
当 kernel 很小时，这些 launch 开销可能比 GPU 实际算的时间还长——GPU 在等 CPU 喂活。

CUDA Graph 的解法：
  - 把一串 kernel 的发射序列"录制(capture)"成一张图；
  - 之后用一次 replay 重放整张图，几乎不需要 CPU 逐个发射；
  => launch overhead 从"每 kernel 一次"降到"每次 replay 一次"。

代价 = 三个硬约束（本周验收核心，也是引擎设计的由来）：
  1. static address（固定地址）：输入/输出张量地址在 capture 和 replay 之间必须不变。
     -> 引擎因此预分配 KV cache 并固定地址（第 3 周我们已经这么做了）。
  2. no input mutation from outside（输入靠原地拷贝更新）：新输入要 copy_ 进那块固定张量，
     不能换一个新张量。
  3. static shape（形状固定）：图录的是特定形状；形状变了要重录。
     -> 引擎因此做 batch 分档 + padding，把动态形状归一到少数几个固定档位。

本脚本在一个"decode-like"的小 workload 上对比 eager 循环 vs CUDA Graph replay。
需要 CUDA；非 CUDA 环境会打印说明并跳过。

运行：python cuda_graph_demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_decode_block(dim=1024, device="cuda", dtype=torch.float16):
    """一个 decode 单步会经历的小算子链：几层 Linear + norm + 激活，seq=1。"""
    layers = nn.Sequential(
        nn.Linear(dim, dim, bias=False),
        nn.LayerNorm(dim),
        nn.GELU(),
        nn.Linear(dim, dim, bias=False),
        nn.LayerNorm(dim),
        nn.GELU(),
        nn.Linear(dim, dim, bias=False),
    ).to(device=device, dtype=dtype).eval()
    return layers


def time_cuda(fn, iters=100, warmup=20):
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


def main():
    if not torch.cuda.is_available():
        print("[skip] CUDA Graph 需要 NVIDIA GPU。当前无 CUDA。")
        print("      逻辑仍可阅读：核心是 capture 一次、replay 多次，避免逐 kernel 发射。")
        print("      三约束：static address / 原地更新输入 / static shape。")
        return

    device, dtype = "cuda", torch.float16
    dim, batch = 1024, 1  # decode: batch=1, seq=1 —— 小 kernel，launch 开销占比高
    model = build_decode_block(dim, device, dtype)

    # ---- 基线：eager，每步都从 CPU 逐个发射 kernel ----
    x = torch.randn(batch, dim, device=device, dtype=dtype)

    @torch.inference_mode()
    def eager_step():
        return model(x)

    eager_ms = time_cuda(eager_step)

    # ---- CUDA Graph：capture 一次，replay 多次 ----
    # 约束 1：用固定的 static_input / static_output 张量（地址不变）
    static_input = torch.randn(batch, dim, device=device, dtype=dtype)

    # 官方要求：capture 前在 side stream 上预热几次，让 cuBLAS 等初始化好
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.inference_mode():
            for _ in range(3):
                model(static_input)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g):
            static_output = model(static_input)  # 录制这整条 kernel 序列

    def graph_step():
        # 约束 2：新输入用 copy_ 原地写进固定张量，不能换新张量
        static_input.copy_(x)
        g.replay()          # 一次重放整张图，几乎无逐 kernel 发射开销
        return static_output

    graph_ms = time_cuda(graph_step)

    print(f"[env] device=cuda dtype={dtype} dim={dim} batch={batch} (decode-like)\n")
    print(f"eager 每步:       {eager_ms:.4f} ms")
    print(f"CUDA Graph 每步:  {graph_ms:.4f} ms")
    print(f"加速:             {eager_ms / graph_ms:.2f}x  "
          f"(省下的主要是 kernel launch / Python 开销)\n")

    print("=== 三个约束在代码里的体现 ===")
    print("  1. static address：static_input/static_output 全程复用同一块显存。")
    print("  2. 原地更新输入：graph_step 里用 static_input.copy_(x)，不是新建张量。")
    print("  3. static shape：这张图只对 (batch=%d, dim=%d) 有效；换形状要重新 capture。" % (batch, dim))
    print("\n=> 引擎正是为满足这三条，才做：预分配固定地址 KV cache + batch 分档 + padding。")


if __name__ == "__main__":
    main()
