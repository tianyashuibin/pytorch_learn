"""第 2 周：观察 dispatcher —— 一次算子调用到底经过了什么。

对应源码（阅读，不必逐行）：
  c10/core/DispatchKey.h        —— 每个 backend / 功能是一个 key（CPU, CUDA, Autograd, Autocast...）
  c10/core/DispatchKeySet.h     —— tensor 携带一组 key，dispatcher 按优先级选实现
  c10/core/InferenceMode.h      —— inference_mode 关掉 autograd 记账，减少 dispatch 层

心智模型：
    Python 调用 (torch.matmul)
      -> Python/C++ 边界
      -> dispatcher 按 DispatchKeySet 选中实现（可能先过 Autograd/Autocast 再到 CUDA）
      -> ATen kernel
      -> 发射 CUDA kernel（异步）

本脚本用 torch 暴露的 API 把这些"层"打印出来，让抽象变得可见。

运行：python dispatch_explore.py
"""

from __future__ import annotations

import torch

from _common import pick_device, pick_dtype, sync


def show_dispatch_keys(device: torch.device, dtype: torch.dtype) -> None:
    """打印不同上下文下 tensor 携带的 DispatchKeySet，看功能 key 如何增减。"""
    x = torch.randn(4, 4, device=device, dtype=dtype)
    print("=== DispatchKeySet 随上下文变化 ===")
    print(f"普通 tensor:            {torch._C._dispatch_key_set(x)}")

    req = torch.randn(4, 4, device=device, dtype=dtype, requires_grad=True)
    print(f"requires_grad=True:     {torch._C._dispatch_key_set(req)}  <- 多了 Autograd 相关 key")

    with torch.inference_mode():
        y = torch.randn(4, 4, device=device, dtype=dtype)
        print(f"inference_mode 内:      {torch._C._dispatch_key_set(y)}  <- 关掉了 autograd 记账")
    print("要点：key 越多，dispatch 经过的层越多。推理时用 inference_mode 就是砍掉不需要的层。\n")


def trace_one_op(device: torch.device, dtype: torch.dtype) -> None:
    """用 profiler 观察一次 matmul 展开成哪些 aten 算子 + CUDA kernel。"""
    from torch.profiler import ProfilerActivity, profile

    a = torch.randn(512, 512, device=device, dtype=dtype)
    b = torch.randn(512, 512, device=device, dtype=dtype)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # 预热，避免把首次开销算进来
    for _ in range(3):
        torch.matmul(a, b)
    sync(device)

    print("=== 一次 torch.matmul 的算子/内核分解 ===")
    with profile(activities=activities, record_shapes=True) as prof:
        torch.matmul(a, b)
        sync(device)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    print("要点：一行 Python 的 matmul，在 dispatcher 之下会落到 aten::matmul -> aten::mm，")
    print("      再发射具体的 GEMM CUDA kernel。这就是 Python->ATen->dispatch->kernel 的链路。\n")


def inference_mode_overhead(device: torch.device, dtype: torch.dtype) -> None:
    """量化 inference_mode 省下的 dispatch/记账开销（小算子上更明显）。"""
    import time

    x = torch.randn(64, 64, device=device, dtype=dtype)

    def loop(n=2000):
        for _ in range(n):
            _ = x + 1.0
            _ = _ * 2.0
        sync(device)

    # 预热
    loop(200)

    t0 = time.perf_counter()
    loop()
    normal = time.perf_counter() - t0

    with torch.inference_mode():
        t0 = time.perf_counter()
        loop()
        infer = time.perf_counter() - t0

    print("=== inference_mode 对小算子的框架开销影响 ===")
    print(f"普通模式:        {normal * 1000:.1f} ms")
    print(f"inference_mode:  {infer * 1000:.1f} ms")
    print("要点：小算子（elementwise）时间被 Python + dispatch 开销主导，")
    print("      去掉 autograd 记账能看到明显收益。大 kernel 上则相对可忽略。\n")


def main() -> None:
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype} torch={torch.__version__}\n")

    show_dispatch_keys(device, dtype)
    trace_one_op(device, dtype)
    inference_mode_overhead(device, dtype)


if __name__ == "__main__":
    main()
