"""第 2 周：观察 CUDACachingAllocator 的显存行为。

对应源码（理解职责，不逐行）：
  c10/cuda/CUDACachingAllocator.cpp  —— PyTorch 不直接向 driver 反复 cudaMalloc/cudaFree，
                                        而是自己缓存显存块复用，避免昂贵的分配和隐式同步。

要建立的直觉（对 LLM 尤其重要，KV cache 就是大块显存）：
  - allocated  = 当前张量真正占用的显存
  - reserved   = allocator 向 driver 要来、缓存着备用的显存（>= allocated）
  - reserved - allocated 里藏着"碎片"和"缓存待复用块"
  - free 掉张量，reserved 通常不还给 driver（留着复用）——这解释了"显存不降"的常见困惑

运行：python allocator_behavior.py
"""

from __future__ import annotations

import torch

from _common import pick_device, pick_dtype


def mb(x: int) -> float:
    return x / (1024 ** 2)


def snapshot(tag: str, device: torch.device) -> None:
    if device.type != "cuda":
        print(f"[{tag}] (非 CUDA 设备，跳过显存统计)")
        return
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"[{tag:<28}] allocated={mb(alloc):8.1f}MB  "
          f"reserved={mb(reserved):8.1f}MB  "
          f"缓存/碎片={mb(reserved - alloc):7.1f}MB")


def demo_caching_reuse(device: torch.device, dtype: torch.dtype) -> None:
    """演示：释放后 reserved 不还给 driver，下次分配直接命中缓存。"""
    print("=== 演示 1：释放显存后 reserved 不下降（缓存复用）===")
    snapshot("初始", device)

    big = torch.empty(256, 1024, 1024, device=device, dtype=dtype)  # 一大块
    snapshot("分配大块后", device)

    del big
    # 注意：不调用 empty_cache，模拟真实运行中的行为
    snapshot("del 之后(未 empty_cache)", device)
    print("  -> allocated 掉了，reserved 没掉：显存被 allocator 缓存着等复用。\n")

    again = torch.empty(256, 1024, 1024, device=device, dtype=dtype)
    snapshot("再次分配同样大小", device)
    print("  -> reserved 基本不变：这次分配命中了缓存，没向 driver 要新显存（快且无隐式同步）。")
    del again

    torch.cuda.empty_cache()
    snapshot("empty_cache 之后", device)
    print("  -> 手动 empty_cache 才把缓存还给 driver。生产中一般不频繁调用（会打断复用）。\n")


def demo_fragmentation(device: torch.device, dtype: torch.dtype) -> None:
    """演示：交替分配不同大小，reserved - allocated 之间的碎片。"""
    print("=== 演示 2：碎片（reserved 明显大于 allocated）===")
    torch.cuda.empty_cache() if device.type == "cuda" else None
    keep = []
    for i in range(8):
        # 大小不一，制造分配模式的差异
        n = (i % 4 + 1) * 32
        keep.append(torch.empty(n, 1024, 1024, device=device, dtype=dtype))
    snapshot("交替分配后", device)
    # 释放一半，制造空洞
    del keep[::2]
    snapshot("释放一半后", device)
    print("  -> reserved 和 allocated 之间的差就是碎片 + 缓存。KV cache 场景下，")
    print("     碎片会直接吃掉本可用于更多请求的显存——这正是引擎要做分页(PagedAttention)的动机。\n")


def demo_summary(device: torch.device) -> None:
    if device.type != "cuda":
        return
    print("=== memory_summary（allocator 的详细账本，重点看 Reserved / Allocated 段）===")
    print(torch.cuda.memory_summary(device, abbreviated=True))


def main() -> None:
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype} torch={torch.__version__}\n")

    if device.type != "cuda":
        print("当前非 CUDA 设备，显存演示需要 GPU。逻辑仍可阅读。\n")

    demo_caching_reuse(device, dtype)
    demo_fragmentation(device, dtype)
    demo_summary(device)


if __name__ == "__main__":
    main()
