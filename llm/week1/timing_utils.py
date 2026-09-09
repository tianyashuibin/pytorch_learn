"""第 1 周：计时与统计工具。

核心原则（本周验收点）：
1. CUDA 是异步的——Python 里 kernel 一发射就返回，真正算完要靠同步点。
   所以任何 GPU 计时前后都必须 torch.cuda.synchronize()，否则测到的是"发射时间"。
2. 用 CUDA event 计时比 time.perf_counter() 更准，因为它记录的是 GPU 时间轴上的时刻。
   但 event 之间仍需保证被测区间的 kernel 都已排队。
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


@contextmanager
def cuda_timer(device: torch.device):
    """用 CUDA event 计时一段 GPU 代码，返回毫秒。

    用法：
        with cuda_timer(device) as t:
            model(x)
        print(t.ms)
    """
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        class _Result:
            ms = float("nan")

        r = _Result()
        start.record()
        yield r
        end.record()
        # 关键：等 GPU 真正跑完这段区间的所有 kernel，event 时间才有意义。
        torch.cuda.synchronize()
        r.ms = start.elapsed_time(end)
    else:
        class _Result:
            ms = float("nan")

        r = _Result()
        t0 = time.perf_counter()
        yield r
        r.ms = (time.perf_counter() - t0) * 1000.0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """把一组毫秒样本汇总成 P50/P90/P99/mean。"""
    if not samples_ms:
        return {}
    s = sorted(samples_ms)

    def pct(p: float) -> float:
        # 最近秩法，样本少时够用。
        idx = min(len(s) - 1, max(0, round(p / 100.0 * (len(s) - 1))))
        return s[idx]

    return {
        "count": len(s),
        "mean": statistics.fmean(s),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "min": s[0],
        "max": s[-1],
    }


def peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return float("nan")


def reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
