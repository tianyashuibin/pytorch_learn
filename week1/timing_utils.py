"""
计时与统计工具 —— 第 1 周的核心。

要点(对应本周验收):
- 异步 CUDA:kernel launch 是异步的,Python 侧 time.perf_counter() 只测到"提交"时间。
  必须在计时开始前和结束前调用 torch.cuda.synchronize(),把 GPU 真正算完的时间算进来。
- 首次调用包含编译 / cudnn benchmark / allocator 预热,绝不能计入稳态平均。
- 用百分位(P50/P99)而不是单纯均值来描述延迟分布。

CPU / MPS 上没有异步 kernel 队列,synchronize 是 no-op,这里做统一封装。
"""
import statistics
import time
from contextlib import contextmanager

import torch


def get_device() -> torch.device:
    """自动选择设备:优先 CUDA,其次 Apple MPS,最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """把异步队列排空,保证计时覆盖真正的 kernel 执行。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU 是同步执行,无需处理


@contextmanager
def timed(device: torch.device):
    """
    计时上下文:进入/退出前都先 sync,确保测到的是稳态 GPU 时间。
    用法:
        with timed(device) as get_ms:
            model(x)
        elapsed_ms = get_ms()
    """
    sync(device)
    start = time.perf_counter()
    result = {}

    def get_ms():
        return result["ms"]

    try:
        yield get_ms
    finally:
        sync(device)
        result["ms"] = (time.perf_counter() - start) * 1000.0


def summarize(latencies_ms):
    """把一组稳态延迟(ms)汇总成 P50/P90/P99/mean 等指标。"""
    xs = sorted(latencies_ms)
    n = len(xs)

    def pct(p):
        if n == 0:
            return float("nan")
        # 最近秩法,足够本周使用
        idx = min(n - 1, max(0, round(p / 100.0 * (n - 1))))
        return xs[idx]

    return {
        "count": n,
        "mean": statistics.fmean(xs) if n else float("nan"),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "min": xs[0] if n else float("nan"),
        "max": xs[-1] if n else float("nan"),
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device):
    """返回峰值显存(MB);非 CUDA 返回 None。"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None
