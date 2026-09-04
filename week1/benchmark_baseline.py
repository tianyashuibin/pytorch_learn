"""
第 1 周主脚本:建立可信的推理性能基线。

对同一模型测量 5 种配置,并分离"冷启动 / warmup / 稳态":
    1. eager
    2. torch.inference_mode()
    3. torch.compile(mode="default")
    4. torch.compile(mode="reduce-overhead")
    5. torch.compile(mode="max-autotune")

对每种配置记录:
    - 冷启动时间(首次调用,含编译)
    - 稳态 P50 / P90 / P99 延迟、吞吐
    - 峰值显存(仅 CUDA)

运行:
    python benchmark_baseline.py                # 默认 transformer
    python benchmark_baseline.py --model mlp
    python benchmark_baseline.py --dtype float16 --iters 100

本周验收要点:
- 为什么异步 CUDA 计时前后要 synchronize?  见 timing_utils.timed。
- 为什么第一次调用不能计入稳态平均?        见下方 cold-start 与 steady-state 的分离。
"""
import argparse

import torch

from model import build_model
from timing_utils import (
    get_device,
    peak_memory_mb,
    reset_peak_memory,
    summarize,
    sync,
    timed,
)


def make_runner(model, mode: str):
    """返回一个可调用对象,代表某种执行配置。"""
    if mode == "eager":
        return model
    if mode == "inference_mode":
        def run(x):
            with torch.inference_mode():
                return model(x)
        return run
    if mode.startswith("compile:"):
        compile_mode = mode.split(":", 1)[1]
        # 每种 compile 配置用独立的 compiled 对象,避免相互影响缓存
        return torch.compile(model, mode=compile_mode)
    raise ValueError(mode)


def bench_one(model, example, device, mode: str, warmup: int, iters: int):
    import torch._dynamo as dynamo

    dynamo.reset()  # 清掉上一配置的编译状态,保证冷启动可比
    runner = make_runner(model, mode)

    reset_peak_memory(device)

    # ---- 冷启动:首次调用,含编译 / autotune / cudnn benchmark ----
    with torch.inference_mode():
        with timed(device) as get_ms:
            runner(example)
        cold_ms = get_ms()

        # ---- warmup:让 allocator、缓存、autotune 结果都稳定下来 ----
        for _ in range(warmup):
            runner(example)
        sync(device)

        # ---- 稳态:逐次计时,取分布 ----
        latencies = []
        for _ in range(iters):
            with timed(device) as get_ms:
                runner(example)
            latencies.append(get_ms())

    stats = summarize(latencies)
    batch = example.shape[0]
    throughput = batch * 1000.0 / stats["p50"] if stats["p50"] > 0 else float("nan")
    return {
        "mode": mode,
        "cold_ms": cold_ms,
        "throughput": throughput,
        "peak_mb": peak_memory_mb(device),
        **stats,
    }


def print_table(rows):
    header = f"{'配置':<26}{'冷启动ms':>10}{'P50ms':>9}{'P90ms':>9}{'P99ms':>9}{'吞吐/s':>11}{'峰值MB':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        peak = f"{r['peak_mb']:.1f}" if r["peak_mb"] is not None else "N/A"
        print(
            f"{r['mode']:<26}{r['cold_ms']:>10.2f}{r['p50']:>9.3f}"
            f"{r['p90']:>9.3f}{r['p99']:>9.3f}{r['throughput']:>11.1f}{peak:>10}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    model, example = build_model(args.model, device, dtype)
    print(f"设备={device}  模型={args.model}  dtype={args.dtype}  "
          f"warmup={args.warmup}  iters={args.iters}")
    print(f"输入形状={tuple(example.shape)}\n")

    modes = [
        "eager",
        "inference_mode",
        "compile:default",
        "compile:reduce-overhead",
        "compile:max-autotune",
    ]

    rows = []
    for mode in modes:
        try:
            rows.append(bench_one(model, example, device, mode, args.warmup, args.iters))
        except Exception as e:  # 某些 compile 模式在 CPU/MPS 上可能不支持,跳过但记录
            print(f"[跳过] {mode}: {type(e).__name__}: {e}")
    print()
    print_table(rows)

    print("\n提示:")
    print("  - 冷启动远高于稳态 = 编译/autotune 开销,不能计入稳态平均。")
    print("  - reduce-overhead 走 CUDA Graph,主要降低 kernel launch / Python 开销。")
    print("  - 想看图捕获/生成代码,配合 TORCH_LOGS 环境变量重跑(见计划第五节速查)。")


if __name__ == "__main__":
    main()
