"""
第 1 周补充:torch.utils.benchmark 与 torch.profiler 的基本用法。

- torch.utils.benchmark.Timer: 自动处理 warmup 和多次测量,内部会做 CUDA 同步,
  比手写 time.perf_counter 更省心,适合快速对比两段代码。
- torch.profiler: 把时间拆成算子 / kernel 级别,是"判断瓶颈"的主力工具。

运行:
    python profiler_demo.py
"""
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile

from model import build_model
from timing_utils import get_device


def timer_compare(model, example, device):
    print("=" * 70)
    print("[1] torch.utils.benchmark.Timer:eager vs compile 稳态延迟")
    print("=" * 70)
    compiled = torch.compile(model)
    with torch.inference_mode():
        compiled(example)  # 触发编译,不计入

    t_eager = benchmark.Timer(
        stmt="model(x)",
        globals={"model": model, "x": example},
        label="inference",
        sub_label="eager",
    )
    t_compiled = benchmark.Timer(
        stmt="m(x)",
        globals={"m": compiled, "x": example},
        label="inference",
        sub_label="compile",
    )
    with torch.inference_mode():
        print(t_eager.blocked_autorange(min_run_time=1.0))
        print(t_compiled.blocked_autorange(min_run_time=1.0))


def profiler_breakdown(model, example, device):
    print("=" * 70)
    print("[2] torch.profiler:把时间拆到算子 / kernel")
    print("=" * 70)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        for _ in range(3):  # warmup
            model(example)
        with profile(activities=activities, record_shapes=True) as prof:
            model(example)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
    print("\n-> 找出 top-3 算子,判断偏计算受限还是带宽受限(第 2 周会深入)。")


def main():
    device = get_device()
    torch.manual_seed(0)
    model, example = build_model("transformer", device)
    print(f"设备={device}\n")
    timer_compare(model, example, device)
    profiler_breakdown(model, example, device)


if __name__ == "__main__":
    main()
