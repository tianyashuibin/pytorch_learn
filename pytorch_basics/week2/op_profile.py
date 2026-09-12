"""
第 2 周实践 1 & 2:profiler 分析 eager 模型,并把时间拆成各阶段。

目标拆分(计划要求):
    Python 开销 / dispatcher / 算子执行 / kernel launch / 同步等待

profiler 的近似口径:
- CPU self time  ≈ 框架 + dispatcher + Python 调度 + kernel launch(host 侧)
- CUDA self time ≈ GPU 上真正执行 kernel 的时间(device 侧)
- 两者的差,配合"Self CPU 很高但 CUDA 很低"可判断是 launch/框架 bound。

运行:
    python op_profile.py                 # 默认 transformer
    python op_profile.py --model mlp
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from _common import build_model, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(0)
    model, example = build_model(args.model, device)
    print(f"设备={device}  模型={args.model}\n")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        for _ in range(5):  # warmup:排除首次分配 / cudnn benchmark
            model(example)

        with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
            for _ in range(args.iters):
                model(example)

    # ---- 按算子聚合,找 top 算子 ----
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("=" * 90)
    print("[1] 按算子聚合(找最耗时的 top-3)")
    print("=" * 90)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))

    # ---- 打印一次前向的算子调用树,直观看父子包含关系 ----
    print("=" * 90)
    print("[1b] 算子调用树(展示父子包含关系,只打印每种顶层算子的第一次调用)")
    print("=" * 90)

    def fmt_us(ev):
        # FunctionEvent 时间单位为 us
        return f"{ev.self_cpu_time_total:.1f}us self-cpu, {getattr(ev, 'self_device_time_total', 0):.1f}us self-cuda"

    def print_tree(ev, depth=0):
        indent = "  " * depth
        print(f"{indent}{ev.name}  [{fmt_us(ev)}]")
        # 该算子直接启动的 GPU kernel
        for k in getattr(ev, "kernels", None) or []:
            print(f"{indent}  └─(kernel) {k.name}  [{k.duration:.1f}us]")
        for child in getattr(ev, "cpu_children", None) or []:
            print_tree(child, depth + 1)

    seen = set()
    for ev in prof.events():
        if getattr(ev, "cpu_parent", None) is not None:
            continue  # 只从顶层算子开始
        if ev.name in seen:
            continue  # 每种顶层算子只画一次(iters 会重复很多遍)
        seen.add(ev.name)
        print_tree(ev)

    # ---- 粗略总账:CPU 侧 vs GPU 侧 ----
    ka = prof.key_averages()
    # 注意:profiler 里时间字段单位是微秒(us),这里换算成 ms 打印。
    total_cpu = sum(e.self_cpu_time_total for e in ka)
    # 只对"真正的 GPU kernel"求和,避免重复计数:
    # aten 算子(如 addmm)的 self_device_time_total 会把它启动的 kernel 时间也算进去,
    # 而 kernel 自身条目(如 volta_sgemm)又记了一遍。kernel 条目没有 CPU 侧时间(self_cpu==0),
    # 以此过滤出叶子 kernel,与 profiler 的 "Self CUDA time total" 对齐。
    total_cuda = sum(
        getattr(e, "self_device_time_total", 0)
        for e in ka
        if e.self_cpu_time_total == 0 and getattr(e, "self_device_time_total", 0) > 0
    )
    print("=" * 90)
    print("[2] 阶段总账(单位 ms,近似)")
    print("=" * 90)
    print(f"  Self CPU 合计 (框架+dispatcher+Python+launch): {total_cpu/1e3:10.1f} ms")
    if device.type == "cuda":
        print(f"  Self CUDA 合计 (GPU kernel 真正执行)         : {total_cuda/1e3:10.1f} ms")
        ratio = total_cpu / total_cuda if total_cuda else float("inf")
        print(f"  CPU/CUDA 比值 = {ratio:.2f}")
        if ratio > 3:
            print("   -> 比值远大于 1 = launch / 框架 bound(CPU 侧开销主导,小算子太多,该融合 / 上 CUDA Graph)。")
        elif ratio >= 1:
            print("   -> 比值接近 1 = CPU 与 GPU 时间大致平衡,可关注 CPU 侧调度是否还能压缩。")
        else:
            print("   -> 比值小于 1 = GPU 计算受限(kernel 吃满时间,较健康),继续优化看算力 / 带宽。")
    else:
        print("  (当前非 CUDA 设备,GPU kernel 时间无法单独统计)")

    print("\n[3] 判断计算受限 vs 带宽受限:见 compute_vs_bandwidth.py")


if __name__ == "__main__":
    main()
