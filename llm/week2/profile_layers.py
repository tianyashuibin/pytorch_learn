"""第 2 周：分层归因 —— 把一次前向的时间拆成 Python / dispatch / kernel / 同步。

本周验收点：看到性能问题时，先判断它属于
  (1) 框架开销（Python + dispatcher）  (2) 算子实现  (3) 内存分配  (4) GPU kernel

profiler 里的对应关系（近似）：
  - CPU self time  ≈ Python + dispatcher + kernel launch 的主机侧开销
  - CUDA time      ≈ GPU 上 kernel 真正的执行时间
  - 若 CPU 总时间 >> CUDA 总时间：说明 GPU 在等主机（launch/Python 受限）——decode 阶段常见。
  - 若 CUDA 时间占主导：说明真在算（compute/bandwidth 受限）——prefill / 大 GEMM 常见。

运行：
    python profile_layers.py          # 有 transformers 则跑真实 GPT，否则跑 attention microbench
    python profile_layers.py --trace  # 额外导出 chrome trace，可在 chrome://tracing 打开
"""

from __future__ import annotations

import argparse

import torch

from _common import pick_device, pick_dtype, sync, try_load_llm


def make_workload(device: torch.device, dtype: torch.dtype):
    """返回一个 forward() 闭包。优先真实 GPT，否则用一段 attention-like microbench。"""
    loaded = try_load_llm(device, dtype)
    if loaded is not None:
        model, tok = loaded
        ids = tok("The quick brown fox jumps over the lazy dog. " * 8,
                  return_tensors="pt").input_ids.to(device)

        @torch.inference_mode()
        def forward():
            model(input_ids=ids, use_cache=True)
        return forward, "TinyLlama forward (prefill)"

    # ---- 退路：手搓一个 attention + MLP 块，纯 Tensor，不依赖 transformers ----
    d, seq, heads = 1024, 256, 16
    hd = d // heads
    q = torch.randn(1, heads, seq, hd, device=device, dtype=dtype)
    k = torch.randn(1, heads, seq, hd, device=device, dtype=dtype)
    v = torch.randn(1, heads, seq, hd, device=device, dtype=dtype)
    w1 = torch.randn(d, 4 * d, device=device, dtype=dtype)
    w2 = torch.randn(4 * d, d, device=device, dtype=dtype)
    x = torch.randn(1, seq, d, device=device, dtype=dtype)

    @torch.inference_mode()
    def forward():
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        _ = attn.transpose(1, 2).reshape(1, seq, d)
        h = torch.nn.functional.gelu(x @ w1)
        _ = h @ w2
    return forward, "attention+MLP microbench"


def profile_forward(forward, device: torch.device, name: str, export_trace: bool) -> None:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # 预热：把冷启动 / autotune / allocator 首次分配排除
    for _ in range(3):
        forward()
    sync(device)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(10):
            forward()
        sync(device)

    print(f"=== 分层归因：{name} ===")
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))

    # 汇总 CPU vs CUDA 总时间，给一个粗判断
    ka = prof.key_averages()
    cpu_total = sum(e.cpu_time_total for e in ka)
    cuda_total = sum(getattr(e, "cuda_time_total", 0) or getattr(e, "device_time_total", 0)
                     for e in ka) if device.type == "cuda" else 0

    print("\n--- 粗判断 ---")
    print(f"CPU self 总时间 ≈ {cpu_total / 1000:.2f} ms（Python + dispatch + launch）")
    if device.type == "cuda":
        print(f"CUDA 总时间     ≈ {cuda_total / 1000:.2f} ms（kernel 执行）")
        if cuda_total > 0 and cpu_total > cuda_total * 1.5:
            print(">> CPU 明显大于 GPU：偏 launch/Python 受限（典型 decode 场景，考虑 CUDA Graph）")
        elif cuda_total > 0:
            print(">> GPU 占主导：偏 compute/bandwidth 受限（典型 prefill / 大 GEMM）")
    else:
        print("（CPU 设备无 CUDA 分解；换到 GPU 才能看清 launch vs kernel）")

    if export_trace:
        out = "trace_forward.json"
        prof.export_chrome_trace(out)
        print(f"\n[trace] 已导出 {out}，用 chrome://tracing 或 https://ui.perfetto.dev 打开")
    print()


def top_ops_compute_vs_bw(prof_note: str = "") -> None:
    print("提示：在上面表里找 self CUDA time 最高的 3 个算子，")
    print("      matmul/attention 类通常 compute 受限，elementwise/norm/copy 类通常 bandwidth 受限。")
    print("      这就是本周验收要求的'找出三个最耗时算子并判断受限类型'。\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true", help="导出 chrome trace")
    args = ap.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype} torch={torch.__version__}\n")

    forward, name = make_workload(device, dtype)
    profile_forward(forward, device, name, args.trace)
    top_ops_compute_vs_bw()


if __name__ == "__main__":
    main()
