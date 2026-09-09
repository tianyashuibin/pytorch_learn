"""第 1 周附加实验：用数据验证两条核心规律。

规律 1：TTFT 随 prompt_len 上升（prefill 计算量 ∝ 序列长度）。
规律 2：TPOT 随生成进行缓慢上升（KV cache 变长，每步 attention 要读更多 K/V）。

跑完后你应该能亲眼看到 prefill 和 decode 是两种不同负载。

运行：
    python scaling_experiment.py
"""

from __future__ import annotations

import torch

from model import DEFAULT_MODEL, load_model_and_tokenizer, pick_device, pick_dtype
from timing_utils import cuda_timer, summarize, sync
from benchmark_baseline import build_prompt, decode_step, prefill


@torch.no_grad()
def measure_ttft(model, input_ids, device, repeat=5) -> dict:
    # 预热
    prefill(model, input_ids)
    sync(device)
    samples = []
    for _ in range(repeat):
        with cuda_timer(device) as t:
            prefill(model, input_ids)
        samples.append(t.ms)
    return summarize(samples)


@torch.no_grad()
def measure_tpot_drift(model, input_ids, device, gen_len=128) -> list[float]:
    """记录每一步 decode 的耗时，看它是否随 KV cache 增长而上升。"""
    token, past = prefill(model, input_ids)
    sync(device)
    step_ms = []
    for _ in range(gen_len):
        with cuda_timer(device) as t:
            token, past = decode_step(model, token, past)
        step_ms.append(t.ms)
    return step_ms


def main() -> None:
    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype}")
    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL, device, dtype)

    print("\n=== 规律 1：TTFT vs prompt_len ===")
    for plen in [32, 128, 512, 1024]:
        ids = build_prompt(tokenizer, plen, device)
        s = measure_ttft(model, ids, device)
        print(f"prompt_len={plen:>5}  TTFT P50={s['p50']:.2f}ms  mean={s['mean']:.2f}ms")

    print("\n=== 规律 2：TPOT 随 decode 步数漂移（KV cache 增长）===")
    ids = build_prompt(tokenizer, 128, device)
    steps = measure_tpot_drift(model, ids, device, gen_len=256)
    # 分段看均值，规避单步抖动。
    n = len(steps)
    for lo, hi in [(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]:
        seg = steps[lo:hi]
        avg = sum(seg) / len(seg)
        print(f"decode step [{lo:>3}-{hi:>3})  mean={avg:.3f}ms")
    print("若后段明显高于前段，说明访存随 KV cache 增长——这正是 decode 访存受限的证据。")


if __name__ == "__main__":
    main()
