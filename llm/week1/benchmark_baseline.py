"""第 1 周主程序：可信的 LLM 推理 benchmark。

本周要建立的核心心智模型：
  - LLM 推理 = prefill（一次处理整段 prompt） + decode（每次生成 1 个 token）。
  - 这两个阶段负载完全不同，必须分开报指标，不能混成一个平均延迟：
      * TTFT (Time To First Token) ≈ prefill 时间，偏计算受限。
      * TPOT (Time Per Output Token) ≈ 单步 decode 时间，偏访存 + kernel launch 受限。
  - 冷启动（首次调用，含 lazy init / 显存分配 / autotune）绝不能算进稳态平均。

为了看清 prefill/decode，这里不用 model.generate()，而是手写解码循环，
自己管 KV cache（past_key_values），这样每一步都能单独计时。

运行：
    pip install torch transformers
    python benchmark_baseline.py --prompt-len 128 --gen-len 64 --steady 5
"""

from __future__ import annotations

import argparse
import json

import torch

from model import (
    DEFAULT_MODEL,
    load_model_and_tokenizer,
    pick_device,
    pick_dtype,
)
from timing_utils import (
    cuda_timer,
    peak_mem_mb,
    reset_peak_mem,
    summarize,
    sync,
)


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor):
    """处理整段 prompt，返回 (下一个 token, KV cache)。这一步的耗时就是 TTFT 的主体。"""
    out = model(input_ids=input_ids, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 贪心，benchmark 只关心速度
    return next_token, out.past_key_values


@torch.no_grad()
def decode_step(model, token: torch.Tensor, past_key_values):
    """喂入 1 个 token，返回 (下一个 token, 新 KV cache)。单步耗时就是一次 TPOT。"""
    out = model(input_ids=token, past_key_values=past_key_values, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, out.past_key_values


@torch.no_grad()
def run_once(model, input_ids: torch.Tensor, gen_len: int, device: torch.device):
    """完整跑一次 generate，分别记录 prefill 时间和每一步 decode 时间。

    返回 dict：ttft_ms, decode_step_ms(list), e2e_ms
    """
    # ---- prefill ----
    with cuda_timer(device) as t_prefill:
        token, past = prefill(model, input_ids)
    ttft_ms = t_prefill.ms

    # ---- decode 循环，逐步计时 ----
    step_ms: list[float] = []
    for _ in range(gen_len - 1):
        with cuda_timer(device) as t_step:
            token, past = decode_step(model, token, past)
        step_ms.append(t_step.ms)

    e2e_ms = ttft_ms + sum(step_ms)
    return {"ttft_ms": ttft_ms, "decode_step_ms": step_ms, "e2e_ms": e2e_ms}


def build_prompt(tokenizer, prompt_len: int, device: torch.device) -> torch.Tensor:
    """构造一段固定长度的 prompt。用真实文本重复填充到目标 token 数，保证可复现。"""
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * ((prompt_len // 10) + 2)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :prompt_len]
    return ids.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2, help="稳态测量前的预热次数（丢弃）")
    ap.add_argument("--steady", type=int, default=5, help="纳入统计的稳态次数")
    args = ap.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype} torch={torch.__version__}")

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)
    input_ids = build_prompt(tokenizer, args.prompt_len, device)
    print(f"[input] prompt_len={input_ids.shape[1]} gen_len={args.gen_len}")

    # ---- 1) 冷启动：单独记录，绝不进稳态平均 ----
    reset_peak_mem(device)
    sync(device)
    cold = run_once(model, input_ids, args.gen_len, device)
    print(f"[cold] TTFT={cold['ttft_ms']:.2f}ms e2e={cold['e2e_ms']:.2f}ms "
          f"(含 lazy init / 显存分配 / 首次 kernel 编译，不代表稳态)")

    # ---- 2) 预热：让 allocator / kernel 进入稳态，结果丢弃 ----
    for _ in range(args.warmup):
        run_once(model, input_ids, args.gen_len, device)
    sync(device)

    # ---- 3) 稳态测量 ----
    reset_peak_mem(device)
    ttft_samples: list[float] = []
    tpot_samples: list[float] = []  # 把所有 decode step 汇总
    e2e_samples: list[float] = []
    for _ in range(args.steady):
        r = run_once(model, input_ids, args.gen_len, device)
        ttft_samples.append(r["ttft_ms"])
        tpot_samples.extend(r["decode_step_ms"])
        e2e_samples.append(r["e2e_ms"])

    ttft = summarize(ttft_samples)
    tpot = summarize(tpot_samples)
    e2e = summarize(e2e_samples)

    # 吞吐：稳态下每秒生成多少 output token。
    mean_e2e_s = e2e["mean"] / 1000.0
    out_tps = args.gen_len / mean_e2e_s if mean_e2e_s > 0 else float("nan")
    # decode 阶段单独的吞吐（去掉 prefill 的影响）。
    decode_tps = 1000.0 / tpot["mean"] if tpot.get("mean") else float("nan")

    print("\n==================== 稳态结果 ====================")
    print(f"TTFT (prefill)   P50={ttft['p50']:.2f}ms  P99={ttft['p99']:.2f}ms  "
          f"(计算受限，随 prompt_len 增长)")
    print(f"TPOT (per token) P50={tpot['p50']:.2f}ms  P99={tpot['p99']:.2f}ms  "
          f"(访存/launch 受限，随 KV cache 增长缓慢上升)")
    print(f"E2E              P50={e2e['p50']:.2f}ms  P99={e2e['p99']:.2f}ms")
    print(f"吞吐             end-to-end={out_tps:.1f} tok/s   decode-only={decode_tps:.1f} tok/s")
    print(f"峰值显存         {peak_mem_mb(device):.1f} MB")

    # 结构化输出，方便后续存表 / 对比。
    report = {
        "env": {"device": str(device), "dtype": str(dtype), "torch": torch.__version__},
        "input": {"prompt_len": int(input_ids.shape[1]), "gen_len": args.gen_len},
        "cold": {"ttft_ms": cold["ttft_ms"], "e2e_ms": cold["e2e_ms"]},
        "steady": {"ttft": ttft, "tpot": tpot, "e2e": e2e,
                    "out_tps": out_tps, "decode_tps": decode_tps,
                    "peak_mem_mb": peak_mem_mb(device)},
    }
    print("\n[json]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
