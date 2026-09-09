"""第 3 周：用 mini GPT 手写 prefill + decode 循环，看清一次 generate 的数据流。

数据流（本周要能默画出来）：
    tokens = tokenize(prompt)
    ── prefill：model(tokens[0:P], start_pos=0)  一次处理整段，填满 KV cache 的前 P 格
    │            └─ 取最后一个位置的 logits -> 采样 -> 第 1 个生成 token
    └── decode 循环：每步
                 model(token, start_pos=当前长度)  只处理 1 个 token，往 cache 追加 1 格
                 └─ logits -> 采样 -> 下一个 token
       直到生成够长或遇到 EOS

关键对比：
  - prefill 一次前向的 seq = prompt_len（大矩阵乘，计算受限）
  - decode 每次前向的 seq = 1（小矩阵乘 + 读越来越长的 KV cache，访存/launch 受限）

运行：python generate.py --prompt-len 64 --gen-len 32 --verbose
"""

from __future__ import annotations

import argparse

import torch

from mini_gpt import GPTConfig, build_mini_gpt


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(model, prompt_ids, gen_len, device, verbose=False):
    B, prompt_len = prompt_ids.shape

    # ---- prefill：start_pos=0，一次喂入整段 prompt，causal=True ----
    logits = model(prompt_ids, start_pos=0, causal=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 取最后位置
    generated = [next_token]
    cur_len = prompt_len

    if verbose:
        kv0 = model.kv_caches[0]
        used = kv0.k[:, :, :cur_len]
        print(f"[prefill] 处理 {prompt_len} 个 token，KV cache 已填入 {cur_len} 格；"
              f"cache 张量形状={tuple(kv0.k.shape)}，已用={tuple(used.shape)}")

    # ---- decode 循环：每次 start_pos=cur_len，只喂 1 个 token，causal 交给 cache 长度 ----
    for step in range(gen_len - 1):
        logits = model(next_token, start_pos=cur_len, causal=False)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        cur_len += 1
        if verbose and step < 3:
            print(f"[decode step {step}] 处理 1 个 token，start_pos={cur_len - 1}，"
                  f"attention 读取历史长度={cur_len}")

    return torch.cat(generated, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[env] device={device} dtype={dtype}")

    cfg = GPTConfig(dim=args.dim, n_layers=args.layers)
    model = build_mini_gpt(device, dtype, cfg)
    model.setup_caches(batch=1, device=device, dtype=dtype)

    # 随机 prompt（本周关心机制，不关心语义）
    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)

    # ---- 分别计时 prefill 和 decode ----
    # 预热
    generate(model, prompt, gen_len=4, device=device)
    model.setup_caches(batch=1, device=device, dtype=dtype)  # 重置 cache
    sync(device)

    # 单独测 prefill
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    import time
    if device.type == "cuda":
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        with torch.no_grad():
            logits = model(prompt, start_pos=0, causal=True)
        e1.record(); sync(device)
        prefill_ms = e0.elapsed_time(e1)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(prompt, start_pos=0, causal=True)
        prefill_ms = (time.perf_counter() - t0) * 1000

    # 单独测一步 decode
    tok = logits[:, -1, :].argmax(-1, keepdim=True)
    if device.type == "cuda":
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        with torch.no_grad():
            model(tok, start_pos=args.prompt_len, causal=False)
        e1.record(); sync(device)
        decode_ms = e0.elapsed_time(e1)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            model(tok, start_pos=args.prompt_len, causal=False)
        decode_ms = (time.perf_counter() - t0) * 1000

    # 完整生成（带 verbose 打印数据流）
    model.setup_caches(batch=1, device=device, dtype=dtype)
    out = generate(model, prompt, args.gen_len, device, verbose=args.verbose)

    print("\n==================== prefill vs decode ====================")
    print(f"prefill（{args.prompt_len} tokens 一次）: {prefill_ms:.3f} ms")
    print(f"decode （1 token 一步）           : {decode_ms:.3f} ms")
    print(f"每 token 摊薄：prefill {prefill_ms / args.prompt_len:.3f} ms/tok  vs  decode {decode_ms:.3f} ms/tok")
    print(f"生成 token 数：{out.shape[1]}")
    print("\n要点：prefill 摊到每 token 更快（大矩阵乘吃满算力），")
    print("      decode 每 token 更贵（seq=1 的小 kernel，且要读越来越长的 KV cache）。")


if __name__ == "__main__":
    main()
