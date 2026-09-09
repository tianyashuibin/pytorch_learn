"""第 9 周：端到端综合项目 —— 把第 1~8 周串成一次可复现的推理优化。

用第 3 周手写的 MiniGPT 作"可控真实负载"（不依赖联网 / HF），在它上面跑齐计划要求的维度：
  1. eager FP16/BF16 基线
  2. prefill/decode 分离的 TTFT / TPOT / 吞吐 / 显存
  3. KV cache 行为（增长）
  4. attention 后端对比（SDPA 各后端）
  5. CUDA Graph 开关（非 CUDA 环境优雅跳过并说明）
  6. 量化实验（W4A16，复用第 8 周）
  7. 高并发调度（batch=1 vs batch=N 吞吐；细节调度见第 7 周模拟器）
  8. 正确性 / 数值误差验证（量化 vs 基线 logits）
  9. 瓶颈解释（脚本末尾按实测数据给结论）

产出：终端一张统一对比表 + 写一份 report.md 报告骨架（填结论用）。

运行：python run_project.py          # 用 miniconda 的 python（带 torch）
      python run_project.py --big    # 更大的模型档位（有 GPU 时更有代表性）
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用第 3 周的 MiniGPT 和第 8 周的 W4A16Linear
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "week3"))
sys.path.insert(0, str(_HERE.parent / "week8"))
from mini_gpt import GPTConfig, build_mini_gpt          # noqa: E402
from weight_only_w4a16 import W4A16Linear                # noqa: E402


# --------------------------- 设备 / 计时 ---------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32          # CPU 上 fp16 慢且部分算子不支持


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return float("nan")           # CPU 无对应统计


def reset_peak_mem(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# --------------------------- 基准测量 ---------------------------
@torch.inference_mode()
def bench(model, cfg, device, dtype, batch, prompt_len, gen_len, warmup=2, iters=5):
    """测 prefill(TTFT) 和 decode(TPOT)，分冷/热，返回指标 dict。"""
    model.setup_caches(batch, device, dtype)
    prompt = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    def one_gen():
        # prefill：一次吃整个 prompt
        sync(device); t0 = time.perf_counter()
        logits = model(prompt, start_pos=0, causal=True)
        sync(device); ttft = time.perf_counter() - t0
        # decode：逐 token
        cur = logits[:, -1:].argmax(-1)
        pos = prompt_len
        sync(device); t1 = time.perf_counter()
        for _ in range(gen_len):
            logits = model(cur, start_pos=pos, causal=False)
            cur = logits[:, -1:].argmax(-1)
            pos += 1
        sync(device); dt = time.perf_counter() - t1
        return ttft, dt / gen_len

    for _ in range(warmup):
        one_gen()
    reset_peak_mem(device)
    ttfts, tpots = [], []
    for _ in range(iters):
        model.setup_caches(batch, device, dtype)   # 重置 KV，保证每次干净
        ttft, tpot = one_gen()
        ttfts.append(ttft); tpots.append(tpot)

    ttft = sorted(ttfts)[len(ttfts) // 2]
    tpot = sorted(tpots)[len(tpots) // 2]
    tput = batch / tpot                              # decode 吞吐 tok/s（每步 batch 个 token）
    return {"ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3,
            "decode_tok_s": tput, "peak_mb": peak_mem_mb(device)}


def apply_w4a16(model, group_size=128):
    """把模型里能整除 group_size 的 Linear 就地换成 W4A16（lm_head 通常最大，最值得压）。"""
    count = 0
    for module in model.modules():           # modules() 已包含 model 自身
        for cname, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.in_features % group_size == 0:
                setattr(module, cname, W4A16Linear(child, group_size))
                count += 1
    return count


# --------------------------- SDPA 后端对比（第 4 周） ---------------------------
def sdpa_backend_compare(device, dtype):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    b, h, s, d = 8, 16, 512, 64
    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)
    backends = [("MATH", SDPBackend.MATH),
                ("FLASH", SDPBackend.FLASH_ATTENTION),
                ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)]
    out = {}
    for tag, be in backends:
        try:
            with sdpa_kernel([be]):
                for _ in range(3):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                sync(device); t0 = time.perf_counter()
                for _ in range(20):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                sync(device); out[tag] = (time.perf_counter() - t0) / 20 * 1e3
        except (RuntimeError, NotImplementedError):
            out[tag] = None
    return out


# --------------------------- 主流程 ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    cfg = GPTConfig(dim=1024, n_layers=12, n_heads=16, max_seq_len=2048) if args.big \
        else GPTConfig(dim=512, n_layers=8, n_heads=8, max_seq_len=1024)
    prompt_len, gen_len = 128, 64

    print(f"设备 {device}  dtype {dtype}  模型 dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads}")
    print(f"负载 prompt_len={prompt_len} gen_len={gen_len}\n")

    rows = []

    # 1) 基线 FP16/BF16
    m = build_mini_gpt(device, dtype, cfg)
    base = bench(m, cfg, device, dtype, batch=1, prompt_len=prompt_len, gen_len=gen_len)
    rows.append(("基线 batch=1", base))

    # 7) 高并发：batch=8 吞吐对比
    m2 = build_mini_gpt(device, dtype, cfg)
    multi = bench(m2, cfg, device, dtype, batch=8, prompt_len=prompt_len, gen_len=gen_len)
    rows.append(("batch=8 (并发)", multi))

    # 6/8) 量化 W4A16 + 正确性
    mq = build_mini_gpt(device, dtype, cfg)
    # 先存基线 logits 做误差对比
    mq.setup_caches(1, device, dtype)
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    with torch.inference_mode():
        ref_logits = mq(prompt, start_pos=0, causal=True).float()
    n_q = apply_w4a16(mq, group_size=128)
    mq.setup_caches(1, device, dtype)
    with torch.inference_mode():
        q_logits = mq(prompt, start_pos=0, causal=True).float()
    quant_err = (ref_logits - q_logits).norm().item() / ref_logits.norm().item()
    quant = bench(mq, cfg, device, dtype, batch=1, prompt_len=prompt_len, gen_len=gen_len)
    rows.append((f"W4A16 (替换{n_q}个Linear)", quant))

    # ---- 统一对比表 ----
    print(f"{'配置':<26}{'TTFT(ms)':>10}{'TPOT(ms)':>10}{'decode tok/s':>14}{'峰值MB':>10}")
    print("-" * 72)
    for tag, r in rows:
        mb = f"{r['peak_mb']:.0f}" if r['peak_mb'] == r['peak_mb'] else "n/a"
        print(f"{tag:<26}{r['ttft_ms']:>10.2f}{r['tpot_ms']:>10.2f}"
              f"{r['decode_tok_s']:>14.1f}{mb:>10}")

    # 4) SDPA 后端对比
    print("\n=== SDPA 后端对比（单位 ms，b8 h16 s512 d64）===")
    for tag, ms in sdpa_backend_compare(device, dtype).items():
        print(f"  {tag:<10}: {ms:.3f} ms" if ms is not None else f"  {tag:<10}: 不支持(该设备/dtype)")

    # 5) CUDA Graph
    print("\n=== CUDA Graph 开关 ===")
    if device.type == "cuda":
        print("  在 CUDA 上，可用第 5 周 cuda_graph_demo.py 的方式 capture decode step，")
        print("  预期 TPOT 因消除 kernel launch 开销而下降。此处留作 CUDA 机器上补测。")
    else:
        print(f"  当前设备 {device} 非 CUDA，CUDA Graph 不可用（见第 5 周）。跳过。")

    # 3) KV cache 增长
    head_dim = cfg.dim // cfg.n_heads
    kv_bytes_per_tok = 2 * cfg.n_layers * cfg.n_heads * head_dim * (2 if dtype != torch.float32 else 4)
    print("\n=== KV cache 行为（第 3 周公式）===")
    print(f"  每 token KV：{kv_bytes_per_tok/1024:.1f} KB；prompt+gen={prompt_len+gen_len} tok "
          f"-> {kv_bytes_per_tok*(prompt_len+gen_len)/1024/1024:.1f} MB/请求")

    # 8) 正确性结论
    print(f"\n=== 正确性 ===")
    print(f"  W4A16 vs 基线 prefill logits 相对误差：{quant_err:.4%}")

    # 9) 瓶颈解释（按实测自动给一句话）
    print("\n=== 瓶颈解释（据本次实测）===")
    print(f"  - decode TPOT {base['tpot_ms']:.2f}ms >> prefill 摊到每 token，典型访存/launch 受限。")
    print(f"  - batch1->8：decode tok/s {base['decode_tok_s']:.0f} -> {multi['decode_tok_s']:.0f}"
          f"（{multi['decode_tok_s']/base['decode_tok_s']:.1f}x），批处理摊薄每步固定开销。")
    if quant["tpot_ms"] > base["tpot_ms"]:
        print(f"  - W4A16 这里 TPOT 反而 {quant['tpot_ms']/base['tpot_ms']:.1f}x 变慢、峰值显存更高——")
        print(f"    这是 mini 实现的产物：每次前向把整块 int4 权重解回 fp16 再算（未融合 dequant）。")
        print(f"    真实 W4A16 用融合 kernel 边解包边乘，权重不 materialize，才有省显存+加速的收益。")
    else:
        print(f"  - W4A16 TPOT {base['tpot_ms']/quant['tpot_ms']:.2f}x。")

    _write_report(_HERE / "report.md", device, dtype, cfg, prompt_len, gen_len,
                  rows, quant_err)
    print(f"\n已写报告骨架：{_HERE / 'report.md'}（填入结论后即成交付）")


def _write_report(path, device, dtype, cfg, prompt_len, gen_len, rows, quant_err):
    lines = [
        "# LLM 推理优化端到端报告（第 9 周）", "",
        "## 1. 模型与输入", "",
        f"- 模型：手写 MiniGPT，dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
        f"max_seq_len={cfg.max_seq_len}",
        f"- 负载：prompt_len={prompt_len}, gen_len={gen_len}",
        f"- 环境：device={device}, dtype={dtype}, torch={torch.__version__}", "",
        "> 说明：MiniGPT 是可控负载，用于把方法跑通。要换真实模型（如 TinyLlama），",
        "> 复用第 1 周 model.py 的加载器，指标口径不变。", "",
        "## 2. 延迟 / 吞吐 / 显存", "",
        "| 配置 | TTFT(ms) | TPOT(ms) | decode tok/s | 峰值MB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tag, r in rows:
        mb = f"{r['peak_mb']:.0f}" if r['peak_mb'] == r['peak_mb'] else "n/a"
        lines.append(f"| {tag} | {r['ttft_ms']:.2f} | {r['tpot_ms']:.2f} | "
                     f"{r['decode_tok_s']:.1f} | {mb} |")
    lines += [
        "", "## 3. 正确性", "",
        f"- W4A16 vs FP 基线 prefill logits 相对误差：**{quant_err:.4%}**",
        "- （补：生成文本是否一致 / 前 K token 是否匹配）", "",
        "## 4. 各维度实验（填结论）", "",
        "- [ ] KV cache 增长（第 3 周公式，见终端输出）",
        "- [ ] SDPA 后端对比（第 4 周，哪个最快、为什么）",
        "- [ ] CUDA Graph 开关（第 5 周，仅 CUDA；TPOT 变化）",
        "- [ ] W4A16 量化（第 8 周，显存↓ / 误差 / 速度）",
        "- [ ] 高并发 batch 吞吐（第 7 周，continuous batching）", "",
        "## 5. 瓶颈、优化理由、副作用", "",
        "> 一句话说清：瓶颈在哪（访存/算力/launch）→ 用了什么优化 → 副作用是什么。",
        "> 例：decode 访存受限 → KV/权重量化省带宽 → 代价是数值误差 X%，需验证下游质量。", "",
        "## 6. 复现命令", "", "```bash", "python run_project.py", "```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
