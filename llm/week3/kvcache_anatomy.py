"""第 3 周附加实验：把 KV cache 掰开看。

三个问题，用数据回答：
  1. KV cache 占多少显存？公式是什么？随什么增长？
  2. decode 每一步 cache 怎么变化（形状、已用格数）？
  3. 为什么"预分配 max_seq_len"是引擎的常见做法（对照 CUDA Graph / 分页）？

运行：python kvcache_anatomy.py
"""

from __future__ import annotations

import torch

from mini_gpt import GPTConfig, build_mini_gpt
from generate import pick_device, pick_dtype


def kv_cache_bytes(cfg: GPTConfig, seq_len: int, batch: int, dtype_bytes: int) -> int:
    """KV cache 总字节数。
    公式：2(K和V) * layers * batch * n_heads * seq_len * head_dim * dtype_bytes
    """
    head_dim = cfg.dim // cfg.n_heads
    return 2 * cfg.n_layers * batch * cfg.n_heads * seq_len * head_dim * dtype_bytes


def main():
    device = pick_device()
    dtype = pick_dtype(device)
    dtype_bytes = torch.finfo(dtype).bits // 8
    print(f"[env] device={device} dtype={dtype} ({dtype_bytes} bytes/elem)\n")

    cfg = GPTConfig()
    print("=== 问题 1：KV cache 显存随序列长度增长 ===")
    print(f"模型：dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"head_dim={cfg.dim // cfg.n_heads}")
    print("公式：2 * layers * batch * n_heads * seq_len * head_dim * dtype_bytes\n")
    for seq in [128, 512, 2048, 8192]:
        for batch in [1, 32]:
            mb = kv_cache_bytes(cfg, seq, batch, dtype_bytes) / (1024 ** 2)
            print(f"  seq_len={seq:>5}  batch={batch:>3}  ->  KV cache = {mb:8.1f} MB")
    print("  -> KV cache ∝ seq_len × batch，线性增长。长上下文 / 高并发时它就是显存主要消耗者，")
    print("     这正是引擎要做 KV cache 量化(第8周)和分页(第6周)的原因。\n")

    print("=== 问题 2：decode 过程中 cache 的变化 ===")
    model = build_mini_gpt(device, dtype, cfg)
    model.setup_caches(batch=1, device=device, dtype=dtype)
    kv0 = model.kv_caches[0]
    print(f"预分配 cache 形状（固定不变）: K={tuple(kv0.k.shape)}")
    prompt_len = 10
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    with torch.no_grad():
        logits = model(prompt, start_pos=0, causal=True)
    print(f"prefill 后：已用格数 = {prompt_len}（后面全是预留的 0）")
    tok = logits[:, -1, :].argmax(-1, keepdim=True)
    cur = prompt_len
    for step in range(4):
        with torch.no_grad():
            logits = model(tok, start_pos=cur, causal=False)
        tok = logits[:, -1, :].argmax(-1, keepdim=True)
        cur += 1
        print(f"decode step {step}: 写入 cache[{cur-1}]，已用格数 = {cur}，"
              f"cache 张量形状仍是 {tuple(kv0.k.shape)}（地址没变）")

    print("\n=== 问题 3：为什么预分配 max_seq_len ===")
    print("  cache 张量地址在整个 generate 中不变 —— 这是第 5 周 CUDA Graph 能 capture/replay 的前提")
    print("  （CUDA Graph 要求输入/输出地址固定）。代价是即使没生成那么长，显存也先占着；")
    print("  引擎用分页(PagedAttention)把'一大块预留'切成小块按需分配，缓解这个浪费。")


if __name__ == "__main__":
    main()
