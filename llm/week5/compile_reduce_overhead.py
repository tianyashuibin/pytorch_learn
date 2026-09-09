"""第 5 周：torch.compile(mode="reduce-overhead") —— 编译器自动帮你上 CUDA Graph。

上一脚本手动 capture CUDA Graph；torch.compile 的 "reduce-overhead" 模式会自动：
  - 用 Inductor 融合 kernel（减少 kernel 数）；
  - 在满足条件时自动套 CUDA Graph（cudagraph_trees.py）。

这正好把"编译器视角"和"引擎视角"摆在一起对比：
  - 编译器：你只管开 mode，它自动 fusion + CUDA Graph，但对动态形状/输入变异敏感。
  - 引擎：手动管这一切，为的是确定性和对 KV cache 布局的极致控制。

对应阅读：
  torch/_inductor/cudagraph_trees.py   (自动 CUDA Graph 的 static address / mutation / shape 处理)
  torch/_inductor/codegen/triton.py    (Inductor 生成的 Triton 长什么样)

配合日志看生成代码：
  TORCH_LOGS="output_code,perf_hints" python compile_reduce_overhead.py
  TORCH_COMPILE_DEBUG=1 python compile_reduce_overhead.py   # 导出完整产物

运行：python compile_reduce_overhead.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_block(dim=1024, device="cuda", dtype=torch.float16):
    return nn.Sequential(
        nn.Linear(dim, dim, bias=False),
        nn.LayerNorm(dim),
        nn.GELU(),
        nn.Linear(dim, dim, bias=False),
    ).to(device=device, dtype=dtype).eval()


def time_cuda(fn, iters=100, warmup=30):
    for _ in range(warmup):  # compile 需要更多 warmup 完成编译
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    if not torch.cuda.is_available():
        print("[skip] 本脚本要 CUDA 才能体现 reduce-overhead 的 CUDA Graph 收益。")
        print("      要点：mode='reduce-overhead' = Inductor 融合 + 自动 CUDA Graph；")
        print("      用 TORCH_LOGS='output_code' 看它生成的 Triton kernel。")
        return

    device, dtype = "cuda", torch.float16
    dim = 1024
    model = build_block(dim, device, dtype)
    x = torch.randn(1, dim, device=device, dtype=dtype)  # decode-like

    @torch.inference_mode()
    def eager():
        return model(x)

    compiled = torch.compile(model, mode="reduce-overhead")

    @torch.inference_mode()
    def run_compiled():
        return compiled(x)

    eager_ms = time_cuda(eager)
    compiled_ms = time_cuda(run_compiled)

    print(f"[env] cuda dim={dim} batch=1 (decode-like)\n")
    print(f"eager:                       {eager_ms:.4f} ms")
    print(f"compile(reduce-overhead):    {compiled_ms:.4f} ms")
    print(f"加速:                        {eager_ms / compiled_ms:.2f}x\n")

    print("=== 编译器视角 vs 引擎视角 ===")
    print("  reduce-overhead 自动完成了你上一脚本手动做的 CUDA Graph capture。")
    print("  但它对动态 shape / 输入地址变化敏感，会触发重编译或退化。")
    print("  引擎(vLLM/SGLang)选择手动管 CUDA Graph + batch 分档，换取确定性和 KV cache 控制。")
    print("\n  想看它到底生成了什么，重跑：")
    print("    TORCH_LOGS='output_code,perf_hints' python compile_reduce_overhead.py")


if __name__ == "__main__":
    main()
