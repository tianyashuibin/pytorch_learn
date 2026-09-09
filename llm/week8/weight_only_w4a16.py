"""第 8 周（2/3）：W4A16 权重量化的最小实现 + 为什么 decode 场景收益大。

W4A16 = 权重存 int4（4-bit），激活保持 fp16（16-bit）。这是 LLM 部署最主流的方案之一
（GPTQ/AWQ 都属于 W4A16 家族，区别在“怎么选 scale / 是否补偿误差”，机制骨架相同）。

核心机制（本文件实现）：
  - 权重按 group-wise int4 量化并“打包”（两个 int4 塞进一个 uint8，显存真正减半再减半）。
  - 前向时：反量化权重回 fp16 -> 正常 fp16 matmul（所以叫 A16，激活不量化）。
  - 对比 fp16 Linear：权重显存、输出误差。

为什么 decode 阶段 W4A16 收益大（关键结论）：
  decode 每步只算 1 个 token，矩阵乘是“瘦高 × 大权重”，**访存受限**（memory-bound）——
  瓶颈是把权重从显存搬进片上。权重从 16-bit 压到 4-bit，搬运量降到 1/4，decode 直接更快。
  （prefill 是 compute-bound，压权重对算力瓶颈帮助小，这也是为什么 A 保持 16-bit 够用。）

不依赖 GPU（CPU/MPS 都能跑，测的是误差和显存，不是绝对速度）。
运行：python weight_only_w4a16.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_weight_int4_grouped(W: torch.Tensor, group_size: int = 128):
    """把权重 (out, in) 按每行内每 group 个元素做对称 int4 量化。

    返回 (q_packed uint8, scale, group_size)。q 范围 [-8, 7]，打包时 +8 变 [0,15]。
    """
    out_features, in_features = W.shape
    assert in_features % group_size == 0, "in_features 必须能被 group_size 整除"
    Wg = W.reshape(out_features, in_features // group_size, group_size)
    qmax = 7
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    q = torch.round(Wg / scale).clamp(-8, 7).to(torch.int8)   # int4 值域，用 int8 承载
    # 打包：+8 移到 [0,15]，相邻两个塞进一个 uint8
    q_u = (q + 8).to(torch.uint8).reshape(out_features, in_features)
    q_packed = (q_u[:, 0::2] | (q_u[:, 1::2] << 4)).contiguous()  # 高低半字节
    return q_packed, scale.squeeze(-1), group_size


def dequantize_weight_int4_grouped(q_packed, scale, group_size, in_features):
    """解包 + 反量化回 fp16 权重。"""
    out_features = q_packed.shape[0]
    low = (q_packed & 0x0F).to(torch.int16) - 8
    high = ((q_packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack([low, high], dim=-1).reshape(out_features, in_features).float()
    qg = q.reshape(out_features, in_features // group_size, group_size)
    W = (qg * scale.unsqueeze(-1)).reshape(out_features, in_features)
    return W


class W4A16Linear(nn.Module):
    """演示用 W4A16 线性层：权重存 int4 打包，前向反量化后走 fp16 matmul。"""

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        q_packed, scale, _ = quantize_weight_int4_grouped(linear.weight.data.float(), group_size)
        self.register_buffer("q_packed", q_packed)          # uint8, 半个 in_features 宽
        self.register_buffer("scale", scale)                 # fp
        self.bias = linear.bias

    def forward(self, x):
        W = dequantize_weight_int4_grouped(
            self.q_packed, self.scale, self.group_size, self.in_features
        ).to(x.dtype)
        return torch.nn.functional.linear(x, W, self.bias)


def tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024 / 1024


def demo():
    torch.manual_seed(0)
    in_f, out_f, gs = 4096, 4096, 128
    lin = nn.Linear(in_f, out_f, bias=False)

    fp16_w_mb = lin.weight.numel() * 2 / 1024 / 1024   # 若以 fp16 存
    ql = W4A16Linear(lin, group_size=gs)
    int4_w_mb = tensor_mb(ql.q_packed) + tensor_mb(ql.scale)

    print(f"Linear({in_f}x{out_f}) 权重显存：")
    print(f"  fp16 权重          : {fp16_w_mb:8.2f} MB")
    print(f"  int4 打包 + scale  : {int4_w_mb:8.2f} MB  (含 group={gs} 的 scale)")
    print(f"  压缩比             : {fp16_w_mb / int4_w_mb:8.2f}x")

    # 输出误差：模拟 decode（1 个 token）和 prefill（一批 token）
    x1 = torch.randn(1, in_f)                # decode：batch/seq = 1
    ref = lin(x1.float())
    got = ql(x1.float())
    print(f"\ndecode(1 token) 输出相对误差：{(ref-got).norm()/ref.norm():.4%}")

    x2 = torch.randn(512, in_f)              # prefill：512 token
    ref2 = lin(x2.float())
    got2 = ql(x2.float())
    print(f"prefill(512 token) 输出相对误差：{(ref2-got2).norm()/ref2.norm():.4%}")

    print("\n要点：")
    print(f"  - 权重压缩近 4x（{fp16_w_mb:.0f}MB -> {int4_w_mb:.1f}MB，含 scale 开销故略低于 4x）。")
    print("  - decode 访存受限：权重搬运量降到 1/4 -> 直接加速。A 保持 16-bit 因为激活量少不是瓶颈。")
    print("  - 真实 GPTQ/AWQ 在此基础上：AWQ 按激活重要性缩放权重、GPTQ 用 Hessian 逐列补偿误差。")
    print("  - 真实实现用融合 dequant kernel（边解包边乘），不会像这里先解回整块 fp16。")


if __name__ == "__main__":
    demo()
