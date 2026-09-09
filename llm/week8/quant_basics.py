"""第 8 周（1/3）：量化的基本旋钮，用数据把概念钉死。

量化 = 把 fp16/fp32 张量用更少的比特（int8/int4）表示，靠一组 scale/zero_point 还原。
三个必须搞清的旋钮：
  1. 对称 vs 非对称：对称只有 scale（零点固定 0），非对称多一个 zero_point（能贴合非对称分布）。
  2. 粒度：per-tensor（整个张量一个 scale）vs per-channel / per-group（每行/每 g 个元素一个 scale）。
     粒度越细，误差越小，但 scale 的存储/计算开销越大。
  3. 位宽：int8 vs int4。int4 更省，但误差更大、且需要“打包”两个 int4 进一个 byte。

本文件只做 quantize -> dequantize 往返，测还原误差，直观感受每个旋钮的影响。
不依赖 GPU。运行：python quant_basics.py
"""

from __future__ import annotations

import torch


def quantize_symmetric(x: torch.Tensor, n_bits: int, dim: int | None = None):
    """对称量化：q = round(x / scale)，scale = max|x| / qmax。返回 (q_int, scale)。

    dim=None -> per-tensor（一个 scale）；dim=0 -> per-row/channel（每行一个 scale）。
    """
    qmax = 2 ** (n_bits - 1) - 1                 # int8 -> 127, int4 -> 7
    if dim is None:
        scale = x.abs().max() / qmax
    else:
        scale = x.abs().amax(dim=dim, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q, scale


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale


def quantize_asymmetric(x: torch.Tensor, n_bits: int, dim: int | None = None):
    """非对称量化：把 [min,max] 线性映射到 [0, 2^bits-1]，多一个 zero_point。"""
    qmax = 2 ** n_bits - 1                         # int8 -> 255（无符号）
    if dim is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin = x.amin(dim=dim, keepdim=True)
        xmax = x.amax(dim=dim, keepdim=True)
    scale = ((xmax - xmin) / qmax).clamp(min=1e-8)
    zero_point = torch.round(-xmin / scale)
    q = torch.round(x / scale + zero_point).clamp(0, qmax)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q - zero_point) * scale


def rel_err(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return (x - x_hat).norm().item() / x.norm().item()


def demo():
    torch.manual_seed(0)
    # 造一个“权重矩阵”，各行分布尺度不同 —— 这正是 per-channel 该赢的场景
    W = torch.randn(64, 256)
    W[0] *= 10.0          # 第 0 行数值范围远大于其它行
    W[1] *= 0.1

    print("=== 位宽：int8 vs int4（per-tensor 对称）===")
    for bits in (8, 4):
        q, s = quantize_symmetric(W, bits, dim=None)
        err = rel_err(W, dequantize_symmetric(q, s))
        print(f"  int{bits} per-tensor: 相对误差 {err:.4%}")

    print("\n=== 粒度：per-tensor vs per-channel（int8 对称）===")
    q, s = quantize_symmetric(W, 8, dim=None)
    print(f"  per-tensor : 相对误差 {rel_err(W, dequantize_symmetric(q, s)):.4%}  (1 个 scale)")
    q, s = quantize_symmetric(W, 8, dim=1)   # 每行一个 scale
    print(f"  per-channel: 相对误差 {rel_err(W, dequantize_symmetric(q, s)):.4%}  ({W.shape[0]} 个 scale)")
    print("  -> 各行尺度差异大时，per-channel 用少量额外 scale 换来明显更小的误差。")

    print("\n=== 对称 vs 非对称（int8, per-channel）===")
    # 造一个非对称分布（如 ReLU/GELU 后的激活，偏正）
    A = torch.randn(64, 256).abs() + 0.5
    q, s = quantize_symmetric(A, 8, dim=1)
    print(f"  对称  : 相对误差 {rel_err(A, dequantize_symmetric(q, s)):.4%}  (浪费了负半轴)")
    q, s, z = quantize_asymmetric(A, 8, dim=1)
    print(f"  非对称: 相对误差 {rel_err(A, dequantize_asymmetric(q, s, z)):.4%}  (零点平移，贴合分布)")

    print("\n=== group-wise：W4A16 的常见做法（每 group 个元素一个 scale）===")
    for g in (256, 128, 64, 32):
        Wg = W.reshape(-1, g)                 # 每 g 个元素一组
        q, s = quantize_symmetric(Wg, 4, dim=1)
        err = rel_err(Wg, dequantize_symmetric(q, s))
        n_scales = Wg.shape[0]
        print(f"  int4 group={g:>3}: 相对误差 {err:.4%}  ({n_scales} 个 scale)")
    print("  -> group 越小误差越小，但 scale 越多。W4A16 常用 group=128：精度/开销的甜点。")

    print("\n要点：")
    print("  - 权重量化误差可控（权重是静态的，可离线细调 scale/group）。")
    print("  - 激活分布偏斜时非对称更省；per-channel/group 是精度的主要来源。")
    print("  - int4 必须配 group-wise 才能压住误差 —— 这就是 GPTQ/AWQ 的起点。")


if __name__ == "__main__":
    demo()
