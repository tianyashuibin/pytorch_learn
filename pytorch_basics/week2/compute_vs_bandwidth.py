"""
第 2 周实践 3:判断算子偏计算受限还是带宽受限。

方法(roofline 直觉):
    算术强度 AI = FLOPs / 访存字节
    - matmul (M,K)x(K,N):FLOPs ≈ 2*M*N*K,访存 ≈ (M*K + K*N + M*N)*dtype_bytes
      K 越大 AI 越高,越偏计算受限。
    - elementwise / norm / activation:FLOPs 与访存同阶,AI≈常数且很低,几乎必然带宽受限。

把 AI 和硬件的 峰值FLOPS/峰值带宽 之比(ridge point)比较:
    AI < ridge_point  -> 带宽受限
    AI > ridge_point  -> 计算受限

这里不追求精确,只建立"看一眼算子就能猜受限类型"的手感,再用 Nsight/profiler 验证。

运行:
    python compute_vs_bandwidth.py
"""


def matmul_intensity(M, K, N, dtype_bytes=4):
    flops = 2.0 * M * N * K
    mem = (M * K + K * N + M * N) * dtype_bytes
    return flops, mem, flops / mem


def elementwise_intensity(n, flops_per_elem=1, tensors_touched=2, dtype_bytes=4):
    flops = flops_per_elem * n
    mem = tensors_touched * n * dtype_bytes
    return flops, mem, flops / mem


def classify(ai, ridge):
    return "计算受限" if ai > ridge else "带宽受限"


def main():
    # 一个示意性的硬件 ridge point(峰值FLOPS / 峰值带宽)。
    # 例:~100 TFLOPS fp16 / ~2 TB/s ≈ 50 FLOP/Byte。真实值请按你的 GPU 填。
    ridge = 50.0
    print(f"假设 ridge point = {ridge:.0f} FLOP/Byte(按实际 GPU 调整)\n")

    print(f"{'算子':<34}{'FLOPs':>14}{'Bytes':>14}{'AI':>9}{'判断':>10}")
    print("-" * 82)

    cases = [
        ("大 matmul 4096x4096x4096", *matmul_intensity(4096, 4096, 4096)),
        ("decode 小 GEMM 1x4096x4096", *matmul_intensity(1, 4096, 4096)),
        ("qkv proj 256x512x1536", *matmul_intensity(256, 512, 1536)),
        ("relu (4M 元素)", *elementwise_intensity(4_000_000, 1, 2)),
        ("layernorm (4M 元素)", *elementwise_intensity(4_000_000, 8, 2)),
        ("add 残差 (4M 元素)", *elementwise_intensity(4_000_000, 1, 3)),
    ]
    for name, flops, mem, ai in cases:
        print(f"{name:<34}{flops:>14.3e}{mem:>14.3e}{ai:>9.2f}{classify(ai, ridge):>10}")

    print("\n观察:")
    print("  - 大 matmul K 大 -> AI 高 -> 计算受限;decode 的 M=1 小 GEMM AI 极低 -> 带宽/launch 受限。")
    print("  - relu/layernorm/add 这类 elementwise 恒为带宽受限,是算子融合的首要目标(第 6 周)。")
    print("  - 这解释了为什么 LLM decode 阶段更吃访存带宽,而 prefill 更吃算力(第 8 周主题)。")


if __name__ == "__main__":
    main()
