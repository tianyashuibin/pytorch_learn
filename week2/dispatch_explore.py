"""
第 2 周:观察 dispatcher 与 InferenceMode。

对应阅读:
- c10/core/InferenceMode.h / .cpp
- c10/core/DispatchKey.h / DispatchKeySet.h

要建立的心智模型:一次 Tensor 操作 = Python -> dispatcher 按 DispatchKeySet
选一个 kernel -> 执行。requires_grad、inference_mode、设备、layout 都会改变
tensor 携带的 DispatchKey,从而改变最终走哪个 kernel。

运行:
    python dispatch_explore.py
"""
import torch

from _common import get_device


def dump_keys(name, t):
    print(f"  {name:<28} keys = {torch._C._dispatch_keys(t)}")


def show_inference_mode_effect(device):
    print("=" * 70)
    print("[1] requires_grad / no_grad / inference_mode 对 DispatchKeySet 的影响")
    print("=" * 70)

    plain = torch.randn(4, 4, device=device)
    dump_keys("普通 tensor", plain)

    leaf = torch.randn(4, 4, device=device, requires_grad=True)
    dump_keys("requires_grad=True", leaf)

    with torch.no_grad():
        y = leaf * 2
        dump_keys("no_grad 下的中间结果", y)

    with torch.inference_mode():
        z = torch.randn(4, 4, device=device)
        dump_keys("inference_mode 下新建", z)
    print("   -> inference_mode 会打上 InferenceMode key,跳过 autograd 的 view/version 追踪,")
    print("      比 no_grad 更省(no_grad 只是不建反向图,仍保留 autograd 元数据)。\n")


def show_dispatch_registrations():
    print("=" * 70)
    print("[2] 一个算子注册了哪些后端 kernel")
    print("=" * 70)
    for op in ("aten::linear", "aten::add.Tensor", "aten::relu"):
        print(f"\n--- {op} ---")
        try:
            print(torch._C._dispatch_dump(op))
        except Exception as e:
            print(f"   (无法 dump: {e})")


def show_meta_no_compute(device):
    print("=" * 70)
    print("[3] meta 设备:只算 shape/dtype,不做真实计算(理解 dispatcher 与 kernel 解耦)")
    print("=" * 70)
    a = torch.randn(128, 256, device="meta")
    b = torch.randn(256, 512, device="meta")
    out = a @ b
    print(f"  meta matmul 输出: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print("   -> 没有真正搬数据/算数,FakeTensor / shape 推断就建立在这套机制上(第 4-5 周会用到)。\n")


def main():
    device = get_device()
    print(f"设备={device}\n")
    show_inference_mode_effect(device)
    show_dispatch_registrations()
    show_meta_no_compute(device)


if __name__ == "__main__":
    main()
