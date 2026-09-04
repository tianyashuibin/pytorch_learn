"""
对比 eager 模式与 torch.compile 两条执行路线。

- Eager : Python -> dispatcher -> 逐个 aten 算子
- Compile: Dynamo -> FX graph -> AOTAutograd(分解成 prim) -> Inductor -> C++/Triton kernel(融合)

运行:
    python compile_vs_eager.py
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)

    def forward(self, x):
        # linear + relu,故意留出可融合的空间
        return torch.nn.functional.relu(self.lin(x))


# ---------------------------------------------------------------------------
# 1) EAGER:拦截并打印真正被 dispatcher 调度的 aten 算子
# ---------------------------------------------------------------------------
class TraceDispatch(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def show_eager(model, x):
    print("=" * 70)
    print("[1] EAGER 模式:dispatcher 实际调度的 aten 算子")
    print("=" * 70)
    tracer = TraceDispatch()
    with tracer:
        y = model(x)
    for op in tracer.ops:
        print("   ", op)
    print("   -> 逐个算子执行,无跨算子融合(linear 内部走 addmm 是库级融合)\n")
    return y


# ---------------------------------------------------------------------------
# 2) COMPILE:用自定义后端截获 Dynamo 抓到的 FX 图
# ---------------------------------------------------------------------------
def show_fx_graph(model, x):
    print("=" * 70)
    print("[2] torch.compile:Dynamo 捕获的 FX 图(图级 IR)")
    print("=" * 70)

    captured = {}

    def inspect_backend(gm: torch.fx.GraphModule, example_inputs):
        # gm 就是 Dynamo trace 出来的图;打印后仍交回 eager 执行
        captured["graph"] = gm
        gm.graph.print_tabular()
        return gm.forward

    compiled = torch.compile(model, backend=inspect_backend, fullgraph=True)
    compiled(x)
    print("   -> 这是融合/优化前的图,节点仍对应上游 API\n")


# ---------------------------------------------------------------------------
# 3) COMPILE:用 Inductor 后端,dump 出融合后的实际代码
# ---------------------------------------------------------------------------
def show_inductor_code(model, x):
    print("=" * 70)
    print("[3] torch.compile + Inductor:融合后生成的 kernel 代码")
    print("=" * 70)
    print("   (设置 TORCH_LOGS=output_code 时会打印生成的 C++/Triton 源码)")

    import torch._dynamo as dynamo
    dynamo.reset()

    compiled = torch.compile(model, backend="inductor", fullgraph=True)
    y = compiled(x)  # 首次调用触发编译

    print("   编译完成,输出形状:", tuple(y.shape))
    print("   -> CPU 上 Inductor 生成 C++/OpenMP,把 addmm+relu 等尽量融进同一循环\n")
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MyModule().eval()
    x = torch.randn(5, 3)

    with torch.no_grad():
        y_eager = show_eager(model, x)
        show_fx_graph(model, x)
        y_compiled = show_inductor_code(model, x)

        # 数值一致性:两条路线结果应当相同
        print("=" * 70)
        print("[4] 数值一致性检查")
        print("=" * 70)
        print("   allclose(eager, compiled):",
              torch.allclose(y_eager, y_compiled, atol=1e-6))
