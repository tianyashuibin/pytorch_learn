import torch
from torch import fx

t = torch.randn(10, 100)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


# 自定义 backend：Dynamo 会把捕获到的 FX 计算图传进来，
# 我们把它打印出来，然后原样返回让它继续正常执行。
def print_graph_backend(gm: fx.GraphModule, example_inputs):
    print("=" * 60)
    print("捕获到的 FX Graph:")
    gm.graph.print_tabular()      # 表格形式，清晰
    print("-" * 60)
    print("对应的 Python 代码:")
    print(gm.code)                # 转成可读的 Python 代码
    print("=" * 60)
    return gm.forward             # 返回可调用对象，继续正常运行


mod1 = MyModule()
mod1_compiled = torch.compile(mod1, backend=print_graph_backend)
print("mod1 输出:")
print(mod1_compiled(torch.randn(3, 3)))
