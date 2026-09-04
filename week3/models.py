"""
第 3 周:一组"故意制造 graph break"的模型。

Dynamo 能把纯 tensor 计算 trace 成一张 FX 图;一旦遇到它无法在图里表达的东西,
就会 graph break —— 把当前图交出去执行,回到 Python 解释器跑那段代码,再从后面
重新开始 trace 下一张图。常见诱因:
- 数据相关控制流(if 依赖 tensor 的值 -> 需要 .item(),强制求值)
- Python side effect(print、修改全局变量、append 到外部 list)
- Dynamo 不支持 / 未建模的调用

每个模型都返回同样 shape 的结果,只是"可 trace 程度"不同。
"""
import torch
import torch.nn as nn


class Clean(nn.Module):
    """纯 tensor 运算,应能整图捕获(fullgraph=True 不报错)。"""

    def __init__(self, dim=256):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


class DataDependentControlFlow(nn.Module):
    """if 依赖 tensor 具体数值 -> 触发 .item() 同步 + graph break。"""

    def __init__(self, dim=256):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.l1(x)
        if h.sum() > 0:          # 需要真实数值才能决定分支
            return torch.relu(h)
        return torch.tanh(h)


class PythonSideEffect(nn.Module):
    """print 是 side effect,Dynamo 无法放进图 -> graph break。"""

    def __init__(self, dim=256):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.l1(x)
        print("  [side effect] mean =", float(h.mean()))  # 打断
        return torch.relu(h)


class UnsupportedCall(nn.Module):
    """调用 Dynamo 未建模的东西(这里用 numpy 往返)-> graph break。"""

    def __init__(self, dim=256):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.l1(x)
        arr = h.detach().cpu().numpy()   # 出图到 numpy
        arr = arr * 2.0
        return torch.relu(torch.from_numpy(arr).to(x.device))


REGISTRY = {
    "clean": Clean,
    "control_flow": DataDependentControlFlow,
    "side_effect": PythonSideEffect,
    "unsupported": UnsupportedCall,
}


def build(name, device="cpu"):
    model = REGISTRY[name]().to(device).eval()
    x = torch.randn(8, 256, device=device)
    return model, x
