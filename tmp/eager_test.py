import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)

    def forward(self, x):
        tmp = self.lin(x)
        result = torch.nn.functional.relu(tmp)
        return result

modl = MyModule()
x = torch.randn(5, 3)
y = modl(x)
print(y)


# 更简单:用 __torch_dispatch__ 拦截
from torch.utils._python_dispatch import TorchDispatchMode

class TraceMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("ATen op:", func)          # 看到 aten::linear、aten::relu 等
        return func(*args, **(kwargs or {}))

with TraceMode():
    y = modl(x)

# 打印算子耗时
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    modl(x)
print(prof.key_averages().table(sort_by="cpu_time_total"))



print(torch._C._dispatch_dump("aten::linear"))   # 该算子注册了哪些后端 kernel
print(torch._C._dispatch_keys(x))                  # tensor 带哪些 DispatchKey
