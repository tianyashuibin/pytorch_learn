import torch
import time

torch._logging.set_logs(graph_code=True)

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo1 = torch.compile(foo)
params = (torch.randn(3, 3), torch.randn(3, 3))
result = opt_foo1(*params)
print(result)


@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


print(opt_foo2(torch.randn(3, 3), torch.randn(3, 3)))

print("--------------------------------")

def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u


# Inductor 后端在 macOS 上会段错误，本机用 aot_eager 才能跑通
opt_foo3 = torch.compile(foo3, backend="aot_eager")

# torch.jit：老一代的图编译（TorchScript）。script 会把函数解析成 IR
jit_foo3 = torch.jit.script(foo3)


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. On CPU there are no CUDA events, and computation is synchronous,
# so a plain wall-clock timer with time.perf_counter() is accurate enough.
def timed(fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return result, end - start


inp = torch.randn(4096, 4096)

# Warm up: the first call to a compiled fn triggers compilation (slow),
# so run it a few times before timing to exclude compile/optimization time.
# TorchScript 也需要预热（前几次运行才会触发 JIT 优化）。
for _ in range(3):
    timed(lambda: opt_foo3(inp))
    timed(lambda: jit_foo3(inp))

print("compile:", timed(lambda: opt_foo3(inp))[1])
print("jit:", timed(lambda: jit_foo3(inp))[1])
print("eager:", timed(lambda: foo3(inp))[1])