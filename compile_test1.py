import torch

t = torch.randn(10, 100)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod1 = MyModule()
mod1.compile()
print(mod1(torch.randn(3, 3)))

mod2 = MyModule()
mod2 = torch.compile(mod2)
print(mod2(torch.randn(3, 3)))