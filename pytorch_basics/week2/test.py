import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(5, 10)
y = torch.add(x, 1)
print(x)
print(y)
model.eval()
with torch.inference_mode():
    output = model(x)
    print(output)
