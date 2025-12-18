import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
v = torch.tensor([1.0, 1.0])
y.backward(v)
print("VJP:", x.grad)
