import torch

x = torch.randn(3, 4)
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Device:", x.device)
print("Storage size:", x.storage().size())
