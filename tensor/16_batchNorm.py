import torch
import torch.nn
x = torch.randn(100 ,16) + 0.5
layer = torch.nn.BatchNorm1d(16)
print(f'layer.running_mean: {layer.running_mean}, layer.running_var: {layer.running_var}\n')

out = layer(x)
print(f'layer.running_mean: {layer.running_mean}, layer.running_var: {layer.running_var}\n')

