import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.rand(1, 1, 28, 28)

layer = nn.Conv2d(1, 3,kernel_size=3, stride=1, padding= 0)
out = layer.forward(x)
print(f'out.shape: {out.shape}')

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding= 1)
out = layer.forward(x)
print(f'out.shape: {out.shape}')

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding= 1)
out = layer.forward(x)
print(f'out.shape: {out.shape}')

x = torch.randn(1, 3, 28, 28)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
out = F.conv2d(x, w, b, stride=1, padding=1)
print(f'out.shape: {out.shape}')

out = F.conv2d(x, w, stride=2, padding=2)
print(f'out.shape: {out.shape}')

x = torch.randn(1, 1, 28, 28)
out= layer(x)
print(f'out.shape: {out.shape}')
print(f'layer.weight: {layer.weight}')
print(f'layer.weight.shape: {layer.weight.shape}, layer.bias.shape: {layer.bias.shape}')
