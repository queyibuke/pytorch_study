import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.rand(1, 1, 28, 28)
layer = nn.Conv2d(1, 3,kernel_size=3, stride=2, padding= 0)
out = layer.forward(x)
print(f'out.shape: {out.shape}')

layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print(f'out.shape: {out.shape}')

out = F.interpolate(x, scale_factor=2, mode= 'nearest')
print(f'out.shape: {out.shape}')
out = F.interpolate(x, scale_factor=3, mode= 'nearest')
print(f'out.shape: {out.shape}')