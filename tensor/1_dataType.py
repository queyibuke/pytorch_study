import torch
#随机初始化
a = torch.randn(2, 3)
print(a)
print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
import torch
# dim 0维
b = torch.tensor(1.)
print(b, b.type(), b.shape, len(b.shape), b.size())
c = torch.tensor(1.3)
print(c)
# dim 1维
d = torch.tensor([1.1])
print(d, d.type(), d.shape, len(d.shape), d.size()) 
e = torch.tensor([1.2, 3.4])
print(e, e.type(), e.shape, len(e.shape), e.size()) 
f = torch.FloatTensor(1)
print(f, f.type(), f.shape, len(f.shape), f.size())
g = torch.FloatTensor(2)
print(g, g.type(), g.shape, len(g.shape), g.size()) 
import numpy as np
data = np.ones(2)
print(data, type(data), data.shape, len(data.shape))
h = torch.from_numpy(data)
print(h, h.type(), h.shape, len(h.shape), h.size()) 