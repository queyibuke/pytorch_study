import numpy as np
import torch

a = np.array([1,2,3])
a_t = torch.from_numpy(a)
print(a_t, a_t.shape, a_t.type())

b = torch.tensor([2.,3.2])
print(b, b.shape, b.type())

#小写tensor接收现成数据，大写Tensor接收shape
c = torch.tensor([2.,3.2])
print(c, c.shape, c.type())

d = torch.FloatTensor(2, 3)
print(d, d.shape, d.type())

e = torch.FloatTensor([[2., 3.2], [1., 22.3]])
print(e, e.shape, e.type())

#申请内存空间
f = torch.empty(2, 3)
print(f, f.shape, f.type())
#均值分布
g = torch.rand(3, 3)
print(g, g.shape, g.type())
 
h = torch.randint(1, 10, (3, 4))
print(h, h.shape, h.type())

#正态分布
i = torch.randn(3, 2)
print(i, i.shape, i.type())
import torch
j = torch.normal(mean = torch.full((10,), 0, dtype=torch.float32), std = torch.arange(1.0, 0.0, -0.1, dtype = torch.float32))
print(j, j.shape, j.type())

#全部赋值
k = torch.full((2, 3), 10)
print(k, k.shape, k.type())
l = torch.full([], 7)
print(l, l.shape, l.type())
m = torch.full([1], 7)
print(m, m.shape, m.type())
#生成等差数列
n = torch.arange(1, 10)
print(n, n.shape, n.type())
#生成线性空间
o = torch.linspace(1, 20, steps=5)#切5份
print(o, o.shape, o.type())
#生成log线性空间
p = torch.logspace(1, 20, steps=5)#切5份
print(p, p.shape, p.type())
#生成全1
q = torch.ones(2, 3)
print(q, q.shape, q.type())

#生成全0
r = 2 * torch.zeros(3, 4)
print(r, r.shape, r.type())

#生成eyes
s = torch.eye(3, 4)
print(s, s.shape, s.type())

#生成随机打散
t = torch.randperm(10)
print(t, t.shape, t.type())
