# 合并与分割 cat stack split chunk
import torch
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
c = torch.cat([a,b],dim=0)
print(c.shape)

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
b = torch.cat([a1, a2], dim=0)
print(b.shape)

a3 = torch.rand(4, 1, 32, 32)
b2 = torch.cat([a1, a3], dim=1)
print(b2.shape)

#stack
a4 = torch.rand(4, 3, 16, 32)
a5 = torch.rand(4, 3, 16, 32)
b3 = torch.stack([a4, a5], dim=2)
print(b3.shape)

a6 = torch.rand(32, 8)
a7 = torch.rand(32, 8)
b4 = torch.stack([a6, a7], dim=0)
print(b4.shape)

#split拆分
a8, a9 = b4.split([1, 31], dim = 1) #根据长度拆分
print(a8.shape, a9.shape)
a10, a11 = b4.split(1, dim = 0) 
print(a10.shape, a11.shape)
#chunk 
a12,a13 = b4.chunk(2, dim = 2)
print(a12.shape, a13.shape)

#矩阵乘matmaul
a14 = torch.rand(4, 3, 28, 64)
a15 = torch.rand(4, 3, 64, 32)
b5 = torch.matmul(a14, a15)
print(b5.shape)

a16 = torch.rand(4, 3, 28, 64)#中间使用broadcast机制
a17 = torch.rand(4, 1, 64, 32)
b6 = torch.matmul(a16, a17)
print(b6.shape)

#数学运算 floor() 向下取整ceil()向上取整 round()四舍五入 turnc()取整数部分 frac()取小数部分
c = torch.tensor(3.1415892)
print(c.floor(), c.ceil(), c.round(), c.trunc(), c.frac())
#梯度裁剪clamp
grad = torch.rand(2, 3) * 15
print(grad)
grad_max = grad.max()
print(grad_max)
grad_median = grad.median()
print(grad_median)
grad_clamp = grad.clamp(10)
print(grad_clamp)

