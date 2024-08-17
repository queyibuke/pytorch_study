import torch
a = torch.rand(4,3,28,28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0,0,2,4])

#取前两张图片
print(a[:2].shape)
#取前两张图片的第一个通道
print(a[:2, :1,:,:].shape)
#取前两张图片的后面两个通道
print(a[:2, 1:,:,:].shape)
#取前两张图片的前两个通道
print(a[:2, -1:,:,:].shape)
#隔行采样
print(a[:, : , 0:28:2, 0:28:2].shape)
#对特定维度进行采样
print(a.index_select(0, torch.tensor([0, 2])).shape)
print(a.index_select(1, torch.tensor([1, 2])).shape)
#用...号寻找维度
print(a[...].shape)
print(a[..., 0].shape)
print(a[:,1,...].shape)
#掩码
x = torch.randn(3, 4)
print(x)
mask = x.ge(0.5)
print(mask)
print(torch.masked_select(x, mask))
print(torch.masked_select(x, mask).shape)