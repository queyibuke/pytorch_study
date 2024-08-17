import torch
a = torch.rand(4,1,28,28)
print(a.shape)
#view/reshape改变形状
b = a.view(4,28*28)
print(b.shape)
c = a.reshape(4*28,28)
print(c.shape)
#squeeze/unsqueeze压缩/展开,新插入一个维度
d = a.unsqueeze(0).shape
print(d)
e = a.unsqueeze(-1).shape
print(e)
f = a.unsqueeze(-4).shape
print(f)

g = torch.rand(1, 32, 1, 1)
print(g.shape)
h = g.squeeze().shape
print(h)
i = g.squeeze(0).shape
print(i)
j = g.squeeze(-1).shape
print(j)
k = g.squeeze(1).shape
print(k)


#expand（不会复制数据）/repeat（会复制数据） 扩展/复制
l = torch.rand(4, 32, 14, 14)
print(l.shape)
m = g.expand(4, 32, 14, 14)#只能变1的维度
print(m.shape)
n = g.expand(-1, 32, -1, -4)#可以变多个维度
print(n.shape)
o = g.repeat(4, 32, 1, 1)#每个参数表示要重复的次数
print(o.shape)

#转置
p = torch.rand(4, 32)
print(p.shape)
q = p.t()#转置
print(q.shape)
#tranpose交换维度
r = torch.rand(4, 32, 14, 14)
s = r.transpose(1, 3)#交换第0和第1个维度
print(s.shape)

r1 = r.transpose(1,3).contiguous().view(4, 32 * 14 * 14).view(4, 32, 14, 14)
r2 = r.transpose(1,3).contiguous().view(4, 32 * 14 * 14).view(4, 14, 14 ,32).transpose(1, 3)
print('r1.shape', r1.shape, 'r2.shape', r2.shape)
print(torch.all(torch.eq(r, r1)))
print(torch.all(torch.eq(r,r2)))

#permute连续执行tranpose交换维度直至达到想要的维度
t = torch.rand(4, 3, 28, 28)
print(t.transpose(1, 3).shape)
u = torch.rand(4, 28, 28, 3)
print(u.transpose(1, 3).shape)

print(t.transpose(1, 3).transpose(1, 2).shape)
print(t.permute(0, 2, 3, 1).shape)
import torch
w = torch.rand(28, 28)
print(w.shape)
w = w.broadcast_to(3, 28, 28)
print(w.shape)

