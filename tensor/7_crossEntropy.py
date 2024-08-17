import torch
import torch.nn.functional as F  
x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x@w.t()
print('logits: ', logits.shape)
print('logits: {}'.format(logits.shape))
print(f'logits: {logits.shape}')

pred = F.softmax(logits, dim=1)
print('pred: ', pred.shape)

pred_log = torch.log(pred)
print('pred_log: ', pred_log.shape)

cross_entropy = F.cross_entropy(logits, torch.tensor([3])) 
print(f'cross_entropy: {cross_entropy}\n')

cross_entropy1 = F.nll_loss(pred_log, torch.tensor([3]))
print(f'cross_entropy1: {cross_entropy1}\n')