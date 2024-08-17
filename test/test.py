import torch
import torch.version
print(torch.__version__)
print("gpu: ", torch.cuda.is_available())
print("hello world!")   
print(torch.version.cuda)