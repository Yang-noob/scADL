import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
tensor = torch.Tensor([[1, 7, 12, 4],
                       [4, 5, 9, 6],
                       [11, 3, 10, 2]])
print(tensor.shape)
print(tensor)

p = torch.argmax(tensor, dim=1)
print(p.shape)
print(p)
print("*"*30)
q = F.softmax(tensor, dim=1)
print(q.shape)
print(q)
