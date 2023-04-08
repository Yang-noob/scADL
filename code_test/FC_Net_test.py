import torch
from torch import nn
import numpy as np

tensor = torch.Tensor([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
print(tensor.shape)
print(tensor)
# print(tensor[1][1])
# print('*'*20)
# tensor = tensor.transpose(0,1)
# print(tensor.shape)
# print(tensor)
# print(tensor[1][1])
fc1 = nn.Linear(4, 2)
print("权重矩阵：",fc1.weight)
print("偏置：",fc1.bias)
z1 = fc1(tensor)
print(z1.shape)
print(z1)

# shuzu = np.array(tensor)
# print(shuzu.shape)
# print(shuzu)