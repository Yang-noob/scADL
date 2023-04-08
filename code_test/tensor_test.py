import torch

weight = torch.Tensor([[0.3512, 0.3078, 0.4066, -0.2437],
                       [-0.0095, -0.2684, 0.4967, -0.1431],
                       [-0.4796, 0.2278, 0.2877, 0.3889],
                       [-0.1529, 0.0977, -0.0743, -0.1262]])

bias = torch.Tensor([-0.1941, -0.4033, -0.2192,  0.1615])
data = torch.Tensor([[1., 2., 3., 4.],
                     [4., 5., 6., 7.],
                     [8., 9., 10., 11.]])
# print(bias.shape)
# data = data.transpose(0,1)
# bias = bias.unsqueeze(1)
# print(bias1.shape)
# print(data)
# z1 = torch.add(torch.matmul(weight, data), bias)
# z1 = z1.transpose(0,1)
# print(z1)


y = torch.matmul(data, weight.t()) + bias
print(y)
