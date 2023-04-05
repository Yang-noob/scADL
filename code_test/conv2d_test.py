from torch import nn
import torch

conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding="same")
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding="same")
x = torch.rand((32, 1, 120, 120))
print(x.shape)
# print(x)
print('*'*60)
out1 = conv1(x)
print(out1.shape)
# print(out)
print('*'*60)
out2 = conv2(out1)
print(out2.shape)