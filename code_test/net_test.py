import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        #nn.Flatten()
        x = x.view(-1, 32 * 8 * 8)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.fc3(x)
        print(x.shape)
        x = self.relu(x)
        out = self.fc4(x)
        print(out.shape)
        return out


net = Net()
# # 生成随机数值为1~10之间，大小为119×119的一维张量
# # arr = np.random.randint(low=1, high=11, size=(119, 119))
# # arr = float(arr)
# # x = torch.from_numpy(arr)
# 生成随机张量
x = torch.rand((32, 1, 119, 119))
# 将张量的值映射到[1, 10]之间
print(x.shape)
inputs = x * 9 + 1
# print(inputs.shape)
# inputs = inputs.unsqueeze(0).unsqueeze(0)
print('*'*20 + "forward函数内部" + '*'*20)
out = net.forward(inputs)
print("输出：")
print(out.shape)
