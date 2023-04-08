import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        # nn.Flatten()
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        # print(x.shape)
        return x


class FCN(nn.Module):
    def __init__(self, ln1=1000, ln2=500, ln3=250, ln4=125,ln5=60, features=14063, total_number_types=5):
        super(FCN, self).__init__()

        # 定义神经网络的结构
        self.fc1 = nn.Linear(features, ln1)  # 输入层 -> 第1个隐藏层
        self.fc2 = nn.Linear(ln1, ln2)  # 第1个隐藏层 -> 第2个隐藏层
        self.fc3 = nn.Linear(ln2, ln3)  # 第2个隐藏层 -> 第3个隐藏层
        self.fc4 = nn.Linear(ln3, ln4)  # 第2个隐藏层 -> 第3个隐藏层
        self.fc5 = nn.Linear(ln4, ln5)  # 第2个隐藏层 -> 第3个隐藏层
        self.fc6 = nn.Linear(ln5, total_number_types)  # 输出层

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # 使用ReLU激活函数
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class FCN_1(nn.Module):
    def __init__(self, ln1=100, ln2=50, ln3=25, features=14063, total_number_types=5):
        super(FCN_1, self).__init__()

        # 定义神经网络的结构
        self.fc1 = nn.Linear(features, ln1)  # 输入层 -> 第1个隐藏层
        self.fc2 = nn.Linear(ln1, ln2)  # 第1个隐藏层 -> 第2个隐藏层
        self.fc3 = nn.Linear(ln2, ln3)  # 第2个隐藏层 -> 第3个隐藏层
        self.fc4 = nn.Linear(ln3, total_number_types)  # 输出层

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # 使用ReLU激活函数
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
