import torch.nn as nn
import torch.optim as optim
from read_datasets import MyDataset
from torch.utils.data import DataLoader
import torch
from Net import CNN_Net
from Net import FC_Net_1
import pandas as pd
from utils import Labels_Process as lp
import numpy as np


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

data_path = "G:/test_data/train_set.h5"
label_path = "G:/test_data/train_label.txt"

dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')
# print(dataset1.shape)
# print(label1.shape)
# print(dataset1)
# print(label1)

label1_dict = lp.type_to_label_dict(label1)
lab = lp.convert_type_to_label(label1, label1_dict)
# print(label1_dict)
# print(lab)

arr_data = np.array(dataset1)
arr_label = np.array(lab)
# print(arr_data.shape)
# print(arr_data)

one_hot_matrix, num_class = lp.one_hot_matrix(lab)
features = arr_data.shape[0]

train_dataset = MyDataset(arr_data, one_hot_matrix, percentage=0.6, train=True, transform=False)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

test_dataset = MyDataset(arr_data, one_hot_matrix, percentage=0.4, train=False, transform=False)
test_loader = DataLoader(test_dataset, batch_size=124, shuffle=True)

# length 长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练集长度: {}".format(train_dataset_size))
print("测试集长度: {}".format(test_dataset_size))

# 初始化神经网络，损失函数和优化器
net = FC_Net_1(features=features, total_number_types=num_class)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.8)

# 设置神经网络的一些参数
total_train_step = 0  # 训练的次数
total_test_step = 0  # 测试的次数
epoch = 200  # 训练的轮数


for i in range(epoch):
    print("--------------第 {} 轮训练--------------".format(i + 1))
    # 训练步骤开始
    net.train()
    for data in train_loader:
        inputs, labels = data
        # inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        # inputs = inputs.permute(1, 0, 2, 3)
        # print(inputs.shape)
        # print(labels.shape)
        inputs = inputs.to(device)  # GPU           #imgs = imgs.to(device)
        labels = labels.to(device)  # GPU     #targets = targets.to(device)
        outputs = net.forward(inputs)  # 让输入通过层层特征提取网络（前向传播）
        # 特征提取：输入的像素点矩阵x * 权重参数矩阵w的过程，像素矩阵x的行列（形状）会发生变化，权重矩阵w的元素值将来会被不断更新。
        loss = criterion(outputs, labels)  # 计算在dataloader中一次训练的损失（每一轮输出的得分和真实值作比较的过程）
        # 优化器优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，求解损失函数梯度
        optimizer.step()  # 更新权重参数

        total_train_step = total_train_step + 1
        if total_train_step % 40 == 0:
            print("第 {} 次训练的Loss：{}".format(total_train_step, loss.item()))

    # 验证步骤开始（验证训练结果怎么样）
    net.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    # 无梯度的目的是:正处在验证阶段,所以不用对梯度进行调整,无需优化神经网络参数,也可以节省内存
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
            # inputs = inputs.permute(1, 0, 2, 3)
            inputs = inputs.to(device)  # GPU          #imgs = imgs.to(device)
            labels = labels.to(device)  # GPU    #targets = targets.to(device)
            outputs = net.forward(inputs)
            # 特征提取网络经过该轮训练，神经网络参数w被更新，将输入图片放入该网络后的到的得分值被记录下来
            loss = criterion(outputs, labels)  # 计算损失
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == labels.argmax(1)).sum()  # 正确率的分子
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_test_accuracy / test_dataset_size))
    total_test_step = total_test_step + 1

    # torch.save(net, "./checkpoints/ywh_{}.pth".format(i+1))
    # print("模型已保存")
