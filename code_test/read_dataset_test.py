import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import Labels_Process as lp
from utils import Datasets_Process as dp
import torch
from models import FCN_1
from torch import nn


class MyDataset(Dataset):
    def __init__(self, dataset, label, train=True, percentage: float = 0.8, transform=False):
        self.dataset = dataset
        self.label = label
        self.train = train
        self.percentage = percentage
        self.transform = transform
        self.dataset_len = dataset.shape[1]

        if 0 <= self.percentage <= 1:
            if self.train:
                self.sample_num = list(range(int(self.dataset_len * percentage)))  # 用前800个样本作为训练集
            else:
                self.sample_num = list(
                    range(int(self.dataset_len * (1 - percentage)), self.dataset_len))  # 用后200个样本作为测试集
        else:
            raise ValueError("percentage范围: [0,1]")

    def __len__(self):
        return len(self.sample_num)

    def __getitem__(self, index):
        sample = self.dataset[:, self.sample_num[index]]
        label = self.label[self.sample_num[index]]

        if self.transform:
            n = int(np.sqrt(len(sample)))  # 计算方形矩阵边长
            w = n
            h = n
            if w * h == len(sample):
                sample = sample.reshape(w, h)  # 转为方形矩阵
            else:
                while w * (h + 1) < len(sample):
                    h += 1
                sample = np.pad(sample, (0, w * (h + 1) - len(sample)), mode='constant', constant_values=0)  # 填充0
                sample = sample.reshape(w, h + 1)  # 转为方形矩阵
                sample = sample.reshape(1, *sample.shape)
        return torch.Tensor(sample), torch.Tensor(label)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataset1 = pd.read_hdf('G:/actinn_dataset/tma_10x_cleaned.h5', key="dge")
label1 = pd.read_csv('G:/actinn_dataset/tma_10x_cleaned_label.txt', header=None, sep='\t')

dataset1 = dataset1.iloc[:, :4000]
train_set = dp.capitalize_genes_name(dataset1)
# print(train_set)
train_set = dp.filt_duplicate_rows(train_set)
# print(train_set)
train_set = dp.normalize(train_set)
# print(train_set)


label1_dict = lp.type_to_label_dict(label1)
lab = lp.convert_type_to_label(label1, label1_dict, return_labels_only=True)
print(label1_dict)
# print(len(lab))
# lab.to_csv("G:/test_data/label_converted.csv", header=None)

arr_data = np.array(train_set)
arr_label = np.array(lab)
features = arr_data.shape[0]
# print(arr_data.shape)
# print(arr_data[:,:3])

one_hot_matrix, num_class = lp.one_hot_matrix(lab)

train_dataset = MyDataset(arr_data, one_hot_matrix, transform=False)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)

print("数据集长度：", train_dataset.__len__())

model = FCN_1(features=features, total_number_types=num_class)

model = model.to(device)

# 定义损失函数、优化器
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
epoch = 20

for i in range(epoch):
    print("----------------第 {} 轮训练----------------".format(i + 1))
    # 训练步骤开始
    model.train()
    for data in train_loader:
        inputs, labels = data
        # inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        # inputs = inputs.permute(1, 0, 2, 3)
        # print(inputs.shape)
        # print(labels.shape)
        inputs = inputs.to(device)  # GPU           #imgs = imgs.to(device)
        labels = labels.to(device)  # GPU     #targets = targets.to(device)
        outputs = model.forward(inputs)  # 让输入通过层层特征提取网络（前向传播）

        print(outputs)
        print(labels)
        # 特征提取：输入的像素点矩阵x * 权重参数矩阵w的过程，像素矩阵x的行列（形状）会发生变化，权重矩阵w的元素值将来会被不断更新。
        loss = criterion(outputs, labels)  # 计算在dataloader中一次训练的损失（每一轮输出的得分和真实值作比较的过程）
        print(loss.item())
        break
    break
