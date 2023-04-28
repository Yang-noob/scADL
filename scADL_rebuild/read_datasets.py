import numpy as np
import torch
from torch.utils.data import Dataset


# class MyDataset(Dataset):
#     def __init__(self, dataset,label, train=True, percentage:float = 0.8, transform=False):
#         self.dataset = dataset
#         self.label = label
#         self.if_train = train
#         self.percentage = percentage
#         self.transform = transform
#         self.dataset_len = dataset.shape[1]
#
#         if 0 <= self.percentage <= 1:
#             if self.if_train:
#                 self.sample_num = list(range(int(self.dataset_len * percentage)))  # 用前800个样本作为训练集
#             else:
#                 self.sample_num = list(
#                     range(int(self.dataset_len * (1 - percentage)), self.dataset_len))  # 用后200个样本作为测试集
#         else:
#             raise ValueError("percentage范围: [0,1]")
#
#     def __len__(self):
#         return len(self.sample_num)
#
#     def __getitem__(self, index):
#         sample = self.dataset[:, self.sample_num[index]]
#         label = self.label[self.sample_num[index]]
#
#         if self.transform:
#             n = int(np.sqrt(len(sample)))  # 计算方形矩阵边长
#             w = n
#             h = n
#             if w * h == len(sample):
#                 sample = sample.reshape(w, h)  # 转为方形矩阵
#             else:
#                 while w * (h + 1) < len(sample):
#                     h += 1
#                 sample = np.pad(sample, (0, w * (h + 1) - len(sample)), mode='constant', constant_values=0)  # 填充0
#                 sample = sample.reshape(w, h + 1)  # 转为方形矩阵
#                 sample = sample.reshape(1, *sample.shape)
#         return torch.Tensor(sample), torch.Tensor(label)


class CNN_Dataset(Dataset):
    def __init__(self, dataset,label, train=True, percentage:float = 0.8, transform=False):
        self.dataset = dataset
        self.label = label
        self.if_train = train
        self.percentage = percentage
        self.transform = transform
        self.dataset_len = dataset.shape[1]

        if 0 <= self.percentage <= 1:
            if self.if_train:
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
                sample = sample.reshape(1, w, h)  # 转为方形矩阵

            else:
                while w * (h + 1) < len(sample):
                    h += 1
                sample = np.pad(sample, (0, w * (h + 1) - len(sample)), mode='constant', constant_values=0)  # 填充0
                sample = sample.reshape(w, h + 1)  # 转为方形矩阵
                sample = sample.reshape(1, *sample.shape)
        return torch.Tensor(sample), torch.Tensor(label)
