import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import Labels_Process as lp


class MyDataset(Dataset):
    def __init__(self, dataset,label, train=True, percentage:float = 0.8, transform=False):
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
        return sample, label


dataset1 = pd.read_hdf('G:/test_data/train_set.h5', key="dge")
label1 = pd.read_csv('G:/test_data/labels.csv', header=None)

label1_dict = lp.type_to_label_dict(label1)
lab = lp.convert_type_to_label(label1, label1_dict)
print(label1_dict)
print(lab)
# lab.to_csv("G:/test_data/label_converted.csv", header=None)

arr_data = np.array(dataset1)
arr_label = np.array(lab)
print(arr_data.shape)
print(arr_data[:,:3])

one_hot_matrix, num_class = lp.one_hot_matrix(lab)

train_dataset = MyDataset(arr_data, one_hot_matrix)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)

print("数据集长度：", train_dataset.__len__())

for data in train_loader:
    da, labels = data
    print(da.shape)
    print(da)
    print(labels.shape)
    print(labels)
    break
