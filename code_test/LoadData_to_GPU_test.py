# from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import torch


def normalize(*datasets, scale_factor=10000, low_range=1, high_range=99):
    datasets_list = list(datasets)
    for i in range(len(datasets_list)):
        datasets_list[i] = torch.Tensor(datasets_list[i])
        datasets_list[i] = datasets_list[i] / torch.sum(datasets_list[i], dim=0, keepdim=True) * scale_factor
        datasets_list[i] = torch.log2(datasets_list[i] + 1)

        # Filter genes by expression level
        expr = torch.sum(datasets_list[i], dim=1)
        expr_low = torch.kthvalue(expr, int(len(expr) * low_range / 100)).values
        expr_high = torch.kthvalue(expr, int(len(expr) * high_range / 100)).values
        expr_mask = (expr >= expr_low) & (expr <= expr_high)
        datasets_list[i] = datasets_list[i][expr_mask, :]

        # Filter genes by coefficient of variation
        mean = torch.mean(datasets_list[i], dim=1, keepdim=True)
        std = torch.std(datasets_list[i], dim=1, keepdim=True)
        cv = std / (mean + 1e-10)
        # cv = torch.squeeze(cv)
        cv = cv.squeeze(-1)
        cv_low = torch.kthvalue(cv, int(len(cv) * low_range / 100)).values
        cv_high = torch.kthvalue(cv,int(len(cv) * high_range / 100)).values
        cv_mask = (cv >= cv_low) & (cv <= cv_high)
        datasets_list[i] = datasets_list[i][cv_mask, :]
    if len(datasets_list) == 1:
        return datasets_list[0]
    return datasets_list

#
# data = np.random.rand(20000, 10000)
#
# tensor = torch.from_numpy(data).float().cuda()
#


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataset1 = pd.read_hdf('G:/actinn_dataset/tma_10x_cleaned.h5', key="dge")
# label1 = pd.read_csv('G:/actinn_dataset/tma_10x_cleaned_label.txt', header=None, sep='\t')
dataset1 = dataset1.iloc[:,:9000]
dataset1 = np.array(dataset1.values, dtype=np.float32)

tensor = torch.from_numpy(dataset1).float().to(device)

# 假设这里使用的是一个名为normalize的预处理函数
for i in range(10000):
    print(f"第 {i + 1} 次计算:")
    a = normalize(tensor)
    # tensor = tensor.to('GPU')
    print(a)
#
# class MyDataset(Dataset):
#     def __init__(self, dataset, label, train=True, percentage:float = 0.8, transform=False):
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
#         return torch.Tensor(sample)  # , torch.Tensor(label)
#
#
# # Define normalization function
# def normalize(x):
#     mean = torch.mean(x, dim=0)
#     std = torch.std(x, dim=0)
#     return (x - mean) / std
#
#
# # Load data from HDF5 file
# dataset1 = pd.read_hdf('G:/test_data/train_set.h5', key="dge")
# label1 = pd.read_csv('G:/test_data/train_label.txt', header=None, sep='\t')
#
#
# dataset1 = np.array(dataset1.values, dtype=np.float32)
#
# # Convert data to PyTorch tensors and move to GPU
# # train_set = torch.from_numpy(dataset1.values).float().to('cuda')
# train_set = torch.tensor(dataset1)
# train_set = train_set.to('cuda:0')
#
# # Normalize data
# train_set = normalize(train_set)
#
# # Create TensorDataset from PyTorch tensor
# dataset = TensorDataset(train_set)
#
# # Create DataLoader and move to GPU
# batch_size = 256
# data_set = MyDataset(dataset, percentage=1)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
#
# # Define model architecture
# model = torch.nn.Sequential(
#     torch.nn.Linear(100, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 25),
#     torch.nn.ReLU(),
#     torch.nn.Linear(25, 6),
# )
#
# # Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss().to('cuda')
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
#
# # Train model
# for epoch in range(10):
#     for data in dataloader:
#         # Forward pass
#         y_pred = model(data)
#
#         # Compute loss
#         # loss = criterion(y_pred, y_batch)
#
#         # Backward pass and optimization
#         # optimizer.zero_grad()
#         # # loss.backward()
#         # optimizer.step()
#
#     # Print progress
#     # print(f'Epoch {epoch + 1}, loss = {loss.item():.4f}')

# import h5py
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
#
# # Define normalization function


# def normalize(x):
#     mean = torch.mean(x, dim=0)
#     std = torch.std(x, dim=0)
#     return (x - mean) / std


#
#
# # Load data from HDF5 file
# with h5py.File('data.h5', 'r') as f:
#     x_train = f['x_train'][()]
#     y_train = f['y_train'][()]
#
# # Convert data to PyTorch tensors and move to GPU
# x_train = torch.from_numpy(x_train).float().to('cuda')
# y_train = torch.from_numpy(y_train).long().to('cuda')
#
# # Normalize data
# x_train = normalize(x_train)
#
# # Create TensorDataset from PyTorch tensor
# dataset = TensorDataset(x_train, y_train)
#
# # Create DataLoader and move to GPU
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#
# # Define model architecture
# model = torch.nn.Sequential(
#     torch.nn.Linear(10, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 2),
# )
#
# # Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # Train model
# for epoch in range(10):
#     for x_batch, y_batch in dataloader:
#         # Forward pass
#         y_pred = model(x_batch)
#
#         # Compute loss
#         loss = criterion(y_pred, y_batch)
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Print progress
#     print(f'Epoch {epoch + 1}, loss = {loss.item():.4f}')



