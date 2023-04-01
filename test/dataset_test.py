import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dataset_path, train=True, transform=None):
        self.dataset_path = dataset_path + "/labels.csv"
        self.train = train
        self.transform = transform
        # 从 CSV 文件中读取标签
        self.labels = pd.read_csv(self.dataset_path)['cell_type']
        if self.train:
            self.samples = list(range(800))  # 用前800个样本作为训练集
        else:
            self.samples = list(range(800, 1000))  # 用后200个样本作为测试集

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 读取样本
        sample_path = "../data/train_set_10x/npy/" + str(self.samples[idx]) + '.npy'
        sample = np.load(sample_path)
        # labeldict = {'B cell', 'T cell', 'NK cell', 'Granulocyte', 'Monocyte', 'Erythrocyte'}
        labelmap = {'B cell': 0, 'T cell': 1, 'NK cell': 2, 'Granulocyte': 3, 'Monocyte': 4, 'Erythrocyte': 5}
        label_dict = torch.tensor([0, 1, 2, 3, 4, 5])
        label_one_hot = torch.eye(6)[label_dict]  # one-hot编码
        # label_encoder = LabelEncoder()
        # label_encoded = label_encoder.fit_transform(labeldict)
        # onehot_encoder = OneHotEncoder(sparse=False)
        # onehot_encoded = onehot_encoder.fit_transform(label_encoded.reshape(-1, 1))

        # 获取标签
        label = self.labels[self.samples[idx]]
        label_num = labelmap[label]
        label_one_hot = label_one_hot[label_num]
        if self.transform:
            sample = self.transform(sample)
        # return sample, label
        return torch.tensor(sample, dtype=torch.float), label_one_hot


data_path = "../data/train_set_10x"

if __name__ == "__main__":
    # #
    # # # length 长度
    # # train_dataset_size = len(train_dataset)
    # # test_dataset_size = len(test_dataset)
    # # print("训练数据集的长度为{}".format(train_dataset_size))
    # # print("测试数据集的长度为{}".format(test_dataset_size))
    # # print(train_dataset)
    # # print(train_loader)
    #
    train_dataset = MyDataset(data_path, train=True, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = MyDataset(data_path, train=False, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    for data in test_loader:
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        inputs = inputs.permute(1, 0, 2, 3)
        # labels = torch.unsqueeze(labels, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        print("********************************************************************************")
        print(inputs.shape)
        print(inputs)
        print("********************************************************************************")
        print(labels.shape)
        print(labels)
        print("********************************************************************************")
        break
