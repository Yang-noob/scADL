from scipy.interpolate import interp1d
import torch
from models import FCN_1
import pandas as pd
from utils import Labels_Process as lp
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PredictDataset(Dataset):
    def __init__(self, dataset, transform=False):
        self.dataset = dataset
        self.transform = transform
        self.dataset_len = dataset.shape[1]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample = self.dataset[:, index]
        if len(sample) != 14063:
            # 定义原始数组对应的下标
            x = np.arange(len(sample))
            # 定义新数组对应的下标
            new_x = np.linspace(0, len(sample) - 1, 14063)
            # 使用线性插值法计算新数组的值
            f = interp1d(x, sample, kind='linear')
            sample = f(new_x)

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
        return torch.Tensor(sample)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

data_path = "G:/test_data/train_set.h5"
label_path = "G:/test_data/train_label.txt"

dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')
barcode = list(dataset1.columns)

label1_dict = lp.type_to_label_dict(label1)
label_to_type_dict = lp.label_to_type_dict(label1_dict)
print("-"*85)
print("字典:", label_to_type_dict)

arr_data = np.array(dataset1)

test_set = PredictDataset(arr_data)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

model = FCN_1()
model.load_state_dict(torch.load('./checkpoints/A_0.9867_2023-04-08_18-34-04.pth'))
model = model.to(device)

# 预测开始
model.eval()
with torch.no_grad():
    predicted_label = []
    for data in test_loader:
        inputs = data
        inputs = inputs.to(device)  # GPU          #imgs = imgs.to(device)
        outputs = model.forward(inputs)
        out = torch.argmax(outputs, dim=1)
        for i in range(len(out)):
            predicted_label.append(label_to_type_dict[out[i].item()])

    error = pd.DataFrame(columns=['细胞编号', '正确类型', '预测类型'])
    la = list(label1.iloc[:, 1])
    sum_c = 0
    for i in range(len(la)):
        if la[i] != predicted_label[i]:
            sum_c += 1
            row = {'细胞编号': barcode[i], '正确类型': la[i], '预测类型': predicted_label[i]}
            error.loc[i] = row
    print("-"*85)
    print("预测完成")
    print("正确率:",(1-(sum_c/len(predicted_label))) * 100,"%")
    print("共有 {} 个细胞预测错误".format(sum_c))
    print("错误情况:")
    print(error)
    # 将预测结果转为dataframe对象并保存
    predicted_label = pd.DataFrame({"cell_name":barcode, "cell_type":predicted_label})
    # predicted_label.to_csv("./results/predict_results/predicted_label.csv", index=False)
    print("-"*85)
    print("预测结果已保存到 ”./results/predict_results“ 目录下")
    print("-"*85)
