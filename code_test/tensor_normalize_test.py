import numpy as np
import pandas as pd
import torch
import time


def normalize(*datasets, scale_factor=10000, low_range=1, high_range=99):
    datasets_list = list(datasets)
    for i in range(len(datasets_list)):
        datasets_list[i] = np.array(datasets_list[i])
        datasets_list[i] = np.divide(datasets_list[i], np.sum(datasets_list[i], axis=0, keepdims=True)) * scale_factor
        datasets_list[i] = np.log2(datasets_list[i] + 1)
        expr = np.sum(datasets_list[i], axis=1)
        datasets_list[i] = datasets_list[i][
            np.logical_and(expr >= np.percentile(expr, low_range), expr <= np.percentile(expr, high_range)),]
        cv = np.std(datasets_list[i], axis=1) / (np.mean(datasets_list[i], axis=1) + 1e-10)
        datasets_list[i] = datasets_list[i][
            np.logical_and(cv >= np.percentile(cv, low_range), cv <= np.percentile(cv, high_range)),]
    if len(datasets_list) == 1:
        return torch.Tensor(datasets_list[0])
    return torch.tensor(datasets_list)


start_time = time.time()
dataset1 = pd.read_hdf('G:/actinn_dataset/tma_both_cleaned.h5', key="dge")
# label1 = pd.read_csv('G:/actinn_dataset/tma_10x_cleaned_label.txt', header=None, sep='\t')
print(np.array(dataset1))
data = normalize(dataset1)
print(data.shape)
print(data)
print('*'*100)
end_time = time.time()
print("文件读取用时：{:.2f}秒".format(end_time - start_time))
