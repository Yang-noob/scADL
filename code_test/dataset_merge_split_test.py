import sys

import numpy as np

sys.path.append("..")

from utils import Datasets_Process as dp
import pandas as pd

dataset1 = pd.read_hdf('G:/dataset_test/tma_10x_cleaned.h5', key="dge")
dataset2 = pd.read_hdf('G:/dataset_test/tma_ss2_cleaned.h5', key="dge")
print(dataset1.shape)
print(dataset1)
print(dataset2.shape)
print(dataset2)
print('*'*100)
data, group = dp.merge_datasets(dataset1,dataset2, return_original_sep=True)
# print(dataset1.index)
print("合并结果：")
print(data.shape)
print(data)
print(group)
print('*'*100)
data = np.array(data, dtype=np.float32)
print(data.shape)
sets = dp.split_datasets(data, group)
print("组数：",len(sets))
for i in range(len(sets)):
    print(sets[i].shape)
    print(sets[i])
