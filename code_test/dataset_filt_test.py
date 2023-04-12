import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import Labels_Process as lp
from utils import Datasets_Process as dp
import torch


def filt_genes(dataset, threshold):
    counts = (dataset != 0).sum(axis=1)
    filtered_genes = counts[counts > threshold].index
    filtered_dataset = dataset.loc[filtered_genes,]
    return filtered_dataset


dataset1 = pd.read_hdf('G:/actinn_dataset/tma_both_cleaned.h5', key="dge")
label1 = pd.read_csv('G:/actinn_dataset/tma_both_cleaned_label.txt', header=None, sep='\t')

print(dataset1.shape)

data = filt_genes(dataset1, 20)
print(data.shape)


# max_id = counts.idxmax()
# min_id = counts.idxmin()
# print(counts[max_id])
# print(counts[min_id])

# cell_type = len(set(label1[1]))
# print(cell_type)
#
# index = dataset1.columns.tolist()
# name = label1[0].tolist()
# print(index)
# print(name)
# sum_m = 0
# for i in range(len(name)):
#     if name[i] == index[i]:
#         sum_m += 1
# print(sum_m)



