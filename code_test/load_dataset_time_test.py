import pandas as pd
import time
from utils import Datasets_Process as dp


start_time = time.time()

dataset1 = pd.read_hdf('E:/actinn_dataset/tma_both_cleaned.h5', key="dge")
# label1 = pd.read_csv('G:/actinn_dataset/tma_10x_cleaned_label.txt', header=None, sep='\t')
print(dataset1)
train_set = dp.capitalize_genes_name(dataset1)
print(train_set.shape)
print(train_set)
train_set = dp.filt_duplicate_rows(train_set)
print(train_set.shape)
print(train_set)
end_time = time.time()

print("文件读取用时：{:.2f}秒".format(end_time - start_time))