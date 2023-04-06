import sys

sys.path.append("..")

from utils import Datasets_Process as dp
import pandas as pd

dataset1 = pd.read_hdf('G:/dataset_test/tma_10x_cleaned.h5', key="dge")
dataset2 = pd.read_hdf('G:/dataset_test/tma_ss2_cleaned.h5', key="dge")
print(dataset1.shape)
# print(dataset1)
print(dataset2.shape)
# print(dataset2)
print('*'*100)
data = dp.get_common_genes(dataset1,dataset2)
# print(dataset1.index)
print(len(data))
