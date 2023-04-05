import sys

sys.path.append("..")

from utils import Datasets_Process
import pandas as pd

dataset1 = pd.read_hdf('G:/dataset_test/tma_10x_cleaned.h5', key="dge")
# dataset2 = pd.read_hdf('G:/dataset_test/tma_ss2_cleaned.h5', key="dge")
sets = Datasets_Process(dataset1)
nor = sets.normalize(dataset1)
group = [30000, 5166]
spl = sets.split_datasets(nor, group)
print(nor)
print(spl[0].shape)
# print(dataset1.shape)
# print(dataset2.shape)
# data = sets.capitalize_genes_name()
# print(data)
# data = sets.filt_duplicate_rows()
# print(data)

