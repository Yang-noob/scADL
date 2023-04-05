import sys

sys.path.append("..")

from utils import Datasets_Process
import pandas as pd

dataset1 = pd.read_hdf('G:/dataset_test/tma_10x_cleaned.h5', key="dge")
dataset2 = pd.read_hdf('G:/dataset_test/tma_ss2_cleaned.h5', key="dge")
print(dataset1.shape)
print(dataset2.shape)
print('*'*100)
data = Datasets_Process(dataset1, dataset2)
common_genes = data.get_common_genes()
print(len(common_genes))
merged_dataset, group_size = data.merge_datasets(common_genes)
print(merged_dataset.shape)
print(group_size)
print('*'*100)
group_size = [1000, 2]
sets = data.split_datasets(merged_dataset, group_size)
if sets is None:
    print("拆分失败")
else:
    for i in range(len(sets)):
        print(sets[i].shape)
        print('*'*100)


# sets = [dataset1, dataset2]
# sep_point = [0]
# for i in range(len(sets)):
#     print(sets[i].shape)
#     sep_point.append(sets[i].shape[1])
# # sets = Datasets_Process(dataset1, dataset2)
# print(sep_point)
#
# for i in range(len(sets)):
#     a = sum(sep_point[:(i+1)])
#     b = sum(sep_point[:(i+2)])
#     print(a)
#     print(b)
# genes = sets.get_common_genes()
# print(len(genes))
# datasets = sets.normalize(genes)
# print('*'*100)
# print(len(datasets))