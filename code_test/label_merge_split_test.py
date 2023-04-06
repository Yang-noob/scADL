import sys

sys.path.append("..")

import pandas as pd
from utils import Labels_Process as lp
import os

label1 = pd.read_csv('G:/dataset_test/tma_facs_cleaned_label.txt', header=None, sep='\t')
label2 = pd.read_csv('G:/dataset_test/tma_mfd_cleaned_label.txt', header=None, sep='\t')

print(label1.shape)
print(label2.shape)
print('*'*100)
merged_label, group = lp.merge_labels(label1, label2, return_original_sep=True)
print(merged_label.shape)
print(merged_label)
print("group:", group)

labels = lp.split_labels(merged_label, group)
for i in range(len(labels)):
    print(labels[i].shape)
    labels[i].to_csv(f"G:/dataset_test/l_{i}.csv", header=None)
print("保存成功")
# filename = "G:/dataset_test/merged_label.csv"
# i = 1
# while os.path.exists(filename):
#     filename = f"G:/dataset_test/merged_label _{i}.csv"
#     i += 1
# merged_label.to_csv(filename, header=None)
