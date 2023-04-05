import sys

sys.path.append("..")

from utils import dataset_label_match


import pandas as pd

dataset = pd.read_hdf('G:/test_data/train_set.h5', key="dge")
train_label = pd.read_csv('G:/labels.csv', header=None)

data, label = dataset_label_match(dataset, train_label, check_common_cell=False)

print(data)
print(label)