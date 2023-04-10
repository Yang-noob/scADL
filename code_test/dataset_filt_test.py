import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import Labels_Process as lp
from utils import Datasets_Process as dp
import torch


dataset1 = pd.read_hdf('G:/actinn_dataset/tma_10x_cleaned.h5', key="dge")
label1 = pd.read_csv('G:/actinn_dataset/tma_10x_cleaned_label.txt', header=None, sep='\t')

counts = (dataset1 != 0).sum(axis=1)
print(counts)
max_id = counts.idxmax()
min_id = counts.idxmin()
print(counts[max_id])
print(counts[min_id])
