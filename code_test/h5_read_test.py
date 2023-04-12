import pandas as pd
import numpy as np


data_path = "G:/actinn_dataset/tma_10x_cleaned.h5"
label_path = "G:/actinn_dataset/tma_10x_cleaned_label.txt"

dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')

print(dataset1.shape)
print(dataset1)
print(label1.shape)
print(label1)