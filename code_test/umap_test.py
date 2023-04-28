import umap
import pandas as pd
import numpy as np


data_path = "G:/test_data/train_set.h5"
data = pd.read_hdf(data_path, key="dge")
data = np.array(data.values, dtype=np.float32)
# 将数据降到2维
umap_ = umap.UMAP(n_components=900)
data_umap = umap_.fit_transform(data)
print(data_umap.shape)
print(data_umap)