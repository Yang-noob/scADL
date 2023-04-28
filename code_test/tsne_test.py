from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


data_path = "G:/test_data/train_set.h5"
data = pd.read_hdf(data_path, key="dge")
data = np.array(data.values, dtype=np.float32)
# 将数据降到2维
data = data.T
tsne = TSNE(n_components=10000)
data_tsne = tsne.fit_transform(data)
data_tsne = data_tsne.T
print(data_tsne.shape)
print(data_tsne)
