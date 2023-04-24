import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 读入 scRNA-seq 数据
data_path = "G:/test_data/train_set.h5"
data = pd.read_hdf(data_path, key="dge")

# 数据标准化
data_norm = (data - data.mean()) / data.std()
print(data_norm)
data_norm = data_norm.T

# 进行 PCA
pca = PCA(n_components=1000)
data_pca = pca.fit_transform(data_norm)
print(data_pca)

# 可视化 PCA 结果
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
