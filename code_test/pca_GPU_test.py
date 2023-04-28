import cudf
import cuml

# 读取数据集
data = cudf.read_csv('path/to/data.csv')

# 拟合PCA模型
pca = cuml.PCA(n_components=2)
pca.fit(data)

# 对数据集进行降维
transformed_data = pca.transform(data)
