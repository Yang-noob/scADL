import numpy as np
import pandas as pd

# 生成一个示例scRNA-seq数据集，其中100个基因和10个细胞
genes = ['gene' + str(i) for i in range(1, 11)]
cells = ['cell' + str(i) for i in range(1, 11)]
data = np.random.randint(low=0, high=10, size=(10, 10))
df = pd.DataFrame(data, index=genes, columns=cells)

# 随机生成2个位置的行索引和列索引
rows = np.random.choice(df.index, size=8)
cols = np.random.choice(df.columns, size=3)

# 将这2个位置的值设为0
df.loc[rows, cols] = 0
print(df)
# # 统计每个细胞的非零表达值数量
counts = (df != 0).sum(axis=0)
print(counts)

# 设置阈值为n，保留非零表达值数量大于n的细胞
n = 5
filtered_cells = counts[counts > n].index
filtered_df = df[filtered_cells]
#
print(filtered_df)
