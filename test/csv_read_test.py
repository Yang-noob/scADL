import pandas as pd

# 读取csv文件
csv_file = "G:/file.csv"
data = pd.read_csv(csv_file)['cell_type']

# 打印数据的形状
print("Data shape:", data.shape)

# 查看前10个细胞和基因的表达量
print("First 10 cells and genes expression:")
print(data)
