import pandas as pd

# 读取csv文件
csv_file = "G:/labels.csv"
data = pd.read_csv(csv_file, header=None)

# 打印数据的形状
print("Data shape:", data.shape)
print(data)
print(data.loc[1, ])
