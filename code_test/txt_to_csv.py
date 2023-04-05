import pandas as pd

# 读取txt文件
txt_file = "G:/train_label.txt"
data = pd.read_csv(txt_file, sep="\t", index_col=0)

# 保存为csv文件
csv_file = "G:/file.csv"
data.to_csv(csv_file)
