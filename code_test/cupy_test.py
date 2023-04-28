import cupy as cp
from cupy import linalg
from cupyx.scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from utils import Labels_Process as lp


data_path = "G:/test_data/train_set.h5"
label_path = "G:/test_data/train_label.txt"

'''读取数据集和标签'''
dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')

'''将标签转为字典'''
label1_dict = lp.type_to_label_dict(label1)
'''将带细胞类型标签转为纯数字标签'''
lab = lp.convert_type_to_label(label1, label1_dict)

data = dataset1.T
# 将scRNA-seq数据转换为cupy array
data_cupy = cp.asarray(data)

# 计算数据的方差
variances = cp.var(data_cupy, axis=0)

# 使用SelectKBest和f_classif进行特征选择
selector = SelectKBest(f_classif, k=900)
selected_data = selector.fit_transform(data_cupy.get(), lab)
selected_indices = selector.get_support(indices=True)

# 获取选择的特征的列名
selected_feature_names = data.columns[selected_indices]

# 打印选择的特征列名
print(selected_feature_names)
