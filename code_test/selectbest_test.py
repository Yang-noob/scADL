from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, mutual_info_regression
import pandas as pd
from utils import Labels_Process as lp
import cupy as cp
import numpy as np

data_path = "G:/test_data/train_set.h5"
label_path = "G:/test_data/train_label.txt"

'''读取数据集和标签'''
dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')

'''将标签转为字典'''
label1_dict = lp.type_to_label_dict(label1)
'''将带细胞类型标签转为纯数字标签'''
lab = lp.convert_type_to_label(label1, label1_dict)

# 使用SelectKBest函数选择最好的两个特征
dataset1 = dataset1.T
selector = SelectKBest(f_classif, k=10000)
X_new = selector.fit_transform(dataset1, lab)
X_new = X_new.T
# 打印选择的特征
print(selector.get_support(indices=True))
print(X_new.shape)
print(X_new)
# 输出：[2, 3]


# # 加载鸢尾花数据集
# iris = load_iris()
# X, y = iris.data, iris.target
#
# # 将特征矩阵和目标向量转换为CuPy数组
# arr_data = np.array(dataset1.values, dtype=np.float32)
# arr_data = arr_data.T
# X_gpu = cp.asarray(arr_data)
# y_gpu = cp.asarray(lab)
# #
# # # 在GPU中计算卡方检验得分
# selector = SelectKBest(chi2, k=800)
# scores_gpu = selector.score_func(X_gpu, y_gpu)
# #
# # # 将选择的特征复制回CPU中
# scores = cp.asnumpy(scores_gpu)
# selected_features = np.argsort(scores)[::-1][:2]
# X_new = dataset1[:, selected_features]
# X_new = X_new.T
# #
# # # 输出选择的特征矩阵X_new
# print(X_new[:5])
