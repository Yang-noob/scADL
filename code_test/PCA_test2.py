import torch
from torch import Tensor
from torch.linalg import svd
from typing import Tuple
import pandas as pd
import numpy as np
from utils import Dimension_Processing as dip

# def pca(X: Tensor, k: int) -> Tuple[Tensor, Tensor]:
#     """
#     :param X: 数据集张量，shape为(N, D)
#     :param k: 降维后的维度
#     :return: 降维后的数据集张量，shape为(N, k) 和 降维后的基向量张量，shape为(D, k)
#     """
#     N, D = X.shape
#     X_centered = X - X.mean(dim=0)  # 中心化
#
#     # 使用SVD分解计算特征向量和特征值
#     U, S, V = svd(X_centered)
#
#     # 选取前k个特征向量
#     U_k = U[:, :k]
#
#     # 将数据集投影到选定的特征向量上
#     X_pca = torch.matmul(X_centered, U_k)
#
#     return X_pca, U_k


data_path = "G:/test_data/train_set.h5"
data = pd.read_hdf(data_path, key="dge")
data = np.array(data.values, dtype=np.float32)
# tensor_data = torch.from_numpy(data).float().to('cuda')
# 定义一个随机的数据集张量X
# X = torch.randn(1000, 500).cuda()
# print(X)
tensor_data = dip.mySVD(data)
print(tensor_data.shape)
# N, D = tensor_data.shape
# print(N)
# print(D)
# print("*")
#
# tensor_data = tensor_data.T
# # X_centered = X - X.mean(dim=0)  # 中心化
# # print(X_centered)
# # 使用SVD分解计算特征向量和特征值
# U, S, V = svd(tensor_data)
# print(U)
# print(S)
# print(V)
# print("*")
# # 选取前k个特征向量
# k = 200
# U_k = V[:, :k]
# print(U_k)
# # 将数据集投影到选定的特征向量上
# X_pca = torch.matmul(tensor_data, U_k)
# print(X_pca)
