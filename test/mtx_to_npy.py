import scipy.io as sio
import numpy as np
# 读取mtx文件
mtx_file = "../data/train_set_10x/matrix.mtx"
sparse_matrix = sio.mmread(mtx_file)

# 将稀疏矩阵转换为密集矩阵
dense_matrix = sparse_matrix.toarray()


# # 打印矩阵的形状
# print("Matrix shape:", dense_matrix.shape)
#
# # 查看前10个细胞和基因的表达量
# print("First 10 cells and genes expression:")
# print(dense_matrix[:10, :10])

for i in range(1000):
    mtx_col = dense_matrix[:, i]
    col = np.asarray(mtx_col).flatten()
    arr = np.pad(col, (0, 98), mode='constant', constant_values=0)
    arr2d = np.reshape(arr, (119, 119))
    path = '../data/train_set_10x/npy/{}.npy'.format(i)
    np.save(path, arr2d)

# mtx_col = dense_matrix[:, 0]
# col = np.asarray(mtx_col).flatten()
# arr = np.pad(col, (0, 98), mode='constant', constant_values=0)
# arr2d = np.reshape(arr, (119, 119))
#
# import matplotlib.pyplot as plt
#
# plt.imshow(arr2d)
# plt.show()
