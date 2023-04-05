import scipy.io as sio

# 读取mtx文件
mtx_file = "G:/test_data/train_set_10x/matrix.mtx"
sparse_matrix = sio.mmread(mtx_file)

# 将稀疏矩阵转换为密集矩阵
dense_matrix = sparse_matrix.toarray()


# 打印矩阵的形状
print("Matrix shape:", dense_matrix.shape)

# 查看前10个细胞和基因的表达量
print("First 10 cells and genes expression:")
print(dense_matrix[:10, :10])