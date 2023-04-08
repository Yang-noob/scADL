import numpy as np

# # 假设原始一维数组长度为10
# arr = np.random.rand(10)
#
# # 转换为方形的二维数组，不足的部分填充为0
# size = int(np.ceil(np.sqrt(len(arr))))
# arr_2d = np.resize(arr, (size, size))
#
# print(arr)
# print(arr_2d)


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
n = int(np.sqrt(len(a)))  # 计算方形矩阵边长
w = n
h = n
if w * h == len(a):
    b = a.reshape(w, h)  # 转为方形矩阵
else:
    while w*(h+1) < len(a):
        h += 1
    b = np.pad(a, (0, w * (h+1)-len(a)), mode='constant', constant_values=0)  # 填充0
    b = b.reshape(w, h+1)  # 转为方形矩阵
print(b)
