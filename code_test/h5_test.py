import h5py
#
#


def print_attrs(name, obj):
    print(name)
#
#


with h5py.File('C:/Users/dell/Desktop/毕业设计/源代码/参考代码/ACTINN-master/test_data/test_set.h5', 'r') as f:
    f.visititems(print_attrs)
    # 访问键为'matrix'的值
    dge = f['dge']
    axis0 = dge['axis0'][:]
    # features = f['matrix/features']
    # # 访问matrix下的data
    # data = matrix['indices'][:]
    # shape = matrix['shape'][:]
    print('*************************')
    print(axis0.shape)
    print(axis0)
    # print('*************************')
    # print(data)
    # print('*************************')
    # print(shape)

# import pandas as pd
# import numpy as np
#
# dataset1 = pd.read_hdf('G:/dataset_test/tma_10x_cleaned.h5', key="dge")
# print(dataset1.shape)
# print(dataset1)
# # print(dataset1.iloc[1, ])
# print('******************************************************************************')
# # di = dataset1.index
# # a = [s.upper() for s in di]
# # print(a)
# # data_columns = dataset1.columns
# # columns = dataset1.columns.tolist()
# # print(columns)
#
# arr = np.array(dataset1, dtype=np.float32)
#
# print(arr.shape)
# print(arr)
# # print(data_columns)