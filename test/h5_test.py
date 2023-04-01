import h5py


def print_attrs(name, obj):
    print(name)


with h5py.File('C:/Users/dell/Desktop/毕业设计/源代码/参考代码/ACTINN-master/test_data/test_set.h5', 'r') as f:
    f.visititems(print_attrs)
    # 访问键为'matrix'的值
    dge = f['dge']
    axis0 = dge['block0_values'][:]
    # features = f['matrix/features']
    # # 访问matrix下的data
    # data = matrix['indices'][:]
    # shape = matrix['shape'][:]
    print('*************************')
    print(axis0.shape)
    # print('*************************')
    # print(data)
    # print('*************************')
    # print(shape)