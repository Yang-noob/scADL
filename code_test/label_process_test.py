import sys

sys.path.append("..")

import pandas as pd
from utils import Labels_Process as lp

train_label = pd.read_csv('G:/labels.csv', header=None)
print(train_label.shape)
print(train_label)
print("*" * 100)
dict1 = lp.type_to_label_dict(train_label)
print(dict1)
dict1 = lp.label_to_type_dict(dict1)
print(dict1)

# lp = Labels_Process()
# type_to_label_dict = lp.type_to_label_dict(train_label.iloc[:, 1])
#
# print(type_to_label_dict)
# print("*" * 100)
# label_to_type_dict = lp.label_to_type_dict(type_to_label_dict)
# print(label_to_type_dict)
# print("*" * 100)
# labels = lp.convert_type_to_label(train_label.iloc[:, 1], type_to_label_dict)
# print(len(labels))
# print(labels)
# # train_label[2] = pd.Series(labels)
# # train_label.to_csv('./labels_new.csv', index=False, header=False)
# print("*" * 100)
# one_hot_matrix, num_classes = lp.one_hot_matrix(labels)
# print(len(one_hot_matrix))
# print(one_hot_matrix.shape)
# print(one_hot_matrix[1])
# print(num_classes)
# print("*" * 100)
