import pandas as pd

train_label = pd.read_csv('G:/labels.csv', header=None)
a = sorted(set(train_label.iloc[:,1]))
print(a)
all_type = list(a)
print(all_type)
# type_to_label_dict = {}
# for i in range(len(all_type)):
#     type_to_label_dict[all_type[i]] = i
#
# print(type_to_label_dict)