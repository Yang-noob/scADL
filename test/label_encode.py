from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# labels = ['cat', 'dog', 'bird', 'fish']
# label_encoder = LabelEncoder()
# label_encoded = label_encoder.fit_transform(labels)
# onehot_encoder = OneHotEncoder(sparse=False)
# onehot_encoded = onehot_encoder.fit_transform(label_encoded.reshape(-1, 1))
# print(onehot_encoded[0])

import torch

y = torch.tensor([0, 2, 1, 4, 3, 5])  # 假设有6个类别
num_classes = 6  # 类别数量

y_onehot = torch.eye(num_classes)[y]  # one-hot编码
inputs = y_onehot.unsqueeze(0)
print(y_onehot)
