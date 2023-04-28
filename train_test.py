import torch.nn as nn
import torch.optim as optim
from read_datasets import MyDataset
from torch.utils.data import DataLoader
import torch
from models import FCN_1, FCN, CNN
import pandas as pd
from utils import Labels_Process as lp
import numpy as np
import datetime
from utils import Datasets_Process as dp
from utils import Dimension_Processing as dip
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import options as ops


# 解析命令行参数
# parser = ops.get_train_parser()
# args = parser.parse_args()
#
# pred_model_path = args.pre_model_path
# train_set_path = args.train_set_path
# train_label_path = args.train_label_path
# epoch = args.epoch
# batch_size = args.batch_size
# model_sava_path = args.model_sava_path
# device = args.device
# learning_rate = args.learning_rate
# print_loss = args.print_loss
# save_logs = args.save_logs
#
#
# 选用 CPU 或者 GPU 训练
# if device == 'GPU' or device == 'gpu':
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
#         print("GPU不可用，将使用CPU进行训练！")
# device = torch.device("cpu")

'''指定训练设备'''
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

'''设置训练集和标签路径'''
data_path = "G:/test_data/train_set.h5"
label_path = "G:/test_data/train_label.txt"

# data_path = "G:/actinn_dataset/tma_10x_cleaned.h5"
# label_path = "G:/actinn_dataset/tma_10x_cleaned_label.txt"


'''读取数据集和标签'''
dataset1 = pd.read_hdf(data_path, key="dge")
label1 = pd.read_csv(label_path, header=None, sep='\t')

# dataset1 = dataset1.iloc[:,:9000]
# print(dataset1.shape)
# print(label1.shape)
# print(dataset1)
# print(label1)

'''将标签转为字典'''
label1_dict = lp.type_to_label_dict(label1)
'''将带细胞类型标签转为纯数字标签'''
lab = lp.convert_type_to_label(label1, label1_dict)
'''将数字标签转为独热编码，获取分类数量'''
one_hot_matrix, num_class = lp.one_hot_matrix(lab)
# print(label1_dict)
# print(lab)

'''将数据集和数字标签转为numpy数组'''
arr_data = np.array(dataset1.values, dtype=np.float32)
arr_label = np.array(lab)
print("原始数据集大小:",arr_data.shape)
# print(arr_data)

'''将数据转为tensor并加载到GPU'''
tensor_data = torch.from_numpy(arr_data).float().to(device)
tensor_data = dp.tensor_normalize(tensor_data)
tensor_data = tensor_data.to('cpu')
tensor_data = dip.myPCA(tensor_data, features=1000)
features = tensor_data.shape[0]
tensor_label = torch.from_numpy(one_hot_matrix).float().to(device)

'''使用Dataset和Dataloader加载训练集，percentage为训练集占比'''
train_dataset = MyDataset(tensor_data, tensor_label, percentage=0.8, train=True, transform=False)
train_loader = DataLoader(train_dataset, batch_size=800, shuffle=True)

'''使用Dataset和Dataloader加载验证集，percentage为验证集占比'''
test_dataset = MyDataset(tensor_data, tensor_label, percentage=0.2, train=False, transform=False)
test_loader = DataLoader(test_dataset, batch_size=800, shuffle=True)

'''打印训练集和测试集合长度'''
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练集大小: {}".format(train_dataset_size))
print("测试集大小: {}".format(test_dataset_size))
print("基因数量: {}".format(features))
print("细胞类型数: {}".format(num_class))

# 实例化网络模型
model = FCN_1(features=features, total_number_types=num_class)

# 加载已训练好的参数
# model.load_state_dict(torch.load('./checkpoints/A_0.9867_2023-04-08_18-34-04.pth'))
# 将模型加载到GPU设备中
model = model.to(device)

# 定义损失函数、优化器
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# 设置神经网络的一些参数
total_train_step = 0  # 训练的次数
total_test_step = 0  # 测试的次数
epoch = 300  # 训练的轮数
writer = SummaryWriter(log_dir='logs/train_logs')

for i in range(epoch):
    print("----------------第 {} 轮训练----------------".format(i + 1))
    # 训练步骤开始
    model.train()
    for data in train_loader:
        inputs, labels = data
        # inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        # inputs = inputs.permute(1, 0, 2, 3)
        # print(inputs.shape)
        # print(labels.shape)

        # train_set = dp.capitalize_genes_name(dataset1)
        # train_set = dp.filt_duplicate_rows(train_set)
        # inputs = normalize(inputs)

        inputs = inputs.to(device)  # GPU           #imgs = imgs.to(device)
        labels = labels.to(device)  # GPU     #targets = targets.to(device)
        outputs = model.forward(inputs)  # 让输入通过层层特征提取网络（前向传播）
        # 特征提取：输入的像素点矩阵x * 权重参数矩阵w的过程，像素矩阵x的行列（形状）会发生变化，权重矩阵w的元素值将来会被不断更新。
        loss = criterion(outputs, labels)  # 计算在dataloader中一次训练的损失（每一轮输出的得分和真实值作比较的过程）
        # 优化器优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，求解损失函数梯度
        optimizer.step()  # 更新权重参数

        total_train_step = total_train_step + 1
        if total_train_step % 7 == 0:
            print("第 {} 次训练的Loss：{}".format(total_train_step, loss.item()))

        writer.add_scalar('train_loss', loss, total_train_step)

    # 验证步骤开始（验证训练结果怎么样）
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    # 无梯度的目的是:正处在验证阶段,所以不用对梯度进行调整,无需优化神经网络参数,也可以节省内存
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
            # inputs = inputs.permute(1, 0, 2, 3)
            inputs = inputs.to(device)  # GPU          #imgs = imgs.to(device)
            labels = labels.to(device)  # GPU    #targets = targets.to(device)
            outputs = model.forward(inputs)
            # 特征提取网络经过该轮训练，神经网络参数w被更新，将输入图片放入该网络后的到的得分值被记录下来
            loss = criterion(outputs, labels)  # 计算损失
            total_test_loss = total_test_loss + loss
            average_loss = total_test_loss / test_dataset_size
            accuracy = (outputs.argmax(1) == labels.argmax(1)).sum()  # 正确率的分子
            total_test_accuracy = total_test_accuracy + accuracy
            CorrectRate = total_test_accuracy / test_dataset_size

    writer.add_scalar('total_test_loss', total_test_loss, i)
    writer.add_scalar('average_test_loss', average_loss, i)
    writer.add_scalar('accuracy', CorrectRate, i)

    print("测试集上的总loss:{}".format(total_test_loss))
    print("测试集上的平均loss:{}".format(average_loss))
    print("整体测试集上的正确率: {}".format(CorrectRate))
    total_test_step = total_test_step + 1

    # if CorrectRate >= 0.995:
    #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     torch.save(model.state_dict(), "./checkpoints/A_{}_{}.pth".format(format(CorrectRate, '.4f'), timestamp))
    #     print("模型已保存")

    # if (i+1) % 200 == 0:
    #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     torch.save(model.state_dict(), "./checkpoints/B_{}_epoch{}_{}.pth".format(format(CorrectRate, '.4f'),i+1, timestamp))
    #     print("模型已保存")

writer.close()
