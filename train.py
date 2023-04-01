import torch.nn as nn
import torch.optim as optim
from my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch
from Net import Net


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

data_path = "./data/train_set_10x"
train_dataset = MyDataset(data_path, train=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = MyDataset(data_path, train=False, transform=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# length 长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练数据集的长度为{}".format(train_dataset_size))
print("测试数据集的长度为{}".format(test_dataset_size))

# 初始化神经网络，损失函数和优化器
net = Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

# 设置神经网络的一些参数
total_train_step = 0  # 训练的次数
total_test_step = 0  # 测试的次数
epoch = 100  # 训练的轮数


for i in range(epoch):
    print("-------第 {}轮训练开始-------".format(i + 1))
    # 训练步骤开始
    net.train()
    for data in train_loader:
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
        #inputs = inputs.permute(1, 0, 2, 3)
        inputs = inputs.to(device)  # GPU           #imgs = imgs.to(device)
        labels = labels.to(device)  # GPU     #targets = targets.to(device)
        outputs = net.forward(inputs)  # 让输入通过层层特征提取网络（前向传播）
        # 特征提取：输入的像素点矩阵x * 权重参数矩阵w的过程，像素矩阵x的行列（形状）会发生变化，权重矩阵w的元素值将来会被不断更新。
        loss = criterion(outputs, labels)  # 计算在dataloader中一次训练的损失（每一轮输出的得分和真实值作比较的过程）

        # 优化器优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，求解损失函数梯度
        optimizer.step()  # 更新权重参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("第 {} 次训练的Loss：{}".format(total_train_step, loss.item()))

    # 验证步骤开始（验证训练结果怎么样）
    net.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    # 无梯度的目的是:正处在验证阶段,所以不用对梯度进行调整,无需优化神经网络参数,也可以节省内存
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = torch.unsqueeze(inputs, dim=0)  # 在第0维增加一维，变为(N, C, H, W)
            inputs = inputs.to(device)  # GPU          #imgs = imgs.to(device)
            labels = labels.to(device)  # GPU    #targets = targets.to(device)
            outputs = net.forward(inputs)
            # 特征提取网络经过该轮训练，神经网络参数w被更新，将输入图片放入该网络后的到的得分值被记录下来
            loss = criterion(outputs, labels)  # 计算损失
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == labels).sum()  # 正确率的分子
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_test_accuracy / test_dataset_size))
    total_test_step = total_test_step + 1

    torch.save(net, "./checkpoints/ywh_{}.pth".format(i))
    print("模型已保存")
# # 训练神经网络
# for epoch in range(epoch):  # 进行10个epoch的训练
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = inputs.cuda(0)
#         labels = labels.cuda(0)
#         optimizer.zero_grad()
#
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 100 == 0:  # 每 100 批次打印一次损失函数值
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# # 测试神经网络
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, labels = data
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the %d test images: %d %%' % (total,
#                                                                 100 * correct / total))


