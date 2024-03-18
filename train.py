import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset,random_split
from Package_dataset import package_dataset


# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Models.LeNet import LeNet
from Models.AlexNet import AlexNet
from Models.ZFNet import ZFNet
from Models.VGG19 import VGG19
from Models.GoogLeNet import GoogLeNet
from Models.ResNet50 import ResNet50
from Models.DenseNet import DenseNet
from Models.SqueezeNet import SqueezeNet
from Models.Mnasnet import MnasNetA1
from Models.MobileNetV1 import MobileNetV1
from Models.MobileNetV2 import MobileNetV2
from Models.MobileNetV3 import MobileNetV3_large, MobileNetV3_small
from Models.shuffuleNetV1 import shuffuleNetV1_G3
from Models.shuffuleNetV2 import shuffuleNetV2
from Models.Xception import Xception
from Models.EfficientNet import EfficientNetB0

data = np.load('Dataset/data.npy')
label = np.load('Dataset/label.npy')

dataset_partition_rate = 0.7
epoch_number = 1000
show_result_epoch = 10

dataset, channels, length, classes = package_dataset(data, label)

# partition dataset
train_len = int(len(dataset) * dataset_partition_rate)
test_len = int(len(dataset)) - train_len
train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, test_len])


# 数据库加载
class Dataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 数据库dataloader
Train_dataset = Dataset(train_dataset)
Test_dataset = Dataset(test_dataset)
dataloader = DataLoader(Train_dataset, shuffle=True, batch_size=50)
testloader = DataLoader(Test_dataset, shuffle=True, batch_size=50)
# 训练设备选择GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型初始化
# model = LeNet(in_channels=channels, input_sample_points=length, classes=classes)
# model = AlexNet(in_channels=channels, input_sample_points=length, classes=classes)
# model = AlexNet(in_channels=channels, input_sample_points=length, classes=classes)
# model = ZFNet(in_channels=channels, input_sample_points=length, classes=classes)
# model = VGG19(in_channels=channels, classes=classes)
# model = GoogLeNet(in_channels=channels, classes=classes)
# model =ResNet50(in_channels=channels, classes=classes)
# model =DenseNet(in_channels=channels, classes=classes)
# model =SqueezeNet(in_channels=channels, classes=classes)
# model =MobileNetV1(in_channels=channels, classes=classes)
# model =MobileNetV2(in_channels=channels, classes=classes)
# model =MobileNetV3_small(in_channels=channels, classes=classes)
# model =MobileNetV3_large(in_channels=channels, classes=classes)
# model =shuffuleNetV1_G3(in_channels=channels, classes=classes)
# model =shuffuleNetV2(in_channels=channels, classes=classes)
# model =Xception(in_channels=channels, classes=classes)
model =EfficientNetB0(in_channels=channels, classes=classes)
model.to(device)

# 损失函数选择
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
# 优化器选择
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_acc_list = []
test_acc_list = []


# 训练函数
def train(epoch):
    model.train()
    train_correct = 0
    train_total = 0

    for data in dataloader:
        train_data_value, train_data_label = data
        train_data_value, train_data_label = train_data_value.to(device), train_data_label.to(device)
        train_data_label_pred = model(train_data_value)
        loss = criterion(train_data_label_pred, train_data_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % show_result_epoch == 0:
        probability, predicted = torch.max(train_data_label_pred.data, dim=1)
        train_total += train_data_label_pred.size(0)
        train_correct += (predicted == train_data_label).sum().item()
        train_acc = round(100 * train_correct / train_total, 4)
        train_acc_list.append(train_acc)
        print('=' * 10, epoch // 10, '=' * 10)
        print('loss:', loss.item())
        print(f'Train accuracy:{train_acc}%')
        test()


# 测试函数
def test():
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for testdata in testloader:
            test_data_value, test_data_label = testdata
            test_data_value, test_data_label = test_data_value.to(device), test_data_label.to(device)
            test_data_label_pred = model(test_data_value)
            test_probability, test_predicted = torch.max(test_data_label_pred.data, dim=1)
            test_total += test_data_label_pred.size(0)
            test_correct += (test_predicted == test_data_label).sum().item()
    test_acc = round(100 * test_correct / test_total, 3)
    test_acc_list.append(test_acc)
    print(f'Test accuracy:{(test_acc)}%')


for epoch in range(1, epoch_number+1):
    train(epoch)

plt.plot(np.array(range(epoch_number//show_result_epoch)) * show_result_epoch, train_acc_list)
plt.plot(np.array(range(epoch_number//show_result_epoch)) * show_result_epoch, test_acc_list)
plt.legend(['train', 'test'])
plt.title('Result')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
