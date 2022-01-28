# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/8/30 17:14
# @File    :点云识别3.py
# @Software:PyCharm
"""
@remarks :
"""
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


x_train = np.array([])
y_train = np.array([])
x_test = np.array([])
y_test = np.array([])
for i in range(24):
    for f in os.listdir("pointcloud2" + "/" + str(i)):
        Data = np.loadtxt("pointcloud2" + "/" + str(i) + "/" + f, delimiter=' ')
        x_train = np.append(x_train, Data)
        y_train = np.append(y_train, i)
x_train = x_train.reshape(1440, 3, 49, 73)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
x_train = x_train.float()
x_test = x_train.float()
y_train = y_train.long()
y_test = y_train.long()
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)


batch_size = 12
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    # 卷积神经网络
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 10, kernel_size=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(10, 16, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 24, kernel_size=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(24*5*8, 800),
                                 # nn.BatchNorm1d(120),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(800, 500),
                                 # nn.BatchNorm1d(120),
                                 nn.ReLU())
        # self.fc3 = nn .Sequential(nn.Linear(8400, 1000),
        #                          # nn.BatchNorm1d(120),
        #                          nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(500, 100),
                                 # nn.BatchNorm1d(84),
                                 nn.ReLU(),
                                 nn.Linear(100, 24))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        return x


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 12 == 11:
            print('[%d,%5d] loss:%.3f' % (epoch+1, batch_idx+1, running_loss/12))
            running_loss = 0.0


def tist():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predict = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print('Accurate on test set: %d %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(60):
        train(epoch)
        tist()
