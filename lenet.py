import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch._C import device
from torch.optim import optimizer
from torchvision import datasets, transforms
import torch.utils.data
import torch.nn.functional as F


device = ('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()])

train_set = datasets.MNIST(
    r'D:\program file\pyw\data\mnist', download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = datasets.MNIST(
    r'D:\program file\pyw\data\mnist', download=True, train=False, transform=transforms)
testloader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)
train_data_size = len(train_set)
test_data_size = len(test_set)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Relu3 = nn.ReLU()
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.Relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.Relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.Relu3(out)
        out = self.pool3(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


model = LeNet5().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


epochs = 10
train_loss, val_loss = [], []
for epoch in range(epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    for idx, (image, label) in enumerate(trainloader):
        imgae, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        pred = criterion(pred, label)
        loss = criterion(pred, label)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_train_loss = total_train_loss / (idx+1)
    train_loss.append(total_train_loss)

    model.eval()
    total = 0

    for idx, (image, label) in enumerate(testloader):
        imgae, label = image.to(device), label.to(device)
        pred = criterion(pred, label)
        loss = criterion(pred, label)
        total_val_loss += loss.item()
        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total += 1
    accuracy = total / test_data_size

    total_val_loss = total_val_loss / (idx+1)
    val_loss.append(total_val_loss)
    print(f"epoch:{epoch} | train loss: {total_train_loss} | validation loss {total_val_loss} | accuracy: {accuracy}")
