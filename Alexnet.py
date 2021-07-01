
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = ('cuda' if torch.cuda.is_available() else 'cpu')

batch = 128
epochs = 30
lr = 0.0001
classes = 2

transform = transforms.Compose(
    [transforms.Resize(size=(227, 227)),
     transforms.CenterCrop(224),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     ]
)
train_set = datasets.ImageFolder(
    root=r'D:\program file\pyw\data\dog and cat\train', transform=transform)

test_set = datasets.ImageFolder(
    root=r'D:\program file\pyw\data\dog and cat\test', transform=transform)

train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
vaild_loader = DataLoader(test_set, batch_size=batch, shuffle=False)


class Alexnet(nn.module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


model = Alexnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    total_train = 0
    correct_train = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        data, target = data.cuda(), target.cuda()

        # clear gradient
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)
        loss = criterion(output, target)

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        predicted = torch.max(output.data, 1)[1]
        total_train += len(target)
        correct_train += sum((predicted == target).float())
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print("Train Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
                epoch+1, epochs, batch_idx+1, len(train_loader),
                correct_train / float((batch_idx + 1) * batch),
                train_loss / float((batch_idx + 1) * batch)))

    train_acc_ = 100 * (correct_train / float(total_train))
    train_loss_ = train_loss / total_train

    return train_acc_, train_loss_


def validate(valid_loader, model, criterion, epoch):

    model.eval()
    total_valid = 0
    correct_valid = 0
    valid_loss = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = Variable(data), Variable(target)

        data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)

        predicted = torch.max(output.data, 1)[1]
        total_valid += len(target)
        correct_valid += sum((predicted == target).float())
        valid_loss += loss.item()

        if batch_idx % 100 == 0:
            print("Valid Epoch: {}/{} [iter： {}/{}], acc： {:.6f}, loss： {:.6f}".format(
                epoch+1, epochs, batch_idx+1, len(valid_loader),
                correct_valid / float((batch_idx + 1) * batch),
                valid_loss / float((batch_idx + 1) * batch)))

    valid_acc_ = 100 * (correct_valid / float(total_valid))
    valid_loss_ = valid_loss / total_valid

    return valid_acc_, valid_loss_


def training_loop(model, criterion, optimizer, train_loader, valid_loader):
    # set objects for storing metrics
    total_train_loss = []
    total_valid_loss = []
    total_train_accuracy = []
    total_valid_accuracy = []

    # Train model
    for epoch in range(epochs):
        # training
        train_acc_, train_loss_ = train(
            train_loader, model, criterion, optimizer, epoch)
        total_train_loss.append(train_loss_)
        total_train_accuracy.append(train_acc_)

        # validation
        with torch.no_grad():
            valid_acc_, valid_loss_ = validate(
                valid_loader, model, criterion, epoch)
            total_valid_loss.append(valid_loss_)
            total_valid_accuracy.append(valid_acc_)

        print('==========================================================================')
        print("Epoch: {}/{}， Train acc： {:.6f}， Train loss： {:.6f}， Valid acc： {:.6f}， Valid loss： {:.6f}".format(
            epoch+1, epochs,
            train_acc_, train_loss_,
            valid_acc_, valid_loss_))
        print('==========================================================================')

    print("====== END ==========")

    return total_train_loss, total_valid_loss, total_train_accuracy, total_valid_accuracy


total_train_loss, total_valid_loss, total_train_accuracy, total_valid_accuracy = training_loop(
    model, criterion, optimizer, train_loader, vaild_loader)
