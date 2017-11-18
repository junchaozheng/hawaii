from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy import ndimage
from sub import subMNIST
import logging
import time

# create a log file to record training process
check_point = int(time.time())
logging.basicConfig(filename="{}.log".format(check_point), level = logging.DEBUG)

# load data
logging.info("loading data...")
trainset_imoprt = pickle.load(open("./data/train_labeled_allmethod.p", "rb"))
trainset_unlabel_import = pickle.load(open("./data/train_unlabeled.p", "rb"))
validset_import = pickle.load(open("./data/validation.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.conv3_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Net()

"""
This function is to train labled and unlabeled data together.
"""

def train_pl(epoch, optimizer, train_unlabel_loader, alpha, lr, mo):
    model.train()
    for (batch_idx, (data1, target1)), (batch_idx2, (data2, target2)) in zip(enumerate(train_loader), enumerate(train_unlabel_loader)):
        data1, target1 = Variable(data1), Variable(target1)
        data2, target2 = Variable(data2), Variable(target2)
        optimizer.zero_grad()
        output1 = model(data1)
        output2 = model(data2)
        loss1 = F.nll_loss(output1, target1)
        loss2 = F.nll_loss(output2, target2)
        loss = loss1 + alpha * loss2
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            logging.info('Train Epoch: {} [Label: {}/{} ({:.0f}%)] [Unlabel: {}/{} ({:.0f}%)] \tLoss: {:.6f}, alpha={}, lr={}, mo={}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_idx * len(data2), len(train_unlabel_loader.dataset),
                100. * batch_idx / len(train_unlabel_loader), loss.data[0], alpha, lr, mo))

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

"""
This function take input from train unlabel loader, then use model to do prediction on unlabel data
"""
def predict_unlabel(train_unlabel_loader):
    label_predict = np.array([])
    model.eval()
    for data, target in train_unlabel_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp = output.data.max(1)[1].numpy().reshape(-1)
        label_predict = np.concatenate((label_predict, temp))
    pl_label = torch.from_numpy(label_predict)
    pl_label = pl_label.byte()
    return pl_label

"""
This function calculates alpha for the loss of unlabel data.
"""
def getAlpha(epoch):
    if epoch < 100:
        return 0.
    elif epoch >= 100 and epoch < 600:
        adj_alpha = (epoch - 100) / float(500) * 3.
        return adj_alpha
    else:
        return 3.

def main():
    eta = 0.01
    moi = 0.5
    mof = 0.99

    for epoch in range(1, 500):
        if trainset_unlabel_import.train_labels is None:
            trainset_unlabel_import.train_labels = torch.ByteTensor([-1]*len(trainset_unlabel_import))
            train_unlabel_loader = torch.utils.data.DataLoader(trainset_unlabel_import, batch_size=64, shuffle=False)
        else:
            train_unlabel_loader = torch.utils.data.DataLoader(trainset_unlabel_import, batch_size=64, shuffle=False)

        pl_label = predict_unlabel(train_unlabel_loader)
        trainset_unlabel_import.train_labels = pl_label
        train_unlabel_loader = torch.utils.data.DataLoader(trainset_unlabel_import, batch_size=64, shuffle=True)
        alpha = getAlpha(epoch)
        # update momentum
        mo = min(1, epoch / float(500)) * mof + max(0, 1 - epoch / float(500)) * moi
        optimizer = optim.SGD(model.parameters(), lr=eta, momentum=mo)
        train_pl(epoch, optimizer, train_unlabel_loader, alpha, eta, mo)
        # dump model
        if epoch % 20 == 0:
            pickle.dump(model, open("./model_{}_{}.p".format(check_point, epoch), "wb" ))
        test(epoch, valid_loader)
        eta *= 0.998

if __name__ == "__main__":
    main()
