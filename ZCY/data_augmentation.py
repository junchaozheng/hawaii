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
import time
import logging
import random
from sub import subMNIST

trainset_import = pickle.load(open("./data/train_labeled.p", "rb"))
validset_import = pickle.load(open("./data/validation.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_import, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)

# data augmentation: rotate randomly from -45 to 45
def rotate_random(data):
    s = np.append(np.random.randint(-45,45,9), 0)
    tmp = map(lambda x:ndimage.rotate(data.numpy(), x, reshape=False), s)
    #rotated = torch.ByteTensor(tmp)
    rotated = map(lambda x: torch.from_numpy(x), tmp)
    return rotated

# move up or down by 1 to 4 units
def img_translation(data, dist):
    move = random.randint(0,1)
#     dist = random.randint(1,4)
    if move == 0:
        data = data.numpy()[dist: ]
        pad = np.zeros((dist,28))
        new_image = np.concatenate((data,pad))
        return torch.from_numpy(new_image)
    else:
        data = data.numpy()[:28-dist]
        pad = np.zeros((dist,28))
        new_image = np.concatenate((pad,data))
        return torch.from_numpy(new_image)

def add_noise(data, time):
    # noise = np.random.normal(0,1)
    data = data.numpy()
    for i in data:
        for j in i:
            if j != 0:
                j += np.random.normal(0,1)
    return torch.from_numpy(data)

def main():
    # move up and down
    train_data_sub_translation = []
    for i in range(0, 3000):
        train_data_sub_translation.extend(map(lambda x: img_translation(trainset_import.train_data[i], x), np.random.randint(1,4,10)))

    # add noise
    train_data_sub_noise = []
    for i in range(0, 3000):
        train_data_sub_noise.extend(map(lambda x: add_noise(trainset_import.train_data[i], x), range(0, 10)))

    # rotate
    train_data_sub_rotate = []
    for i in range(0, 3000):
        train_data_sub_rotate.extend(rotate_random(trainset_import.train_data[i]))

    train_data_sub = train_data_sub_rotate + train_data_sub_translation + train_data_sub_noise

    # label array
    train_labels_sub = []
    for i in range(0, 3000):
        train_labels_sub.extend([trainset_import.train_labels[i]] * 10)
    train_labels = train_labels_sub + train_labels_sub + train_labels_sub
    train_labels = torch.from_numpy(np.array(train_labels))

    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])
    trainset_new = subMNIST(root='./data/', train=True, download=True, transform=transform, k=90000)

    trainset_new.train_data = train_data_sub
    trainset_new.train_labels = train_labels
    pickle.dump(trainset_new, open("./data/train_labeled_allmethod.p", "wb" ))

