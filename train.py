import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from cifar.datasets import loader
from cifar.models.models import MyNetwork
from cifar.trainer import task

base_dir = Path(curr_dir).joinpath('dataset')
train_loader, trainset = loader.train_loader(base_dir)
valid_loader, validset = loader.valid_loader(base_dir)

print(len(validset)), print(len(trainset))

clazz = 10
net = MyNetwork(ichan=3, clazz=clazz, imsize=(64,64)).to(task.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
if __name__ == '__main__':
    train = task.train_network(2, train_loader, valid_loader, net, criterion, optimizer, log_freq=20)

state_dict = net.state_dict()
fpath = Path(curr_dir).joinpath('mynetwork_state_dict.pth')
torch.save(state_dict, fpath)