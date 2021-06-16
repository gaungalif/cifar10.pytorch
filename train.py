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
fpath = Path(curr_dir).joinpath('mynetwork_state_dict.pth')

# print(len(validset)), print(len(trainset))
def training():
    net = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
    
    criterion = nn.CrossEntropyLoss()

    lr= float(input('lr = '))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    epoch = int(input('epoch = '))
    log_freq= int(input('log freq = '))

    task.train_network(epoch, train_loader, valid_loader, net, criterion, optimizer, log_freq=log_freq)  
    state_dict = net.state_dict()
    torch.save(state_dict, fpath)

if __name__ == "__main__":
    training()
