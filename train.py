import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from cifar.datasets import loader
from cifar.models.models import MyNetwork
from cifar.trainer import task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--lrate', type=float, help='learning rate', required=True)
    parser.add_argument('-ep','--epoch',type=int, help='epoch', required=True)
    parser.add_argument('-lf','--lfreq',type=int, help='log frequency')
    parser.add_argument('-m','--momentum',type=float, help='momentum', required=True)
    args = parser.parse_args()
    
    LR = args.lrate
    EPOCHS = args.epoch
    LOG_FREQ = args.lfreq
    MOMENTUM = args.momentum


    curr_dir = Path(curr_dir)
    base_dir = curr_dir.joinpath('dataset')
    train_loader, trainset = loader.train_loader(base_dir)
    valid_loader, validset = loader.valid_loader(base_dir)
    fpath = curr_dir.joinpath('gaung.pth')

    net = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
    criterion = nn.CrossEntropyLoss()
    state_dict = net.state_dict()

    optimizer = optim.SGD(net.parameters(), LR, MOMENTUM)
    task.train_network(EPOCHS, train_loader, valid_loader, net, criterion, optimizer, LOG_FREQ)  
    torch.save(state_dict, fpath)

