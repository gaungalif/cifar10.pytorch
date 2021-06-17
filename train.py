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

curr_dir = Path(curr_dir)
base_dir = curr_dir.joinpath('dataset')
train_loader, trainset = loader.train_loader(base_dir)
valid_loader, validset = loader.valid_loader(base_dir)
fpath = curr_dir.joinpath('mynetwork_state_dict.pth')

net = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
criterion = nn.CrossEntropyLoss()
state_dict = net.state_dict()

def training(lr,epoch,log_freq):

    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    task.train_network(epoch, train_loader, valid_loader, net, criterion, optimizer, log_freq)  
    torch.save(state_dict, fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--lrate', type=float, help='learning rate', required=True)
    parser.add_argument('-ep','--epoch',type=int, help='epoch', required=True)
    parser.add_argument('-lf','--lfreq',type=int, help='log frequency', required=True)
    args = parser.parse_args()
    
    lr = args.lrate
    epoch = args.epoch
    log_freq = args.lfreq
    training(lr,epoch,log_freq)

