from collections import defaultdict
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
    parser.add_argument('-b','--bsize',type=int, help='batch size', default=32)
    parser.add_argument('-n','--num_worker',type=int, help='num worker', default=8)
    parser.add_argument('-sp','--save_path',type=str, help='save path', required=True)
    args = parser.parse_args()
    
    LR = args.lrate
    EPOCHS = args.epoch
    LOG_FREQ = args.lfreq
    MOMENTUM = args.momentum
    
    BSIZE = args.bsize
    NUM_WORKER = args.num_worker
    save_path = args.save_path

    curr_dir = Path(curr_dir)
    base_dir = curr_dir.joinpath('dataset')
    train_loader, trainset = loader.train_loader(base_dir, BSIZE, NUM_WORKER)
    valid_loader, validset = loader.valid_loader(base_dir, BSIZE, NUM_WORKER)
    fpath = curr_dir.joinpath(save_path)

    net = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
    criterion = nn.CrossEntropyLoss()
    state_dict = net.state_dict()

    optimizer = optim.SGD(net.parameters(), LR, MOMENTUM)
    task.train_network(EPOCHS, train_loader, valid_loader, net, criterion, optimizer, LOG_FREQ)  
    torch.save(state_dict, fpath)
