from math import e
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
from cifar.models.squeeze import SqueezeNet
from cifar.models.mobile import MobileNet
from cifar.models.mobilev2 import MobileNetV2
from cifar.trainer import task
from typing import *

import gc
# del variables
gc.collect()
torch.cuda.empty_cache()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--lrate', type=float, help='learning rate', required=True)
    parser.add_argument('-ep','--epoch',type=int, help='epoch', required=True)
    parser.add_argument('-lf','--lfreq',type=int, help='log frequency', required=True)
    parser.add_argument('-m','--momentum',type=float, help='momentum', default=0.9)
    parser.add_argument('-b','--bsize',type=int, help='batch size', default=32)
    parser.add_argument('-n','--num_worker',type=int, help='num worker', default=8)
    parser.add_argument('-o','--optimizer',type=str, help='optimizer', required=True)
    parser.add_argument('-dr','--decay_rate',type=float, help='decay rate', required=True)
    parser.add_argument('-sch','--scheduler',type=str, help='scheduler', required=True)
    parser.add_argument('--net',type=str, help='net', required=True)
    # parser.add_argument('--layer',type=str, help='residual net layer')
    parser.add_argument('-tr','--train_resized',type=int, help='train resized', default=64)
    parser.add_argument('-vr','--valid_resized',type=int, help='valid resized', default=64)
    parser.add_argument('-ro','--train_rotate',type=int, help='train rotate', default=30)
    args = parser.parse_args()
    
    

    LR = args.lrate
    EPOCHS = args.epoch
    LOG_FREQ = args.lfreq
    MOMENTUM = args.momentum
    
    BSIZE = args.bsize
    NUM_WORKER = args.num_worker
    NET = args.net
    if NET == 'conv':
        NET = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
    elif NET == 'sq':
        NET = SqueezeNet(10).to(task.device)
    elif NET == 'mb':
        NET = MobileNet(1000).to(task.device)
    elif NET == 'mb2':
        NET = MobileNetV2().to(task.device)
    # elif NET == 'res':
    #     if args.layer == '18':
    #         NET = resnet18()

    #     elif args.layer == '34':
    #         NET = resnet34()

    #     elif args.layer == '50':
    #         NET = resnet50()

    #     elif args.layer == '101':
    #         NET = resnet101()

    #     else:
    #         NET = resnet152()    

    else:
        print("net kau mana")
    
    OPTIMIZER = args.optimizer
    if OPTIMIZER == 'sgd':
        OPTIMIZER = optim.SGD(NET.parameters(), LR, MOMENTUM)
    elif OPTIMIZER == 'adam':
        OPTIMIZER = optim.Adam(NET.parameters(), LR)
    elif OPTIMIZER == 'rms':
        OPTIMIZER = optim.RMSprop(NET.parameters(), LR, MOMENTUM, weight_decay=0.00004)
    else:
        print("error cuy")
    save_path = 'lr{}_ep{}_opt{}_net{}'.format(LR,EPOCHS,args.optimizer,args.net)
    DECAY = args.decay_rate

    SCHEDULER = args.scheduler
    if SCHEDULER == 'exp':
        SCHEDULER = optim.lr_scheduler.ExponentialLR(optimizer=OPTIMIZER, gamma=DECAY)
    elif SCHEDULER == 'step':
        SCHEDULER = optim.lr_scheduler.StepLR(optimizer=OPTIMIZER, step_size=30, gamma=0.1)
    elif SCHEDULER == 'mul':
        SCHEDULER = optim.lr_scheduler.MultiStepLR(optimizer=OPTIMIZER, milestones=[30,80], gamma=0.1)
    
    TR = args.train_resized
    VR = args.valid_resized
    RO = args.train_rotate

    curr_dir = Path(curr_dir)
    base_dir = curr_dir.joinpath('dataset')
    train_loader, trainset = loader.train_loader(base_dir, BSIZE, NUM_WORKER, RO, TR)
    valid_loader, validset = loader.valid_loader(base_dir, BSIZE, NUM_WORKER, VR)
    fpath = curr_dir.joinpath(save_path)

    
    criterion = nn.CrossEntropyLoss()
    state_dict = NET.state_dict()
    # print(OPTIMIZER)
    task.train_network(EPOCHS, train_loader, valid_loader, NET, criterion, OPTIMIZER, SCHEDULER, LOG_FREQ)  
    torch.save(state_dict, fpath)
