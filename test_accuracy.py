import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

from cifar.models.models import MyNetwork
from cifar.models.squeeze import SqueezeNet
from cifar.models.mobile import MobileNet
from cifar.models.mobilev2 import MobileNetV2
from cifar.trainer import task

from pathlib import Path
import argparse

from cifar.datasets import loader

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-sp','--save_path',type=str, help='save path', required=True)
parser.add_argument('--net',type=str, help='net', required=True)
parser.add_argument('-b','--bsize',type=int, help='batch size', default=32)
parser.add_argument('-n','--num_worker',type=int, help='num worker', default=8)
parser.add_argument('-tr','--train_resized',type=int, help='train resized', default=64)
parser.add_argument('-vr','--valid_resized',type=int, help='valid resized', default=64)
parser.add_argument('-ro','--train_rotate',type=int, help='train rotate', default=30)


args = parser.parse_args()

BSIZE = args.bsize
NUM_WORKER = args.num_worker
TR = args.train_resized
VR = args.valid_resized
RO = args.train_rotate

save_path = args.save_path

NET = args.net
if NET == 'conv':
    NET = MyNetwork(ichan=3, clazz=10, imsize=(64,64)).to(task.device)
elif NET == 'sq':
    NET = SqueezeNet(10).to(task.device)
elif NET == 'mb':
    NET = MobileNet(1000).to(task.device)
elif NET == 'mb2':
    NET = MobileNetV2(1000).to(task.device)
else:
    print("false net")

curr_dir = Path(curr_dir)
base_dir = curr_dir.joinpath('dataset')
train_loader, trainset = loader.train_loader(base_dir, BSIZE, NUM_WORKER, RO, TR)
valid_loader, validset = loader.valid_loader(base_dir, BSIZE, NUM_WORKER, VR)
fpath = curr_dir.joinpath(save_path)


loaded_state_dict = torch.load(fpath, map_location='cpu')
NET.load_state_dict(loaded_state_dict)

idx = 0

imgs, lbls = next(iter(valid_loader))
imgs, lbls = imgs.to(device), lbls.to(device)
out = NET(imgs)
# out = torch.softmax(out, dim=0)
out = torch.argmax(out, dim=0)

true = 0
false = 0
for idx, data in range(len(out)):
    print(out[idx]==lbls[idx])
    if(out[idx]==lbls[idx]):
        true+=1
    else:
        false+=1
print('true = {}'.format(true))
print('false = {}'.format(false))