import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

from cifar.models.models import MyNetwork
from cifar.models.squeeze import SqueezeNet

from pathlib import Path
import argparse

from cifar.datasets import loader

import torch

parser = argparse.ArgumentParser()
parser.add_argument('-sp','--save_path',type=str, help='save path', required=True)
parser.add_argument('--net',type=str, help='net', required=True)
args = parser.parse_args()

save_path = args.save_path
NET = args.net
if NET == 'sq':
    NET = SqueezeNet(10) 
elif NET == 'conv':
    NET = MyNetwork(ichan=3, clazz=10, imsize=(64,64))
else:
    print("false net")

curr_dir = Path(curr_dir)
base_dir = curr_dir.joinpath('dataset')
train_loader, trainset = loader.train_loader(base_dir)
valid_loader, validset = loader.valid_loader(base_dir)
fpath = curr_dir.joinpath(save_path)


loaded_state_dict = torch.load(fpath, map_location='cpu')
NET.load_state_dict(loaded_state_dict)

idx = 0

imgs, lbls = next(iter(valid_loader))
out = NET(imgs)
out = torch.softmax(out, dim=0)
out = torch.argmax(out, dim=1)
# torch.argmax(out[0])
true = 0
false = 0
for idx in range(len(out)):
    print(out[idx]==lbls[idx])
    if(out[idx]==lbls[idx]):
        true+=1
    else:
        false+=1
print('true = {}'.format(true))
print('false = {}'.format(false))