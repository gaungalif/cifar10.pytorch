import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

from cifar.models.models import MyNetwork
from train import *
import torch

from pathlib import Path




net = MyNetwork(ichan=3, clazz=10, imsize=(64,64))
state_dict = net.state_dict()
# fpath = '/kaggle/working/mynetwork_state_dict.pth'
loaded_state_dict = torch.load(Path(curr_dir).joinpath('mynetwork_state_dict.pth'), map_location='cpu')
net.load_state_dict(loaded_state_dict)

idx = 0

imgs, lbls = next(iter(valid_loader))
out = net(imgs)
out = torch.softmax(out, dim=0)
out = torch.argmax(out, dim=1)
# torch.argmax(out[0])
for idx in len(out):
    print('data asli= ', out[idx],' data hasil =', lbls[idx])
