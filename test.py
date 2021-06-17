import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

from cifar.models.models import MyNetwork

import train
import torch

net = MyNetwork(ichan=3, clazz=10, imsize=(64,64))
loaded_state_dict = torch.load(train.curr_dir.joinpath('gaung.pth'), map_location='cpu')
net.load_state_dict(loaded_state_dict)

idx = 0

imgs, lbls = next(iter(train.valid_loader))
out = net(imgs)
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