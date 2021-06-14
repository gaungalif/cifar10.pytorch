from cifar.models.models import MyNetwork
import torch


net = MyNetwork(ichan=3, clazz=10, imsize=(64,64))
state_dict = net.state_dict()
fpath = '/kaggle/working/mynetwork_state_dict.pth'
loaded_state_dict = torch.load(fpath, map_location='gpu')
net.load_state_dict(loaded_state_dict)
