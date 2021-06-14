from train import *

net = MyNetwork(ichan=3, clazz=10, imsize=(64,64))
loaded_state_dict = torch.load(fpath, map_location='gpu')
net.load_state_dict(loaded_state_dict)
