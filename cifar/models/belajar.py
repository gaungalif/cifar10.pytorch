import torch
import torch.nn as nn

x = torch.rand(3,3,16,16)
conv = nn.Conv2d(3,32,7)
bn = nn.BatchNorm2d(16)
relu = nn.ReLU(inplace=True)
x = conv(x)
x = bn(x)
print(x.shape)