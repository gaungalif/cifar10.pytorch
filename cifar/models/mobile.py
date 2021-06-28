import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d



class depthwise(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1, padding=1, bias=False):
        super(depthwise, self).__init__()
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=padding, groups=inplanes, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        return out

class DepthConv(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1,padding=1, bias=False):
        super(DepthConv, self).__init__()
        self.dw_block = nn.Sequential(
            depthwise(inplanes, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.dw_block(x)
        return out

class OneConv(nn.Module):
    def __init__(self, inplanes, outplanes, padding=0, kernel_size=1, stride=1, bias=False):
        super(OneConv, self).__init__()
        self.one_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias ),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.one_conv(x)
        return out


class ThreeConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False):
        super(ThreeConv, self).__init__()
        self.three_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.three_conv(x)
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            ThreeConv(3, 32),
            DepthConv(32, stride=1),
            OneConv(32, 32, kernel_size=3, padding=1),
            OneConv(32, 64),
            DepthConv(64, stride=2),  
            OneConv(64, 64, kernel_size=3, padding=1),
            OneConv(64, 128),       
            DepthConv(128, stride=1),
            OneConv(128, 128, kernel_size=3, padding=1),
            OneConv(128, 128), 
            DepthConv(128, stride=2),
            OneConv(128, 128, kernel_size=3, padding=1),
            OneConv(128, 256),
            DepthConv(256, stride=1),
            OneConv(256, 256, kernel_size=3, padding=1),
            OneConv(256, 256),
            DepthConv(256, stride=2),
            OneConv(256, 256, kernel_size=3, padding=1),
            OneConv(256, 512),
            #5
            DepthConv(512, stride=1),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 512),
            DepthConv(512, stride=1),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 512),
            DepthConv(512, stride=1),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 512),
            DepthConv(512, stride=1),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 512),
            DepthConv(512, stride=1),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 512),
            #5 end
            DepthConv(512, stride=2),
            OneConv(512, 512, kernel_size=3, padding=1),
            OneConv(512, 1024),
            DepthConv(1024, stride=2, padding=4),
            OneConv(1024, 1024, kernel_size=3, padding=1),
            OneConv(1024, 1024),
            nn.AvgPool2d(7,7),
        )

        self.classifier = nn.Linear(1024,1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
# x = torch.rand(3,3,224,224)
# net = MobileNet()
# x = net(x)
# print(x.shape)