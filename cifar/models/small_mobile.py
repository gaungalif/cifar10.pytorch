import torch
import torch.nn as nn



class Depthwise(nn.Module):
    def __init__(self, inplanes, outplanes, stride, padding=1,
                kernel_size = 3):
        super(Depthwise, self).__init__()
        self.inplanes = inplanes
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, groups=inplanes, padding=padding, stride=stride)
        self.pointwise = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size)
        self.depthwise = torch.nn.Sequential(self.depthwise, self.pointwise)
        self.conv = nn.Conv2d(outplanes, outplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SingleConvLayer(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=1, stride=1,padding=1):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SmallMobile(nn.Module):
    def __init__(self, num_classes=1000):
        super(SmallMobile, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            SingleConvLayer(3, 32, kernel_size=3, stride=2),
            Depthwise(32, 32,stride=1),
            SingleConvLayer(32, 64, kernel_size=1, stride=1),
            Depthwise(64, 64,stride=2),         
            SingleConvLayer(64, 128, kernel_size=1, stride=1),
            Depthwise(128, 128, stride=1),
            SingleConvLayer(128, 128, kernel_size=1, stride=1),
            Depthwise(128, 128, stride=2),
            SingleConvLayer(128, 256, kernel_size=1, stride=1),
            Depthwise(256, 256, stride=1),
            SingleConvLayer(256, 256, kernel_size=1, stride=1),           
            Depthwise(256, 256, stride=2),
            SingleConvLayer(256, 512, kernel_size=1, stride=1),
            # #5 times          
            Depthwise(512, 512, stride=1),
            SingleConvLayer(512, 512, kernel_size=1, stride=1),
            Depthwise(512, 512, stride=1),
            SingleConvLayer(512, 512, kernel_size=1, stride=1),
            Depthwise(512, 512, stride=1),
            SingleConvLayer(512, 512, kernel_size=1, stride=1),
            Depthwise(512, 512, stride=1),
            SingleConvLayer(512, 512, kernel_size=1, stride=1),
            Depthwise(512, 512, stride=1),
            SingleConvLayer(512, 512, kernel_size=1, stride=1),
            # #end 5 times
            Depthwise(512, 512, stride=2,padding=3),
            SingleConvLayer(512, 1024, kernel_size=1, stride=1),
            Depthwise(1024, 1024, stride=2,padding=2),
            SingleConvLayer(1024, 1024, kernel_size=1, stride=1, padding=3),
            nn.AvgPool2d(7,7),
        )

        self.classifier = nn.Linear(1024,1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

