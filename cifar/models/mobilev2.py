import torch
from torch import nn
from torch import Tensor
from typing import *
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBnRelu(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, 
                stride=1, groups=1, bias=True, 
                norm_layer=True, activation=True):
        super(ConvBnRelu, self).__init__()
        padding = (kernel_size - 1) // 2 * 1
        self.norm_layer = norm_layer
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=bias)
        if self.norm_layer:
            self.norm_layer = nn.BatchNorm2d(outplanes)
        
        if self.activation:
            self.activation = nn.ReLU6(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer:
            x = self.norm_layer(x)
        if self.activation:
            x = self.activation(x)

        return x

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=1, bias=True):
        super(SeparableConv2d, self).__init__()
        hidden_dim = inplanes

        self.depthwise = ConvBnRelu(inplanes, hidden_dim, 
                                    stride=stride, groups=groups, bias=bias)

        self.pointwise = ConvBnRelu(hidden_dim, outplanes, 
                                    kernel_size=1, stride=1,  
                                    bias=False, activation=False)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)


        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBnRelu(inp, hidden_dim, kernel_size=1))
            
        layers.extend([
            SeparableConv2d(hidden_dim, oup, stride=stride, groups=hidden_dim),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult: float = 1.0,
                round_nearest: int = 8, num_classes: int =1000):
        super(MobileNetV2, self).__init__()
        in_planes = 32
        last_planes = 1280
        self.in_channel = _make_divisible(in_planes * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_planes * width_mult, round_nearest)

        self.num_classes = num_classes

        inverted_residual_setting = [
         #   t  c   n  s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = [ConvBnRelu(3, self.in_channel, stride=2)]
        # features

        for t, c, n, s in inverted_residual_setting:
            out_channel = c
            for num in range(n):
                strd = s
                if num>0: strd = 1
                features.append(InvertedResidual(inp=self.in_channel, oup=out_channel, 
                            stride=strd, expand_ratio=t))
                self.in_channel = out_channel
        features.append(ConvBnRelu(self.in_channel, self.last_channel, kernel_size=1))
        features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

# mob = MobileNetV2()
# x = torch.rand(3,3,224,224)
# x = mob(x)
# print(x.shape)
        
        
        


