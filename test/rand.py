import torch
from torch import nn
from torch import Tensor
from typing import *


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, 
        in_planes: int, 
        out_planes: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        groups: int = 1, 
        norm_layer = None, 
        activation_layer = None, 
        dilation: int = 1,
    ) -> None:

        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, 
                        dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes
        
Convolution = ConvBNReLU

class LinearBottleneck(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion,
        norm_layer
    ) -> None:
        super(LinearBottleneck, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride ==1 and inp == oup
        
        layers: List[nn.Module] = []
        if expansion != 1:
            layers.append(Convolution(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            Convolution(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(
            self,
        num_classes: int = 1000,
        bottleneck_variable: Optional[List[List[int]]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MobileNetV2, self).__init__()

        if block is None:
            block = LinearBottleneck
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_channel = 32
        last_channel = 1280

        if bottleneck_variable is None:
            bottleneck_variable = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        self.last_channel = last_channel
        features: List[nn.Module] = [Convolution(3, in_channel, stride=2, norm_layer=norm_layer)]
        for t, c, n, s in  bottleneck_variable:
            out_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channel, out_channel, stride, expansion=t, norm_layer=norm_layer))
                in_channel = out_channel
        features.append(Convolution(in_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*features)


        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

x = torch.rand(3,3,224,224)
net = MobileNetV2()
net.train()
x = net(x)
print(x.shape)    
