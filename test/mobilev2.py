import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import *



def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(
        self, 
        in_planes: int, 
        out_planes: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        groups: int = 1, 
        norm_layer:  Optional[Callable[..., nn.Module]] = None, 
        activation_layer: Optional[Callable[..., nn.Module]] = None, 
        dilation: int = 1
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
            super().__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, groups=groups),
                norm_layer(out_planes),
                activation_layer(inplace=True),
            )
# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        expansion: int,
        norm_layer:  Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(LinearBottleneck, self).__init__()
        self.stride = stride 
        assert stride in [1, 2]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(in_planes * expansion))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers: List[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBNReLU(in_planes, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            #depthwise
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            #pointwise-linear
            nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
            norm_layer(out_planes),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_planes
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)  
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        bottleneck_variable: Optional[List[List[int]]] = None,
        k_channel = 1280,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(MobileNetV2, self).__init__()
        
        if block is None:
            block = LinearBottleneck
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if bottleneck_variable is None:
            bottleneck_variable = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ]

        in_channel = 32
        
        self.k_channel = k_channel
        features = [ConvBNReLU(3, in_channel, stride=2, norm_layer=norm_layer)]
        for t, c, n, s in bottleneck_variable:
            out_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channel, out_channel, stride, expansion=t))
                in_channel = out_channel
        features.append(ConvBNReLU(in_channel, self.k_channel, kernel_size=1, norm_layer=norm_layer))
        # features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.k_channel, num_classes),
        )

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
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

x = torch.rand(3,3,224,224)
net = MobileNetV2()
net.train()
x = net(x)
print(x.shape)    
