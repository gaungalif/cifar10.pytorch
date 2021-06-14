import torch.nn as nn

class SingleConvLayer(nn.Module):
    def __init__(self, ichan, ochan, ksize=3):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=ichan, out_channels=ochan, kernel_size=ksize)
        self.bn = nn.BatchNorm2d(ochan)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
class DoubleConvLayer(nn.Module):
    def __init__(self, ichan, hchan, ochan, ksize=3):
        super(DoubleConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ichan, out_channels=hchan, kernel_size=ksize)
        self.bn1 = nn.BatchNorm2d(hchan)
        self.conv2 = nn.Conv2d(in_channels=hchan, out_channels=ochan, kernel_size=ksize)
        self.bn2 = nn.BatchNorm2d(ochan)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return x
class TripleConvLayer(nn.Module):
    def __init__(self, ichan, hchan, ochan, ksize=3):
        super(TripleConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ichan, out_channels=hchan, kernel_size=ksize)
        self.bn1 = nn.BatchNorm2d(hchan)
        self.conv2 = nn.Conv2d(in_channels=hchan, out_channels=hchan, kernel_size=ksize)
        self.bn2 = nn.BatchNorm2d(hchan)
        self.conv3 = nn.Conv2d(in_channels=hchan, out_channels=ochan, kernel_size=ksize)
        self.bn3 = nn.BatchNorm2d(ochan)
        self.pool = nn.MaxPool2d((2,2))
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class MyNetwork(nn.Module):
    def __init__(self, ichan, clazz, imsize=(224,224)):
        '''
        default imsize=((224,224))
        '''
        super(MyNetwork, self).__init__()
        self.imsize = imsize
        self.layer1 = SingleConvLayer(ichan=ichan, ochan=64)
        self.layer2 = SingleConvLayer(ichan=64, ochan=64)
        flatval = self._flatval(imsize)
        self.fc1 = nn.Linear(flatval, 512)
        self.fc2 = nn.Linear(512, clazz)
        self.flatten = Flatten()
    
    def _last_res(self, x):
        import math
        last = (x-6)/4
        last = math.floor(last)
        return last
    
    def _flatval(self, imsize):
        h,w = imsize
        hl,wl = self._last_res(h), self._last_res(w)
        flat = 64*hl*wl
        return flat
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x