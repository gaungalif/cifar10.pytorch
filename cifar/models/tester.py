from torch import nn
from typing import *
n: List[nn.Module] = [1,2,3,4,3,3,1]
for i in n:
    for j in range(i):
        print(j)
    print('next')