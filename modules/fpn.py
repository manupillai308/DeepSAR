import torch
import torch.nn as nn

class FPN(nn.Module):
    '''
    The FPN class that takes [F1, F2, F3, F4] and calculates [B4, B3, B2, B1]

    Args:
        features_out (int): The no. of channels for Bi or Mi feature maps (256 according to implementation)
        features_in (int): The list of no. of channels (as list of int) in F1, F2, F3 and F4 respectively 
                           ([256, 512, 1024, 2048] for resnet50)
        
    Attributes:
        conv1 (nn.ModuleList): The list of convolution modules for calculating (1x1) convolution of each 
                               F1, F2, F3, F4 separately
        conv2 (nn.ModuleList): The list of convolution modules for calculating (3x3) convolution to create
                               B1, B2, B3, B4 individually
        feature_maps (int): The total number of feature maps (4 in case of [F1, F2, F3, F4]) 
    '''
    def __init__(self, features_out, features_in):
        super(FPN, self).__init__()
        
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.feature_maps = len(features_in)
        for i in range(self.feature_maps):
            self.conv1.append(nn.Conv2d(features_in[i], features_out, 1))
            self.conv2.append(nn.Conv2d(features_out, features_out, 3, padding=1))
        
    def forward(self, x):
        assert self.feature_maps == len(x), f"Shape mismatch, expected input feature maps {self.features_maps}, got {len(x)}"
        
        Bs = []
        F_prev = self.conv1[-1](x[-1])
        B = self.conv2[-1](F_prev)
        
        Bs.append(B)
        
        for i in range(self.feature_maps-2, -1, -1):
            F = self.conv1[i](x[i])
            F_prev = nn.functional.interpolate(F_prev, scale_factor=2, mode="nearest")
            F_prev += F
            B = self.conv2[i](F_prev)
            Bs.insert(0, B)
        
        assert self.feature_maps == len(Bs), f"Size mismatch, input feature maps are {self.feature_maps}, output feature maps are {len(Bs)}"
        
        return Bs ## [B4, B3, B2, B1]
   