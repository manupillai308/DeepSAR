import torch
import torch.nn as nn
import math

class Upsampling(nn.Module):
    '''
    The Upsampling class that implements the 'Upsampling Stage' performed of Bs individually

    Args:
        features_in (int): The no. of channels for Bi or Mi feature maps (256 according to implementation)
        features_out (int): The no. of channels for Pb or Pm (128 according to implementation)
        norm_fn (nn.Module): Normalization layer for Upsampling Stage (GroupNorm according to 
                             manuscript, but BatchNorm according to implementation)
        out_map_stride (int): The factor by which the input image is reduced with respect to the size of Bi
        
    Attributes:
        times (int): No. of times the upsampling stage is to be done according to the 'out_map_stride'
        n_layers (int): No. of layer for upsampling stage based on 'times'
        upsampling (nn.Sequential): Complete Upsampling layer arranged according to the n_layers
    '''
    def __init__(self, out_map_stride, features_out, features_in, norm_fn=nn.BatchNorm2d):
        super(Upsampling, self).__init__()
        

        times = int(math.log2(out_map_stride) - 2)
        ## if the total no. of times to perform upsample is 0, just simply perform the convolution, norm and activation without resizing x2
        n_layers = times if times != 0 else 1 
        
        self.upsampling = nn.Sequential(*[
        nn.Sequential(
            nn.Conv2d(features_in if idx == 0 else features_out, features_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_fn(num_features=features_out),
            nn.ReLU(inplace=True), #to save memory
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if times != 0 else nn.Identity(),
            # this upsample is different from 'Upsampling Stage'
        )
        for idx in range(n_layers)])
        
    def forward(self, x):
        x = self.upsampling(x)
        return x