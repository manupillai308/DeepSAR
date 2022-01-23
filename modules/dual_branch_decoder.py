import torch
import torch.nn as nn
from modules.fpn import FPN
from modules.upsampling_stage import Upsampling


class DualBranchDecoder(nn.Module):
    def __init__(self, fpn_features_out, dbd_features_out, features_in, strides):
        super(DualBranchDecoder, self).__init__()
        
        self.fpn = FPN(features_out=fpn_features_out, features_in=features_in)
        self.upsamplings = nn.ModuleList([
            Upsampling(out_map_stride=stride, features_out=dbd_features_out, features_in=fpn_features_out)
            for stride in strides])
        # self.conv = nn.Conv2d(dbd_features_out, nfeats, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        
    
    def forward(self, x):
        Bs = self.fpn(x)
        
        feats = 0
        for i in range(len(self.upsamplings)):
            feats += self.upsamplings[i](Bs[i]) ## Summation of Bs after the respective Upsampling stage.
    
        # feats = self.conv(feats) # Conv 1x1
        feats = self.upsample(feats) # 4x upsampling (Note: This upsampling is different from Upsampling stage)
        
        return feats
