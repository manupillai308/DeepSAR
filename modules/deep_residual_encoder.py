import torch
import torch.nn as nn

class DeepResidualEncoder(nn.Module):
    '''
    The Deep Residual Encoder class that generates the feature maps F1, F2, F3 and F4

    Args:
        backbone (nn.Module): The backbone network to use to extract feature maps
        layer_no (int): The index of the layer according to '.children()' method of backbone
                        that precedes the F1 feature map
        layer_names (list(str)): The names of the layer (as a list of strings) that corresponds
                                 F1, F2, F3 and F4. 

    Attributes:
        pre_encoder (nn.Sequential): The layer preceding the F1 layer. It is used to extract input 
                                     to layers to computer F1, F2, F3 and F4
        fmaps (nn.ModuleList): Layers that computes the feature maps F1, F2, F3 and F4
    '''
    def __init__(self, backbone, layer_no, layer_names):
        super(DeepResidualEncoder, self).__init__()
        self.pre_encoder = nn.Sequential(*list(backbone.children())[:layer_no])
        self.fmaps = nn.ModuleList([getattr(backbone, layer_name) for layer_name in layer_names])
        
    def forward(self, x):
        Fs = []
        x = self.pre_encoder(x)
        for layer in self.fmaps:
            x = layer(x)
            Fs.append(x)
        
        return Fs ## [F1, F2, F3, F4]