import torch
import torch.nn as nn
from modules.deep_residual_encoder import DeepResidualEncoder
from modules.dual_branch_decoder import DualBranchDecoder


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        
        self.dr_encoder = DeepResidualEncoder(**config["dr_encoder"])
        self.db_decoder_FA = DualBranchDecoder(**config["db_decoder_FA"])
    
    
    def forward(self, x):
        x = self.dr_encoder(x) # calculate F1, F2, F3, F4
        
        fa = self.db_decoder_FA(x) # calculate Pb
        
        return fa