import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(self, features_in):
        super(RPN, self).__init__()
        
        self.conv = nn.Conv2d(features_in, 1, 1)


    def forward(self, x):

        x = self.conv(x)
        
        return torch.sigmoid(x)

class DN(nn.Module):
    def __init__(self, features_in, n_class):
        super(DN, self).__init__()
        
#         self.conv1 = nn.Conv2d(features_in, features_in, 5, padding=2)
#         self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(features_in, n_class, 1)


    def forward(self, x):

#         x = self.conv1(x)
#         x = self.relu1(x)
        x = self.conv2(x)
        
        return torch.softmax(x, dim=1)