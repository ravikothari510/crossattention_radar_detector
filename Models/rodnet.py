import torch.nn as nn
import torch

from backbone import Encode, Decode

'''
It'S a 2D CNN implemenation of RodNet-CDC 
Based on the paper "RODNet: Radar Object Detection using Cross-Modal Supervision"
'''



class RODNetCDC(nn.Module):
    def __init__(self, in_channels=1, n_class=3):
        super(RODNetCDC,self).__init__()

        self.raencode = Encode(in_channels=in_channels)  
        self.radecode = Decode(out_channel=n_class)

    def forward(self, ra):
        ra_e = self.raencode(ra)
        return self.radecode(ra_e)


