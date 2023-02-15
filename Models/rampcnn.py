import torch.nn as nn
import torch
from .backbone import Encode, Decode

'''
It's a 2D implementation of RAMP CNN with the inception module
'''



class RAMPCNN(nn.Module):
    def __init__(self, in_channels=1, n_class=3):
        super(RAMPCNN, self).__init__()

        self.raencode = Encode(in_channels=in_channels)
        self.rdencode = Encode(in_channels=in_channels)
        self.adencode = Encode(in_channels=in_channels)
        
        self.radecode = Decode()
        self.rddecode = Decode()
        self.addecode = Decode()

        self.inception1 = nn.Conv2d(in_channels=96, out_channels=64, stride=(1,1), padding=(1,1), kernel_size=(3,3))
        self.inception2 = nn.Conv2d(in_channels=64, out_channels=32, stride=(1,1), padding=(2,2), kernel_size=(5,5))
        self.inception3 = nn.Conv2d(in_channels=32, out_channels=n_class, stride=(1,1), padding=(0,10), kernel_size=(1,21))

        self.PRelu = nn.PReLU()


    def forward(self,ra,rd,ad):
        ra = self.raencode(ra) #(B,1,256,256)-> (B,256,32,32)
        ra = self.radecode(ra) #(B,256,32,32)->(B,32,256,256)
       
        rd = self.rdencode(rd) #(B,1,256,64)-> (B,256,32,8)
        rd = self.rddecode(rd) #(B,256,32,32)->(B,32,256,256)
        
        ad = self.adencode(ad) #(B,1,64,256)-> (B,256,8,32)
        ad = self.addecode(ad) #(B,256,32,32)->(B,32,256,256)

        # Fusion Module
        rd = torch.mean(rd,dim=3, keepdim=True)
        rd = torch.tile(rd, (1,1,1,256))

        ad = torch.mean(ad,dim=2, keepdim=True)
        ad = torch.tile(ad, (1,1,256,1))
        
        x = torch.cat([ra,rd,ad],dim=1)

        #Inception module
        x = self.PRelu(self.inception1(x)) # (B,96,256,256) -> (B,64,256,256)
        x = self.PRelu(self.inception2(x)) # (B,64,256,256) -> (B,64,256,256)
        x = self.PRelu(self.inception3(x)) # (B,64,256,256) -> (B,3,256,256)
        return x