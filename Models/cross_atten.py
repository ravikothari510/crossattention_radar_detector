import torch.nn as nn
import torch

from .backbone import Encode, Decode

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.norm = nn.LayerNorm([256,32,32])
    
    def forward(self, ra_e, rd_e, ad_e):
        x = torch.matmul(rd_e, ad_e) # B,256,32,32
        x = self.softmax(x.view(*x.size()[:2],-1)).view_as(x)
        x = ra_e*x
        x = self.norm(ra_e + x)
        return x




class RadarCross(nn.Module):
    def __init__(self, in_channels, n_class,
                center_offset=True, orentation=False):
        super(RadarCross, self).__init__()
        
        self.center_offset = center_offset
        self.orentation = orentation

        self.raencode = Encode(in_channels=in_channels)
        self.rdencode = Encode(in_channels=in_channels)
        self.adencode = Encode(in_channels=in_channels)
        self.decode = Decode()
        self.attention = CrossAttention()
        
        self.finalConv_cls = nn.Conv2d(in_channels=32, out_channels=n_class, stride=(1,1), padding=(1,1), kernel_size=(3,3))
        self.finalConv_center = nn.Conv2d(in_channels=32, out_channels=2, stride=(1,1), padding=(1,1), kernel_size=(3,3))
        self.finalConv_orent = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,stride=(2,2),padding=(2,2),kernel_size=(5,5)),
            nn.PReLU(),
            nn.Conv2d(in_channels=16,out_channels=2,stride=(2,2),padding=(2,2),kernel_size=(5,5)),
            nn.Tanh()
        )

    def forward(self, ra, rd, ad):
        ra_e = self.raencode(ra) # (B,1,256,256)-> (B,256,32,32)
        rd_e = self.rdencode(rd) # (B,1,256,64)-> (B,256,32,8)
        ad_e = self.adencode(ad)# (B,1,64,256)-> (B,256,8,32)

        ca = self.attention(ra_e=ra_e, rd_e=rd_e, ad_e=ad_e)

        x = self.decode(ca)

        x_cls = self.finalConv_cls(x) # (B,32,256,256) -> (B,3,256,256)
        x_center = 0
        x_orent = 0
        if self.center_offset:
            x_center = self.finalConv_center(x) # (B,32,256,256) -> (B,2,256,256)
            
        if self.orentation:
            x_orent = self.finalConv_orent(x) # (B,32,256,256) -> (B,2,64,64)

        return x_cls, x_center, x_orent
        
