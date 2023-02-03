from albumentations.augmentations.crops.functional import keypoint_random_crop
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF

class Encode(nn.Module):
    def __init__(self,in_channels):
        super(Encode,self).__init__()

        self.conv1a = nn.Conv2d(in_channels,64,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.conv1b = nn.Conv2d(in_channels =64,out_channels=64,kernel_size=(5,5),stride=(2,2),padding=(2,2))
        
        self.conv2a = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.conv2b = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(5,5),stride=(2,2),padding=(2,2))

        self.conv3a = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.conv3b = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(5,5),stride=(2,2),padding=(2,2))

        self.bn1a = nn.BatchNorm2d(num_features=64)
        self.bn1b = nn.BatchNorm2d(num_features=64)
        self.bn2a = nn.BatchNorm2d(num_features=128)
        self.bn2b = nn.BatchNorm2d(num_features=128)
        self.bn3a = nn.BatchNorm2d(num_features=256)
        self.bn3b = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, 256, 256) -> (B, 64,  256, 256)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, 256, 256) -> (B, 64, 128, 128)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, 128, 128) -> (B, 128, 128, 128)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, 128, 128) -> (B, 128, 64, 64)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, 64, 64) -> (B, 256, 64, 64 )
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, 64, 64) -> (B, 256, 32, 32)

        return x    

class Decode(nn.Module):
    def __init__(self):
        super(Decode,self).__init__()

  
        self.convt1 = nn.ConvTranspose2d(in_channels=256,out_channels=128,stride=(2,2),padding=(2,2),kernel_size=(6,6))
        self.convt2 = nn.ConvTranspose2d(in_channels=128,out_channels=64,stride=(2,2),padding=(2,2),kernel_size=(6,6))
        self.convt3 = nn.ConvTranspose2d(in_channels=64,out_channels=32,stride=(2,2),padding=(2,2),kernel_size=(6,6))

        self.bn_cn = nn.BatchNorm2d(num_features=2)
        self.PRelu = nn.PReLU()

    def forward(self,x):
        x = self.PRelu(self.convt1(x)) # (B,256,32,32) -> (B,128,64,64)
        x = self.PRelu(self.convt2(x)) # (B,128,64,64) -> (B,64,128,128)   
        x = self.PRelu(self.convt3(x)) # (B,64,128,128) -> (B,32,256,256)


        return x

class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
        #self.attention_conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1))
        self.attention_mlp = nn.Sequential(nn.Linear(in_features=512,out_features=256,bias=True),
                             nn.PReLU(),
                             nn.Linear(in_features=256,out_features=512,bias=True))

        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        max_pool = torch.amax(x,dim=(2,3))
        avg_pool = torch.mean(x,dim=(2,3))
        max_pool = self.attention_mlp(max_pool)
        avg_pool = self.attention_mlp(avg_pool)

        chn_atten = self.sigmoid(max_pool+avg_pool)
        chn_atten = chn_atten.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        return x*chn_atten

def blackout(ra_map,rd_map,ad_map):
    n = torch.abs(torch.randn(1))
    if n > 0.3 :return ra_map,rd_map,ad_map
    elif n < 0.1:return 0*ra_map,rd_map,ad_map
    elif n > 0.1 and n < 0.2 : return ra_map,0*rd_map,ad_map
    else : return ra_map,rd_map,0*ad_map




class RODnet(nn.Module):
    def __init__(self,in_channels,n_class):
        super(RODnet,self).__init__()

        self.raencode = Encode(in_channels=in_channels)
        self.decode = Decode()
        
        self.rdencode = Encode(in_channels=in_channels)
        
        self.adencode = Encode(in_channels=in_channels)

        #self.compress_conv_rd = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,64))
        #self.expand_conv_rd = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(1,4),stride=(1,4))
        self.expand_conv_rd = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(1,1),stride=(1,1))

        #self.compress_conv_ad = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(64,1))
        #self.expand_conv_ad = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,1),stride=(4,1))
        self.expand_conv_ad = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(1,1),stride=(1,1))

        self.attention = attention()
        self.softmax = nn.Softmax(dim=(2,3))
        self.norm = nn.LayerNorm([256,32,32])
        self.compress = nn.Conv2d(in_channels=512, out_channels=256, stride=(1,1),padding=(1,1),kernel_size=(3,3))
        self.inception1 = nn.Conv2d(in_channels=512,out_channels=256,stride=(1,1),padding=(1,1),kernel_size=(3,3))
        self.inception2 = nn.Conv2d(in_channels=256,out_channels=128,stride=(1,1),padding=(2,2),kernel_size=(5,5))
        self.inception3 = nn.Conv2d(in_channels=128,out_channels=64,stride=(1,1),padding=(0,10),kernel_size=(1,21))

        self.PRelu = nn.PReLU()



        self.finalConv_cls = nn.Conv2d(in_channels=32,out_channels=n_class,stride=(1,1),padding=(1,1),kernel_size=(3,3))
        self.finalConv_center = nn.Conv2d(in_channels=32,out_channels=2,stride=(1,1),padding=(1,1),kernel_size=(3,3))


    def forward(self,ra,rd,ad):
        ra = self.raencode(ra) # (B,1,256,256)-> (B,256,32,32)
        #ra_attention = self.attention(ra)
    

        rd = self.rdencode(rd) # (B,1,256,64)-> (B,256,32,8)
        #rd = self.PRelu(self.expand_conv_rd(rd)) #(B,128,32,8)
        #rd = torch.amax(rd,dim=3,keepdim=True)
        #rd = torch.tile(rd,(1,1,1,32))

        # rd = self.PRelu(self.compress_conv_rd(rd))
        # rd = self.PRelu(self.expand_conv_rd(rd))
        #rd = torch.matmul(rd_1,ra_attention)
        
        ad = self.adencode(ad)# (B,1,64,256)-> (B,256,8,32)
        #ad = self.PRelu(self.expand_conv_ad(ad)) #(B,128,8,32)
        #ad = torch.amax(ad,dim=2,keepdim=True)
        #ad = torch.tile(ad,(1,1,32,1))
        # ad = self.PRelu(self.compress_conv_ad(ad))
        #ad = self.PRelu(self.expand_conv_ad(ad))
        #ad = torch.matmul(ad_1,ra_attention)

        # if self.training: 
        #     ra,rd,ad = blackout(ra,rd,ad)
        x = torch.matmul(rd,ad) # B,256,32,32
        x = ra*(nn.Softmax(2)(x.view(*x.size()[:2],-1)).view_as(x))
        x = self.norm(ra + x)

        #x =torch.cat([ra,rd,ad],dim=1)
        #x = self.attention(x)

        #x = self.PRelu(self.compress(x))
        x = self.decode(x) # (B,32,256,256)

        #x = self.PRelu(self.inception1(x)) # (B,96,256,256) -> (B,64,256,256)
        #x = self.PRelu(self.inception2(x)) # (B,64,256,256) -> (B,64,256,256)
        #x = self.PRelu(self.inception3(x)) # (B,64,256,256) -> (B,64,256,256)
    
        x_cls = self.finalConv_cls(x) # (B,32,256,256) -> (B,3,256,256)
        x_center = self.finalConv_center(x) # (B,32,256,256) -> (B,2,256,256)
        #ra,rd_1,ad_1,ra_attention,rd_b,ad_b
        

        return x_cls,x_center



def test():
    ra= torch.randn(5,1,256,256)
    rd= torch.randn(5,1,256,64)
    ad= torch.randn(5,1,64,256)
    model = RODnet(in_channels=1,n_class=3)
    out_cls,out_center = model(ra,rd,ad)
    print(out_cls.shape,out_center.shape,ra.shape,rd.shape,ad.shape)

if __name__=="__main__":
    test()