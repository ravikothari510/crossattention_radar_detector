import torch.nn as nn
import torch


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
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 1, 256, 256) -> (B, 64,  256, 256)
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


class RODnet(nn.Module):
    def __init__(self,in_channels,n_class):
        super(RODnet,self).__init__()

        self.raencode = Encode(in_channels=in_channels)
        self.rdencode = Encode(in_channels=in_channels)
        self.adencode = Encode(in_channels=in_channels)
        
        self.radecode = Decode()
        self.rddecode = Decode()
        self.addecode = Decode()

        self.inception1 = nn.Conv2d(in_channels=96,out_channels=64,stride=(1,1),padding=(1,1),kernel_size=(3,3))
        self.inception2 = nn.Conv2d(in_channels=64,out_channels=32,stride=(1,1),padding=(2,2),kernel_size=(5,5))
        self.inception3 = nn.Conv2d(in_channels=32,out_channels=3,stride=(1,1),padding=(0,10),kernel_size=(1,21))

        self.PRelu = nn.PReLU()


    def forward(self,ra,rd,ad):
        ra = self.raencode(ra) # (B,1,256,256)-> (B,256,32,32)
        ra = self.radecode(ra) #(B,256,32,32)->(B,32,256,256)
       
    

        rd = self.rdencode(rd) # (B,1,256,64)-> (B,256,32,8)
        rd = self.rddecode(rd) #(B,256,32,32)->(B,32,256,256)
        
        ad = self.adencode(ad)# (B,1,64,256)-> (B,256,8,32)
        ad = self.addecode(ad) #(B,256,32,32)->(B,32,256,256)

        # Fusion Module
        rd = torch.mean(rd,dim=3,keepdim=True)
        rd = torch.tile(rd,(1,1,1,256))

        ad = torch.mean(ad,dim=2,keepdim=True)
        ad = torch.tile(ad,(1,1,256,1))
        

        x = torch.cat([ra,rd,ad],dim=1)

        #Inception module
        x = self.PRelu(self.inception1(x)) # (B,96,256,256) -> (B,64,256,256)
        x = self.PRelu(self.inception2(x)) # (B,64,256,256) -> (B,64,256,256)
        x = self.PRelu(self.inception3(x)) # (B,64,256,256) -> (B,64,256,256)
        return x



def test():
    ra= torch.randn(5,1,256,256)
    rd= torch.randn(5,1,256,64)
    ad= torch.randn(5,1,64,256)
    model = RODnet(in_channels=1,n_class=3)
    x= model(ra,rd,ad)
    print(x.shape)

if __name__=="__main__":
    test()