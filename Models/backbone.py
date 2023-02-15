import torch.nn as nn
import torch

class Encode(nn.Module):
    def __init__(self,in_channels):
        super(Encode,self).__init__()

        self.conv1a = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv1b = nn.Conv2d(in_channels =64, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(2,2))

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(2,2))

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
    def __init__(self, out_channel=32):
        super(Decode,self).__init__()

        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=(2,2), padding=(2,2), kernel_size=(6,6))
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=(2,2), padding=(2,2), kernel_size=(6,6))
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channel, stride=(2,2), padding=(2,2), kernel_size=(6,6))

        self.bn_cn = nn.BatchNorm2d(num_features=2)
        self.PRelu = nn.PReLU()

    def forward(self,x):
        x = self.PRelu(self.convt1(x)) # (B,256,32,32) -> (B,128,64,64)
        x = self.PRelu(self.convt2(x)) # (B,128,64,64) -> (B,64,128,128)   
        x = self.PRelu(self.convt3(x)) # (B,64,128,128) -> (B,32,256,256)

        return x


def get_model(args):
    if args.model == 'RODNet':
        from Models.rodnet import RODNetCDC
        model = RODNetCDC(in_channels=args.frame, n_class=args.no_class)
    
    if args.model == 'RAMP':
        from Models.rampcnn import RAMPCNN
        model = RAMPCNN(in_channels=args.frame, n_class=args.no_class)

    if args.model == 'Crossatten':
        from Models.cross_atten import RadarCross
        model = RadarCross(in_channels=args.frame,
                        n_class=args.no_class,
                        center_offset=args.co,
                        orentation=args.oren)
    return model