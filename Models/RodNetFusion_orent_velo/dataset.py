import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Carrada(Dataset):
    def __init__(self,ra_map_dir,rd_map_dir,ad_map_dir,mask_dir,center_dir,orent_dir,ra_transform = None,rd_transform=None,ad_transform=None):
        self.ra_map_dir = ra_map_dir
        self.rd_map_dir = rd_map_dir
        self.ad_map_dir = ad_map_dir
        self.mask_dir = mask_dir
        self.center_dir = center_dir
        self.orent_dir = orent_dir

        self.ra_transform = ra_transform
        self.rd_transform = rd_transform
        self.ad_transform = ad_transform
        self.ra_map_list = os.listdir(ra_map_dir)

    def __len__(self):
        return len(self.ra_map_list)

    def __getitem__(self,index):
        ra_map_path = os.path.join(self.ra_map_dir,self.ra_map_list[index])  
        rd_map_path = os.path.join(self.rd_map_dir,self.ra_map_list[index]) 
        ad_map_path = os.path.join(self.ad_map_dir,self.ra_map_list[index]) 
        mask_path = os.path.join(self.mask_dir,self.ra_map_list[index].replace(".npy","_mask.npy")) 
        center_path = os.path.join(self.center_dir,self.ra_map_list[index].replace(".npy","_center.npy")) 
        orent_path = os.path.join(self.orent_dir,self.ra_map_list[index].replace(".npy","_orent.npy")) 
        ra_map = torch.from_numpy(np.load(ra_map_path))[:5,::]
        rd_map = torch.from_numpy(np.load(rd_map_path))[:5,::]
        ad_map = torch.from_numpy(np.load(ad_map_path))[:5,::]
        mask = torch.from_numpy(np.load(mask_path))
        center = torch.from_numpy(np.load(center_path))
        orent = torch.from_numpy(np.load(orent_path))

        
        ra_map = self.ra_transform(ra_map)
        rd_map = self.rd_transform(rd_map)
        ad_map = self.ad_transform(ad_map)

        return ra_map, rd_map, ad_map, mask ,center, orent,self.ra_map_list[index]   

