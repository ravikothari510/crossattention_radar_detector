import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Carrada(Dataset):
    def __init__(self,ra_map_dir,rd_map_dir,ad_map_dir,mask_dir,ra_transform = None,rd_transform=None,ad_transform=None):
        self.ra_map_dir = ra_map_dir
        self.rd_map_dir = rd_map_dir
        self.ad_map_dir = ad_map_dir
        self.mask_dir = mask_dir

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

        ra_map = np.load(ra_map_path)
        rd_map = np.load(rd_map_path)
        ad_map = np.load(ad_map_path)
        mask = torch.from_numpy(np.load(mask_path))

        if self.ra_transform is not None:
            ra_augmentations = self.ra_transform(image=ra_map)
            rd_augmentations = self.rd_transform(image=rd_map)
            ad_augmentations = self.ad_transform(image=ad_map)
            ra_map = ra_augmentations["image"]
            rd_map = rd_augmentations["image"]
            ad_map = ad_augmentations["image"]

        return ra_map, rd_map, ad_map, mask ,self.ra_map_list[index]   

