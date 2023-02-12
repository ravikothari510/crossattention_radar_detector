import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

class RadarData(Dataset):
    def __init__(self, dataset_dir, frame_list, transform,
                no_frames=1, orent=False, center=False, rad=True,
                bivar=False):
        self.dataset_dir = dataset_dir
        self.transform = transform 
        self.orent = orent
        self.center = center
        self.bivar = bivar
        self.rad = rad # if doppler maps available
        self.no_frames = no_frames
        with open(os.path.join(self.dataset_dir, frame_list+'.txt'),'r') as f:
            self.frames = [line.strip() for line in f.readlines()]

    def __len__(self):
        
        return len(self.frames)
    
    def __getitem__ (self, index):


        frame = self.frames[index]
        seq, frame_no = frame.split('_')

        rd_map = 0
        ad_map = 0
        orent = 0
        center = 0

        ra_map = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
            'fft_maps', seq, 'ra_map', frame_no + '.npy' )))[:self.no_frames,::]
        ra_map = self.transform['ra'](ra_map)

        if self.rad:
            rd_map = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'fft_maps', seq, 'rd_map', frame_no + '.npy' )))[:self.no_frames,::]
            rd_map = self.transform['rd'](rd_map)

            ad_map = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'fft_maps', seq, 'ad_map', frame_no + '.npy' )))[:self.no_frames,::]
            ad_map = self.transform['ad'](ad_map)

        if self.orent:
            orent = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'gt_maps', seq, 'orent', frame_no + '.npy')))
        
        if self.center:
            center = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'gt_maps', seq, 'center', frame_no + '.npy')))
        
        if self.bivar:
            mask = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'gt_maps', seq, 'bivar_gauss', frame_no + '.npy')))
        else:
            mask = torch.from_numpy(np.load(os.path.join(self.dataset_dir,\
                'gt_maps', seq, 'gauss', frame_no + '.npy')))

        return ra_map, rd_map, ad_map,\
            mask, center, orent, frame  


def load_data(cfg, args, transform):

    train_ds = RadarData(dataset_dir=args.data_dir,
                        frame_list='train_frames',
                        transform=transform,
                        no_frames=args.frame,
                        orent=args.oren,
                        center=args.co,
                        rad = args.model!='RODNet',
                        bivar=args.gauss=='Bivar')
    
    train_load = DataLoader(train_ds,
                            batch_size=int(cfg['SOLVER']['BATCH_SIZE']),
                            num_workers=int(cfg['SOLVER']['NUM_WORKERS']),
                            pin_memory=True,
                            shuffle=True)
    
    val_ds = RadarData(dataset_dir=args.data_dir,
                        frame_list='val_frames',
                        transform=transform,
                        no_frames=args.frame,
                        orent=args.oren,
                        center=args.co,
                        rad = args.model!='RODNet',
                        bivar=args.gauss=='Bivar')

    val_load = DataLoader(val_ds,
                          batch_size=int(cfg['SOLVER']['BATCH_SIZE']),
                          num_workers=int(cfg['SOLVER']['NUM_WORKERS']),
                          pin_memory=True,
                          shuffle=False)
    return train_load, val_load