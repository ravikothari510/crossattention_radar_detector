import os
import sys
#sys.path.append('/home/ravikothari/dev/code/architectures')
sys.path.append(r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\code_1\architectures')
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import json
import torch
from post_process_numpy_neu import create_default,peaks_detect, association,distribute
import matplotlib.pyplot as plt
from bi_gauss_update import center_create

gauss_mask =r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\train_mask_updated'
center_path = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\train_center'
device = "cuda"

dummy = torch.rand((1,3,256,256))
mask, peak_cls = create_default(dummy.shape,(3,5))
mask =mask.to(device=device)
peak_cls = peak_cls.to(device=device)

vect = np.zeros((2,9,9))
for i in range(9):
    for j in range(9):
        vect[0,i,j] = 4-i
        vect[1,i,j] = 4-j

for name in tqdm(os.listdir(gauss_mask)):
    seq,frame,_=name.split('_')
    map = torch.from_numpy(np.load(os.path.join(gauss_mask,name))).to(device=device)
    map = map.unsqueeze(dim=0)
    intent, idx = peaks_detect(map,mask,peak_cls, heat_thresh=0.8)
    idx = distribute(idx, device= device)
    idx,_ = association(intent,idx , device=device)
    idx = idx.to(device='cpu').numpy().astype('int')
    center_map = center_create(idx,vect).astype('float32')
    np.save(os.path.join(center_path,name.replace('_mask','_center')), center_map)