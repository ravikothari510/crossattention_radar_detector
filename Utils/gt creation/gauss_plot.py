import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\source_code\Utils')
from bi_gauss_update import gauss

s_dir = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\train_mask'
annotate = np.load('gt_annotation.npy',allow_pickle=True).item()
save_dir = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\train_mask_base_15'

sigma = np.zeros((2,2))
sigma[0,0],sigma[1,1] = 15,15
for name in tqdm(os.listdir(s_dir)):
    seq,frame,_= name.split('_')
    mask = np.zeros((3,256,256))
    
    for cnt,obj in enumerate(annotate[seq][frame]['cls']):
        pass
        mu = [0,0]
        mu[0]=annotate[seq][frame]['mu_r'][cnt]
        mu[1]=annotate[seq][frame]['mu_a'][cnt]
        mask[int(obj)-1] = gauss(mu,sigma,mask[int(obj)-1])
    mask = np.float32(mask)
    np.save(os.path.join(save_dir,name),mask)