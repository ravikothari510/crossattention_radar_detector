import numpy as np
import os
import matplotlib.pyplot as plt
from OS_CFAR_2D import DetectPeaksOSCFAR_2D

main_dir =r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\train_ra_map'
seq = r'2019-09-16-13-13-01_000187.npy'
ra_map = np.load(os.path.join(main_dir,seq))


out = DetectPeaksOSCFAR_2D(ra_map,10,10,10,10,0.24) #(x,TrainCell_R,TrainCell_V,GuardCell_R,GuardCell_V,FA_rate)

img1 = plt.imshow(ra_map)
for cord in out:
    col,row = cord
    plt.scatter(row,col,c='white')