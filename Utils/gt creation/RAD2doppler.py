import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib 
import os
import random
from tqdm import tqdm

rd_dir_main = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\Carrada\raw_radar'
main_dir = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\datasets_master\Carrada_RAD'

for seq in os.listdir(main_dir):

    try:
        os.makedirs(os.path.join(main_dir,seq,'visualise'))
    except:pass

    rad_dir = os.path.join(main_dir,seq,'RAD_numpy')
    img_save_dir = os.path.join(main_dir,seq,'visualise')
    box_dir = os.path.join(rd_dir_main,seq,'box_proj.json')
    rd_dir = os.path.join(rd_dir_main,seq,'range_doppler_numpy')
    box = json.load(open(box_dir))


    print(seq)
    for ra_map_name in tqdm(os.listdir(rad_dir)):
        frame,_= ra_map_name.split('.')
        if frame in box:
            if len(box[frame])!=0:
                rd_map = np.load(os.path.join(rd_dir,ra_map_name))
                rd_map = np.flip(rd_map,0)
                rd_map = np.flip(rd_map,1)
                rad_map = np.load(os.path.join(rad_dir,ra_map_name))

                ad_map = np.fft.ifftshift(rad_map, axes=0)
                ad_map = np.fft.ifft(ad_map, axis=0)
                ad_map = pow(np.abs(ad_map), 2)
                ad_map = np.sum(ad_map, axis=0)
                ad_map = 10*np.log10(ad_map + 1)
                ad_map = np.transpose(ad_map)
                #ad_map = np.flip(ad_map,axis=1)
                ad_map[31:34,:]=0
                rd_map[:,31:34]=0

                ad_map = np.float32(ad_map)
                rd_map = np.float32(rd_map)
                
                ra_size = [64,256]
                fig,ax = plt.subplots(1,2,figsize =(15,7))
                
                ax[0].imshow(rd_map)
                ax[0].set_xticks([0, ra_size[0]*1/4-1,ra_size[0]*2/4-1,ra_size[0]*3/4-1,ra_size[0]-1])
                ax[0].set_yticks([0,ra_size[1]*1/5-1,ra_size[1]*2/5-1,ra_size[1]*3/5-1, ra_size[1]*4/5-1,ra_size[1]-1])
                ax[0].set_xticklabels(np.round(np.linspace(-13.5,13.5,5)))
                ax[0].set_yticklabels([50,40,30,20,10,0])



                ax[1].imshow(ad_map)
                ax[1].set_xticks([0, ra_size[1]*1/4-1,ra_size[1]*2/4-1,ra_size[1]*3/4-1,ra_size[1]-1])
                ax[1].set_yticks([0,ra_size[0]*1/4-1,ra_size[0]*2/4-1,ra_size[0]*3/4-1, ra_size[0]-1])
                ax[1].set_yticklabels(np.round(np.linspace(13.5,-13.5,5)))
                ax[1].set_xticklabels(np.round(np.rad2deg(np.arcsin(np.linspace(-1,1,5))),1))

                idx = box[frame]['idx']
                for cord in idx:
                    a1,a2 = int(cord[0]),int(cord[2])
                    r1,r2 = int(cord[1]),int(cord[3])
                    dummy = np.zeros((256,64))
                    dummy[r1:r2,:] = rd_map[r1:r2,:]
                    index = np.unravel_index(np.argmax(dummy),dummy.shape)
                    ax[0].scatter(index[1],index[0],c='r')

                    dummy = np.zeros((64,256))
                    dummy[:,a1:a2] = ad_map[:,a1:a2]
                    index = np.unravel_index(np.argmax(dummy),dummy.shape)
                    ax[1].scatter(index[1],index[0],c='r')
                
                plt.savefig(os.path.join(img_save_dir,frame+str('.jpg')))
                plt.close(fig)