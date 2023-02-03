import numpy as np
from transform import polToCart
import os
import matplotlib.pyplot as plt

seq = r'2019-09-16-13-13-01'
frame = r'000120'
dir = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\Dataset\CARRADA_old\50_Dataset\bi_variate_norm\val_ra_map'

destination = r'G:\EFS-GX6\4130_Arbeitsgruppen\4137_KHO_Performance\Studentische_Themen\RaviKothari\thesis\images'

ra_map = np.load(os.path.join(dir,seq+'_'+frame+'.npy'))

ra_cart = polToCart(ra_map)

plt.imshow(ra_cart)
plt.xticks(np.linspace(0,ra_cart.shape[1],5),labels=[-50,-25,0,25,50])
plt.yticks(ticks=np.linspace(0,ra_cart.shape[0],6).astype('int'),labels= [50,40,30,20,10,0])

plt.savefig(os.path.join(destination,'bev.jpg'),dpi=1200)