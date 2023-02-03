import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from post_process_numpy import pol2cord
import matplotlib.image as mpimg

def velo_map(anno,orent_map):
    map = np.zeros((3,64,64))
    for cnt,cls in enumerate(anno['cls']):
        mu = [anno['mu_r'][cnt],anno['mu_a'][cnt]]
        mu = [int(mu[0]),int(mu[1])]
        map[2,mu[0]//4,mu[1]//4] = (anno['velo'][cnt]/14)
    map[0:2,::]= orent_map[0:2,::]
    return map.astype('float32')


def bi_var_gauss(anno,vect):
    map = np.zeros((3,256,256))
    center_map = np.zeros((2,256,256))
    orent_map = np.zeros((2,64,64))
    for cnt,cls in enumerate(anno['cls']):
        if type(anno['orent']) is not list:
            anno['orent'] = [anno['orent']]

        sigma_r = anno['sigma_r'][cnt]
        sigma_a = anno['sigma_a'][cnt]
        mu = [anno['mu_r'][cnt],anno['mu_a'][cnt]]
        if cnt>0:
             row = anno['sigma_cov'][1][1]
        else:
            row = anno['sigma_cov'][0][0]
        
        i_mat = np.arange(map.shape[1])
        i_mat = np.reshape(i_mat,(map.shape[1],1))
        i_mat = np.tile(i_mat,(1,map.shape[1]))

        j_mat = np.arange(map.shape[1])
        j_mat = np.tile(j_mat,(map.shape[1],1))

        dist = ((i_mat-mu[0])/(sigma_r))**2 + ((j_mat-mu[1])/(sigma_a))**2 -2*row*(i_mat-mu[0])/(sigma_r)*(j_mat-mu[1])/(sigma_a)
        dist = dist/(2*(1-row**2))

        map[int(cls-1),::] = np.amax((np.stack((map[int(cls-1),::],np.exp(-dist)),axis=0)),axis=0)

        mu = [int(mu[0]),int(mu[1])]
        r_p = np.min((4,mu[0]))
        a_p = np.min((4,mu[1]))
        r_n = np.min((4,255-mu[0]))
        a_n = np.min((4,255-mu[1]))
        center_map[:,(mu[0]-r_p):(mu[0]+r_n+1),(mu[1]-a_p):(mu[1]+a_n+1)] = vect[:,(4-r_p):(4+r_n+1), (4-a_p):(4+a_n+1)]
        orent_map[:,mu[0]//4,mu[1]//4] = [np.sin(np.deg2rad(anno['orent'][cnt])) ,np.cos(np.deg2rad(anno['orent'][cnt]))]
          
    return map.astype('float32'),center_map.astype('float32'),orent_map.astype('float32')
        
















def gauss_parameters(map,box):
    
    mu_r,mu_a = [],[]
    sigma_r, sigma_a, sigma_cov = [],[],[]
    for count,label in enumerate(box['cls']):
        label = int(label)
        a1,r1,a2,r2 = [int(i) for i in box['idx'][count]]

        if a2-a1 >40:
            a1 = int(np.max((5,np.mean((a1,a2))-20)))
            a2 = int(np.min((250,(a1+40))))
        if r2-r1 >20:
            r1 = int(np.max((5,np.mean((r1,r2))-10)))
            r2 = int(np.min((250,(r1+20))))          

        patch = map[r1:r2,a1:a2]
        patch = (patch-np.mean(patch))/(np.max(patch)-np.min(patch))
        temp_mask = np.zeros((256,256))
        temp_mask[r1:r2,a1:a2] = patch
        cord = np.where(temp_mask>0.1)
        sigma = np.cov([cord[0],cord[1]])
        mu_r.append(np.mean(cord[0]))
        mu_a.append(np.mean(cord[1]))
        sigma_r.append(np.sqrt(sigma[0,0]))
        sigma_a.append(np.sqrt(sigma[1,1]))
        #print(sigma)
        sigma_cov.append(sigma[0,1]/sigma_r/sigma_a)
    box['mu_r'] = mu_r
    box['mu_a'] = mu_a
    box['sigma_r'] = sigma_r
    box['sigma_a'] = sigma_a
    box['sigma_cov'] = sigma_cov
    return box



def gauss(mu,sigma,map):
    sigma_r = np.sqrt(sigma[0,0])
    sigma_a = np.sqrt(sigma[1,1])
    row = sigma[0,1]/sigma_r/sigma_a

    i_mat = np.arange(map.shape[0])
    i_mat = np.reshape(i_mat,(map.shape[0],1))
    i_mat = np.tile(i_mat,(1,map.shape[1]))

    j_mat = np.arange(map.shape[1])
    j_mat = np.tile(j_mat,(map.shape[0],1))

    dist = ((i_mat-mu[0])/(sigma_r))**2 + ((j_mat-mu[1])/(sigma_a))**2 -2*row*(i_mat-mu[0])/(sigma_r)*(j_mat-mu[1])/(sigma_a)
    dist = dist/(2*(1-row**2))
    
    map = np.amax((np.stack((map,np.exp(-dist)),axis=0)),axis=0)
    return map

def gauss_map_2(map,box,mask):
    mask_gauss = np.zeros((3,256,256))
    for count,label in enumerate(box['cls']):
        label = int(label)
        a1,r1,a2,r2 = [int(i) for i in box['idx'][count]]

        if a2-a1 >40:
            a1 = int(np.max((5,np.mean((a1,a2))-20)))
            a2 = int(np.min((250,(a1+40))))
        if r2-r1 >20:
            r1 = int(np.max((5,np.mean((r1,r2))-10)))
            r2 = int(np.min((250,(r1+20))))          

        patch = map[r1:r2,a1:a2]
        patch = (patch-np.mean(patch))/(np.max(patch)-np.min(patch))
        temp_mask = np.zeros((256,256))
        temp_mask[r1:r2,a1:a2] = patch
        cord = np.where(temp_mask>0.1)
        sigma = np.cov([cord[0],cord[1]])
        mu= [np.mean(cord[0]), np.mean(cord[1])]
        mask_gauss[label-1,:,:] = gauss(mu,sigma,mask_gauss[label-1,:,:])
        mask[label-1,r1:r2,a1:a2]=patch
    return mask,mask_gauss


def center_create(cord,vect):
    center_map = np.zeros((2,256,256))
    for idx in cord:
        r_idx = idx[2]
        a_idx = idx[3]
        r_p = np.min((4,r_idx))
        a_p = np.min((4,a_idx))
        r_n = np.min((4,255-r_idx))
        a_n = np.min((4,255-a_idx))

        center_map[:,(r_idx-r_p):(r_idx+r_n+1),(a_idx-a_p):(a_idx+a_n+1)] = vect[:,(4-r_p):(4+r_n+1), (4-a_p):(4+a_n+1)]
          
    return center_map




def gauss_map(map,box,mask):
    for count,label in enumerate(box['labels']):
        cord = box['boxes'][count]
        r1 = np.min((cord[0],cord[2]))
        r2 = np.max((cord[0],cord[2]))
        a1 = np.min((cord[1],cord[3]))
        a2 = np.max((cord[2],cord[3]))
        if a2-a1<40:
            a1 = int(np.max((0,np.mean((cord[1],cord[3]))-20)))
            a2 = int(np.min((250,np.mean((cord[1],cord[3]))+20)))
        if r2-r1 <20:
            r1 = int(np.max((0,np.mean((cord[0],cord[2]))-10)))
            r2 = int(np.min((250,np.mean((cord[0],cord[2]))+10)))
        patch = map[r1:r2,a1:a2]
        patch = (patch-np.mean(patch))/(np.max(patch)-np.min(patch))
        temp_mask = np.zeros((256,256))
        temp_mask[r1:r2,a1:a2] = patch
        cord = np.where(temp_mask>0.1)
        sigma = np.cov([cord[0],cord[1]])
        mu= [np.mean(cord[0]), np.mean(cord[1])]

        mask[label-1,:,:] = gauss(mu,sigma,mask[label-1,:,:])
    return mask

def save_gauss_2(mask,gauss_mask,map,img_dir,plot_path):
    ra_size = map.shape
    fig, ax  = plt.subplots(2,4,figsize = (20,10))
    plt.suptitle('gauss_annotate')
    title =['pedestrian','cyclist','car']
    for i in range(2):
        for j in range(3):
            if i==0: ax[i,j].imshow(mask[j,:,:])
            else : ax[i,j].imshow(gauss_mask[j,:,:])
            ax[i,j].title.set_text(title[j])
            ax[i,j].set_xticks([0, ra_size[1]*1/4-1,ra_size[1]*2/4-1,ra_size[1]*3/4-1,ra_size[1]-1])
            ax[i,j].set_yticks([0,ra_size[1]*1/5-1,ra_size[1]*2/5-1,ra_size[1]*3/5-1, ra_size[1]*4/5-1,ra_size[1]-1])
            ax[i,j].set_yticklabels([50, 40, 30, 20, 10, 0])
            ax[i,j].set_xticklabels(np.round(np.rad2deg(np.arcsin(np.linspace(-1,1,5))),1))
            ax[i,j].set_ylabel('Distance (m)')
            ax[i,j].set_xlabel('Angle (Degree)')
    ax[0,3].imshow(map)
    img = mpimg.imread(img_dir)
    ax[1,3].imshow(img)
    plt.savefig(plot_path)
    plt.close()
    

def box_vis(map,box):
    mask = np.zeros((256,256))
    cord_list = dict()
    cord_list['cls']=[]
    cord_list['idx']=[]

    for count,label in enumerate(box['labels']):
        cord = box['boxes'][count]
        r1 = np.min((cord[0],cord[2]))
        r2 = np.max((cord[0],cord[2]))
        a1 = np.min((cord[1],cord[3]))
        a2 = np.max((cord[1],cord[3]))
        if a2-a1<40:
            a1 = int(np.max((5,np.mean((cord[1],cord[3]))-20)))
            a2 = int(np.min((250,np.mean((cord[1],cord[3]))+20)))
        if r2-r1 <20:
            r1 = int(np.max((5,np.mean((cord[0],cord[2]))-10)))
            r2 = int(np.min((250,np.mean((cord[0],cord[2]))+10)))
        patch = map[r1:r2,a1:a2]
        patch = (patch-np.mean(patch))/(np.max(patch)-np.min(patch))
        
        cord_list['cls']=np.append(cord_list['cls'],label)
        cord_list['idx']=np.append(cord_list['idx'],[a1,r1,a2,r2])
        mask[r1:r2,a1:a2] = patch
        cord_list['idx'] = np.reshape(cord_list['idx'],(len(cord_list['cls']),4))

    return mask, cord_list

def save_annotate_plot(map, cord_list, plot_path, img_dir, cam):

    title = ['map','mask']
    mask = np.zeros((256,256))
    ra_size = mask.shape
    fig, ax  = plt.subplots(1,3,figsize = (15,5))
    plt.suptitle('update annotation')
    fig.tight_layout()

    ax[0].imshow(map) 
    img = mpimg.imread(img_dir)
    ax[2].imshow(img) 
    
    
    for cnt,_ in enumerate(cord_list['cls']):
        a1,r1,a2,r2 = [int(i) for i in cord_list['idx'][cnt]]
        rect = patches.Rectangle((a1, r1),(a2-a1), (r2-r1), linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        patch = map[r1:r2,a1:a2]
        patch = (patch-np.mean(patch))/(np.max(patch)-np.min(patch))
        mask[r1:r2,a1:a2] = patch
        x,y=pol2cord(int(r2),int(np.mean((a1,a2))), device="cpu")
        x,y=cam.worldToImage_opcv(y,x,0)
        ax[2].scatter(x,y, c='red')


    for chn in range(2):
        ax[chn].title.set_text(title[chn])
        ax[chn].set_xticks([0, ra_size[1]*1/4-1,ra_size[1]*2/4-1,ra_size[1]*3/4-1,ra_size[1]-1])
        ax[chn].set_yticks([0,ra_size[1]*1/5-1,ra_size[1]*2/5-1,ra_size[1]*3/5-1, ra_size[1]*4/5-1,ra_size[1]-1])
        ax[chn].set_yticklabels([50, 40, 30, 20, 10, 0])
        ax[chn].set_xticklabels(np.round(np.rad2deg(np.arcsin(np.linspace(-1,1,5))),1))
        ax[chn].set_ylabel('Distance (m)')
        ax[chn].set_xlabel('Angle (Degree)')

    ax[1].imshow(mask,cmap='viridis',vmin = 0,vmax =1) 
    plt.savefig(plot_path)
    plt.close()
    


def save_update_plot(map,mask_old,mask_neu, cord_list,cord_neu, plot_path, cam, img_dir):

    title = ['ra_map','mask_old', 'mask_neu','camera']
    img = mpimg.imread(img_dir)
    ra_size = mask_old.shape
    fig, ax  = plt.subplots(1,4,figsize = (20,5))
    plt.suptitle('visualisation')
    fig.tight_layout()

    ax[1].imshow(mask_old,cmap='viridis',vmin = 0,vmax =1) 
    ax[2].imshow(mask_neu,cmap='viridis',vmin = 0,vmax =1)
    
    ax[0].imshow(map)  
    ax[3].imshow(img)
    
    
    for cnt,_ in enumerate(cord_list['cls']):
        a1,r1,a2,r2 = cord_list['idx'][cnt]
        rect = patches.Rectangle((a1, r1),(a2-a1), (r2-r1), linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        x,y=pol2cord(int(r2),int(np.mean((a1,a2))), device="cpu")
        x,y=cam.worldToImage_opcv(y,x,0)
        ax[3].scatter(x,y, c='red')
    
    for cnt,_ in enumerate(cord_neu['cls']):
        a1,r1,a2,r2 = cord_neu['idx'][cnt]
        rect = patches.Rectangle((a1, r1),(a2-a1), (r2-r1), linewidth=1, edgecolor='yellow', facecolor='none')
        ax[0].add_patch(rect)
        x,y=pol2cord(int(r2),int(np.mean((a1,a2))), device="cpu")
        x,y=cam.worldToImage_opcv(y,x,0)
        ax[3].scatter(x,y, c='yellow')

    for chn in range(3):
        ax[chn].title.set_text(title[chn])
        ax[chn].set_xticks([0, ra_size[1]*1/4-1,ra_size[1]*2/4-1,ra_size[1]*3/4-1,ra_size[1]-1])
        ax[chn].set_yticks([0,ra_size[1]*1/5-1,ra_size[1]*2/5-1,ra_size[1]*3/5-1, ra_size[1]*4/5-1,ra_size[1]-1])
        ax[chn].set_yticklabels([50, 40, 30, 20, 10, 0])
        ax[chn].set_xticklabels(np.round(np.rad2deg(np.arcsin(np.linspace(-1,1,5))),1))
        ax[chn].set_ylabel('Distance (m)')
        ax[chn].set_xlabel('Angle (Degree)')


    plt.savefig(plot_path)
    plt.close()   

def center_shift(map,cord_list,angle_prior,range_prior):
    mask = np.zeros((256,256))

    cord_neu = dict()
    cord_neu['cls']=[]
    cord_neu['idx']=[]

    for cnt, cls in enumerate(cord_list['cls']):
        a1,r1,a2,r2 = [int(i) for i in cord_list['idx'][cnt]]
        a_c = (a1+a2)//2
        r_c = (r1+r2)//2
        rng = r_c/256*50
        patch = np.zeros(map.shape)
        patch[r1:r2,a1:a2] = map[r1:r2,a1:a2]
        idx = np.where(patch == np.max(patch))

        if a2-a1 >40:
            a1 = int(np.max((5,np.mean((a1,a2))-20)))
            a2 = int(np.min((250,(a1+40))))
        if r2-r1 >20:
            r1 = int(np.max((5,np.mean((r1,r2))-10)))
            r2 = int(np.min((250,(r1+20))))          
        # if not range_prior:
        #     if rng<47 and rng>2:
        #         if (np.abs(idx[0]-r_c)>2) or (np.abs(idx[1]-a_c)>2):
        #             a_c = idx[1]
        #             r_c = idx[0]
        #             h = np.abs((r1-r2)//2)
        #             w = np.abs((a1-a2)//2)
        #             if not angle_prior:
        #                 a1 = int(np.max((5,a_c-w )))
        #                 a2 = int(np.min((250,a_c+w )))
                    
        #             r1 = int(np.max((5,r_c-h)))
        #             r2 = int(np.min((250,r_c+h)))

        if rng<47 and rng>2:
            if (np.abs(idx[0]-r_c)>2) or (np.abs(idx[1]-a_c)>2):
                a_c = idx[1]
                r_c = idx[0]
                h = np.abs((r1-r2)//2)
                w = np.abs((a1-a2)//2)
                a1 = int(np.max((5,a_c-w )))
                a2 = int(np.min((250,a_c+w )))
                r1 = int(np.max((5,r_c-h)))
                r2 = int(np.min((250,r_c+h)))
        


        cord_neu['cls']=np.append(cord_neu['cls'],cls)
        cord_neu['idx']=np.append(cord_neu['idx'],[a1,r1,a2,r2])
        cord_neu['idx'] = np.reshape(cord_neu['idx'],(len(cord_neu['cls']),4))
        patch_s = map[r1:r2,a1:a2]
        patch_s = (patch_s-np.mean(patch_s))/(np.max(patch_s)-np.min(patch_s))
        mask[r1:r2,a1:a2] = patch_s


    return cord_neu, mask

def clean_up (anno_proj,cord_list,cord_neu):
    cord_update = dict()
    labels = dict()
    for label in anno_proj:
        name = anno_proj[label]['filename']
        name = name.replace('.jpg','')
        _,name = name.split('_')
        labels[name]= anno_proj[label]
    anno_proj = labels

    for frame in cord_list:

        if 'type' in anno_proj[frame]['file_attributes']: 
            if 'out' in anno_proj[frame]['file_attributes']['type']: cord_update[frame]=[]
            elif 'keep_old'in anno_proj[frame]['file_attributes']['type']: cord_update[frame]=cord_list[frame]
            else:cord_update[frame]=cord_neu[frame]
        else:cord_update[frame]=cord_neu[frame]

        if len(anno_proj[frame]['regions'])!=0:
             for cord in anno_proj[frame]['regions']:
                place=False
                x = cord['shape_attributes']['x']
                y = cord['shape_attributes']['y']
                w = cord['shape_attributes']['width']
                h = cord['shape_attributes']['height']

                if 'pedestrian_1' in cord['region_attributes']['class'] : cls =1
                elif 'cyclist_1' in cord['region_attributes']['class'] : cls =2
                else : cls=3

                a1 = int((x-45)/294*256)
                r1 = int((y-39)/294*256)
                a2 = int(a1+ (w/294*256))
                r2 = int(r1 + (h/294*256))
                for cnt,label in enumerate(cord_list[frame]['cls']):
                    if label == cls:
                        place=True
                        cord_update[frame]['cls'][cnt]=label
                        cord_update[frame]['idx'][cnt]=[a1,r1,a2,r2]
                if not place:
                    cord_update[frame]['cls']=np.append(cord_update[frame]['cls'],cls)
                    cord_update[frame]['idx']=np.append(cord_update[frame]['idx'],[a1,r1,a2,r2])
                    cord_update[frame]['idx'] = np.reshape(cord_update[frame]['idx'],(len(cord_update[frame]['cls']),4))

                    #cord_update[frame]['idx'] = np.reshape(cord_update[frame]['idx'],(len(cord_update[frame]['cls']),4))

    return anno_proj, cord_update


def critical_exm(anno_proj,cord_list,cord_neu,cycle_anno,cord_update):
    if 'type' in anno_proj['file_attributes']: 
            if 'out' in anno_proj['file_attributes']['type']: cord_update=[]
            elif 'keep_old'in anno_proj['file_attributes']['type']: cord_update=cord_list
            else:cord_update=cord_neu
    else:cord_update=cord_neu

    for cord in anno_proj['regions']:
        x = cord['shape_attributes']['x']
        y = cord['shape_attributes']['y']
        w = cord['shape_attributes']['width']
        h = cord['shape_attributes']['height']
        
        a1 = int((x-45)/294*256)
        r1 = int((y-39)/294*256)
        a2 = int(a1+ (w/294*256))
        r2 = int(r1 + (h/294*256))

        if cycle_anno:
            cord_update['cls']=np.append(cord_update['cls'],2)
            cord_update['idx']=np.append(cord_update['idx'],[a1,r1,a2,r2])
            cord_update['idx'] = np.reshape(cord_update['idx'],(len(cord_update['cls']),4))

    return cord_update
