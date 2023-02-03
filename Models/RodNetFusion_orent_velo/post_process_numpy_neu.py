import numpy as np
from numpy.core.fromnumeric import shape, size
import torch
import torch.nn as nn



def create_default(map_shape, kernel_window):

    peaks_cls = nn.MaxPool2d(kernel_size= kernel_window,stride=1,return_indices=True)

    h,w = torch.div(torch.Tensor(kernel_window),2, rounding_mode='floor')

    mask_h = (map_shape[2]-2*h).int()
    mask_w = (map_shape[3]-2*w).int()
    mask_t = torch.zeros(mask_h,mask_w)


    for i in range(mask_t.shape[0]):
        for j in range(mask_t.shape[1]):
            mask_t[i,j]=(i+h)*map_shape[3]+ j+w
    
    mask = torch.zeros(map_shape[0],map_shape[1],mask_h,mask_w)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i,j,:,:]= mask_t


    
    return mask, peaks_cls

def peaks_detect(map,mask, peaks_cls, heat_thresh):
    out = peaks_cls(map)

    peak = out[1]==mask
    peak_thresh = out[0]>heat_thresh

    idx = torch.where(peak*peak_thresh ==True)

    return out[0][idx], idx  # intensity, index

def distribute(index_t,device='cpu'):
    index = torch.tensor(()).to(device=device)
    for i in range(len(index_t[0])):
        index = torch.cat((index, torch.stack((index_t[0][i],index_t[1][i],index_t[2][i],index_t[3][i]),dim=0)))

    index = torch.reshape(index,(len(index_t[0]),4))
    return index

def association(intensity, index,device ="cpu"):

    idx_list = torch.argsort(intensity)
    idx_list = torch.flip(idx_list,dims=[-1])
    out_intent =torch.Tensor().to(device=device)
    out_idx =torch.Tensor().to(device=device)


    while len(idx_list)!=0:
        out_intent=torch.cat((out_intent,torch.Tensor([intensity[idx_list[0]]]).to(device=device)))
        out_idx = torch.cat((out_idx, index[idx_list[0]]))
        p1_row , p1_col = index[idx_list[0]][2],index[idx_list[0]][3]
        frame_1, cls_1 = index[idx_list[0]][0],index[idx_list[0]][1]
        #print(p1_row,p1_col)
        x1,y1 = pol2cord(p1_row,p1_col)
        #print(x1,y1)
        idx_list = idx_list[1:]
        

        count = 0
        while len(idx_list)!=0 and count != len(idx_list):
            frame_2, cls_2 = index[idx_list[count]][0],index[idx_list[count]][1]
            #if frame_2==frame_1 and cls_1==cls_2:
            if frame_2==frame_1:
                p2_row , p2_col = index[idx_list[count]][2],index[idx_list[count]][3]
                x2,y2 = pol2cord(p2_row,p2_col)
                #print(x2,y2)
                dist = distance(x1,y1,x2,y2)
                #print(dist)
            
                if dist <2:
                   idx_list = torch.cat([idx_list[:count], idx_list[count+1:]])
                else:count +=1
            else: count +=1
    out_idx = torch.reshape(out_idx,(len(out_idx)//4,4))
    return out_idx,out_intent



def pol2cord(rng_idx, agl_idx):
    range_array = torch.linspace(50,0,steps=256)
    w = torch.linspace(-1,1,steps=256)
    angle_array = torch.arcsin(w)
    range = range_array[int(rng_idx)]
    angle = angle_array[int(agl_idx)]

    return range*torch.sin(angle), range*torch.cos(angle)

def distance (x1,y1,x2,y2):
    #print(f"{x1,y1,x2,y2}\n")
    dx = x1-x2
    dy = y1-y2

    return torch.sqrt(dx**2 + dy**2)


def analyse(grd_idx,pred_idx,grd_cp,pred_cp,idx,dist_thresh = 2,device='cpu'):
    idx = torch.Tensor([idx]).to(device=device)

    for prd_cord in pred_idx:
        p_r,p_c= prd_cord[2],prd_cord[3]
        p_fr,p_cls = prd_cord[0],int(prd_cord[1])

        x1,y1 = pol2cord(p_r,p_c)
        
        pred_idx = pred_idx[1:]
        
        if len(grd_idx)!=0:
            dist=torch.Tensor().to(device =device)
            for grd_cord in grd_idx:
                g_fr,g_cls = grd_cord[0],int(grd_cord[1])
                if g_fr==p_fr and g_cls==p_cls:
                    g_r,g_c= grd_cord[2],grd_cord[3]
                    x2,y2 = pol2cord(g_r,g_c)
                    dist = torch.cat([dist,torch.Tensor([distance(x1,y1,x2,y2)]).to(device=device)])
                    
                else :
                    dist = torch.cat([dist,torch.Tensor([100]).to(device=device)])

            if torch.min(dist)<dist_thresh:
                #als[f"{p_cls}"]['true']+=1
                min_idx = torch.argmin(dist)
                grd_idx = torch.cat([grd_idx[0:min_idx],grd_idx[1+min_idx:]])
                    #print(len(grd_idx))
            else:
                pred_cp = torch.cat([pred_cp,prd_cord,idx])
            
        else:
            pred_cp = torch.cat([pred_cp,prd_cord,idx])

    

    for grd_cord in grd_idx: 
        grd_cp = torch.cat([grd_cp,grd_cord,idx])

    #     #print('in_1')
    # pred_cp = torch.reshape(pred_cp, (len(pred_cp)//4,4))
    # grd_cp = torch.reshape(grd_cp, (len(grd_cp)//4,4))
    # for prd_cord in pred_cp:
    #     p_r,p_c= prd_cord[2],prd_cord[3]
    #     p_fr,p_cls = prd_cord[0],int(prd_cord[1])

    #     x1,y1 = pol2cord(p_r,p_c)
        
    #     pred_cp = pred_cp[1:]
        
    #     if len(grd_cp)!=0:
    #         dist=torch.Tensor().to(device =device)
    #         for grd_cord in grd_cp:
    #             g_fr,g_cls = grd_cord[0],int(grd_cord[1])
    #             if g_fr==p_fr:
    #                 g_r,g_c= grd_cord[2],grd_cord[3]
    #                 x2,y2 = pol2cord(g_r,g_c)
    #                 dist = torch.cat([dist,torch.Tensor([distance(x1,y1,x2,y2)]).to(device=device)])
                    
    #             else :
    #                 dist = torch.cat([dist,torch.Tensor([100]).to(device=device)])

    #         if torch.min(dist)<dist_thresh:
    #             min_idx = torch.argmin(dist)
    #             mcls =int(grd_cp[min_idx][1])
    #             als[f"{p_cls}"]['miss_cls'][f"{mcls}"] +=1
    #             grd_cp = torch.cat([grd_cp[0:min_idx],grd_cp[1+min_idx:]])
    #         else :als[f"{p_cls}"]['left_fp'] +=1
    #     else :als[f"{p_cls}"]['left_fp'] +=1

    # for grd_cord in grd_idx: 
    #     g_cls = int(grd_cord[1])         
    #     als[f"{g_cls}"]['left_fn'] +=1      

    return grd_cp,pred_cp


def orent (orent_map,r,a,velo=0,pred=False):

    s_t = orent_map[0,r,a]
    c_t = orent_map[1,r,a]
    s = s_t/((s_t**2 + c_t**2)**0.5)
    c = c_t/((s_t**2 + c_t**2)**0.5)
    angle = torch.arccos(c).rad2deg()
    if np.isnan(angle.cpu().numpy()): print(s_t,c_t,s,c)
    if s<0:
        angle= -angle

    if pred and torch.abs(angle)>150:
        w = torch.linspace(-1,1,steps=64)
        angle_array = torch.rad2deg(torch.arcsin(w))
        prj_angle = angle_array[a]
        delta = angle-prj_angle

        if velo>0:
            if delta<90 and delta>-90:pass
            else:
                #print(v,angle,prj_angle)
                angle +=180
        else:
            if delta<90 and delta>-90: 
                angle+=180
        if angle>180:
            angle = angle-360
        return angle
    return angle

def velo(orent_map,r,a,pred=False,rd=0):
    v = orent_map[2,r,a]

    if pred:
        r_ = 4*r
        rd[:,:,31:33] = 0
        sum= 0
        for i in range(5):
            dummy = torch.zeros((256,64))
            dummy[r_-5:r_+5,:]= torch.abs(rd[i,r_-5:r_+5,:])
            index = np.unravel_index(torch.argmax(dummy).cpu().numpy(),shape=(256,64))
            sum+=index[1]
        sum = sum/5
        v=(sum-32)/64

    return v



    



def validation(grd_idx,pred_idx,pred_int,tr_o,pr_o,rd_map,cls,dist_thresh = 2,device='cpu'):

    foo =0

    for grd_cord in grd_idx:
        g_fr,g_cls = grd_cord[0],int(grd_cord[1])
        cls[f"{g_cls}"]['grd_no'] +=1

    while len(pred_int)!=0:
        cord = pred_idx[0]
        p_r,p_c= cord[2],cord[3]
        p_fr,p_cls = cord[0],int(cord[1])
        cls[f"{p_cls}"]['Conf'] = np.append(cls[f"{p_cls}"]['Conf'],pred_int[0].to('cpu').numpy())
        x1,y1 = pol2cord(p_r,p_c)
        dist=torch.Tensor().to(device =device)
        pred_int = pred_int[1:]
        pred_idx = pred_idx[1:]
        
        for grd_cord in grd_idx:
            g_fr,g_cls = grd_cord[0],int(grd_cord[1])
            if g_fr==p_fr and g_cls==p_cls:
                g_r,g_c= grd_cord[2],grd_cord[3]
                x2,y2 = pol2cord(g_r,g_c)
                dist = torch.cat([dist,torch.Tensor([distance(x1,y1,x2,y2)]).to(device=device)])
            else :
                dist = torch.cat([dist,torch.Tensor([100]).to(device=device)])

        if len(dist)!=0:
            #print(f"{dist}\n")
            if torch.min(dist)<dist_thresh:
                min_idx = torch.argmin(dist)



                #cls[f"{p_cls}"]['asc']['gt_r']=np.append(cls[f"{p_cls}"]['asc']['gt_r'],grd_idx[min_idx][2].to('cpu').numpy())
                #cls[f"{p_cls}"]['asc']['gt_a']=np.append(cls[f"{p_cls}"]['asc']['gt_a'],grd_idx[min_idx][3].to('cpu').numpy())
                #cls[f"{p_cls}"]['asc']['pd_r']=np.append(cls[f"{p_cls}"]['asc']['pd_r'],p_r.to('cpu').numpy())
                #cls[f"{p_cls}"]['asc']['pd_a']=np.append(cls[f"{p_cls}"]['asc']['pd_a'],p_c.to('cpu').numpy())
                t_r,t_c = int(grd_idx[min_idx][2]),int(grd_idx[min_idx][3])
                grd_idx = torch.cat([grd_idx[0:min_idx],grd_idx[1+min_idx:]])
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],1)
                
                p_fr = int(p_fr)
                tro_id = torch.nonzero(tr_o[p_fr,1,::],as_tuple=True)
                if len(tro_id[0])>0 :
                    tro_min = torch.argmin(torch.abs(tro_id[0]-(t_r//4)))
                else:
                    foo +=0
                    tro_id = torch.nonzero(tr_o[p_fr,0,::],as_tuple=True)
                    tro_min = torch.argmin(torch.abs(tro_id[0]-t_r))

                #print((tro_id[0][tro_min]).cpu())


                cls[f"{p_cls}"]['O_T'].append(orent(tr_o[p_fr,::],tro_id[0][tro_min],tro_id[1][tro_min]))
                cls[f"{p_cls}"]['V_T'].append(velo(tr_o[p_fr,::],tro_id[0][tro_min],tro_id[1][tro_min]))
                cls[f"{p_cls}"]['asc']['gt_r']=np.append(cls[f"{p_cls}"]['asc']['gt_r'],(tro_id[0][tro_min]).cpu())
                cls[f"{p_cls}"]['asc']['gt_a']=np.append(cls[f"{p_cls}"]['asc']['gt_a'],(tro_id[1][tro_min]).cpu())
                cls[f"{p_cls}"]['asc']['pd_r']=np.append(cls[f"{p_cls}"]['asc']['pd_r'],int(p_r//4))
                cls[f"{p_cls}"]['asc']['pd_a']=np.append(cls[f"{p_cls}"]['asc']['pd_a'],int(p_c//4))



                velo_p = velo(pr_o[p_fr,::],r=int(p_r//4),a=int(p_r//4),pred=False,rd=rd_map[p_fr,::])
                #cls[f"{p_cls}"]['O_P'].append(orent(tr_o[p_fr,::],tro_id[0][tro_min],tro_id[1][tro_min],velo_p,pred=False))
                cls[f"{p_cls}"]['O_P'].append(orent(pr_o[p_fr,::],int(p_r//4),int(p_c//4),velo_p,pred=True))
                cls[f"{p_cls}"]['V_P'].append(velo_p)
                

                
            else:
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)

        else:
    
            cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)
    if foo>0:print(foo)
    return cls

def offset_peaks (pred_c,pred_map,device,thresh):
    r_off = pred_c[:,0,:,:]
    a_off = pred_c[:,1,:,:]
    pred_idx =torch.Tensor().to(device =device)
    pred_int = torch.Tensor().to(device =device)

    frame_size = pred_c.shape[0]
    r_idx = torch.where(torch.abs(r_off)<0.5)
    dummy_a = torch.ones((frame_size,256,256)).to(device=device)
    dummy_a[r_idx] = a_off[r_idx]
    a_idx = torch.where(torch.abs(dummy_a)<0.5)
    print(len(a_idx[0]))
    for i in range(3):
        cls_map = pred_map[:,i,:,:]
        dummy = torch.zeros((frame_size,256,256)).to(device=device)
        dummy[a_idx]= cls_map[a_idx]
        cls_idx = torch.where(dummy>thresh)
        print(cls_idx)
        print(len(cls_idx[0]))
        for j in range(len(cls_idx[0])):
            dummy_idx = torch.zeros((1,4)).to(device=device)
            dummy_idx[0,0]= cls_idx[0][j]
            dummy_idx[0,1]=i
            dummy_idx[0,2],dummy_idx[0,3] = cls_idx[1][j],cls_idx[2][j] 
            #print(dummy_idx)
            pred_idx = torch.cat([pred_idx,dummy_idx])
        
        pred_int = torch.cat([pred_int,cls_map[cls_idx]])

    idx_list = torch.argsort(pred_int)
    idx_list = torch.flip(idx_list,dims=[-1])

    return pred_idx[idx_list],pred_int[idx_list]

        

def update_peaks(pred_cen,pred_idx):
    for cnt,cord in enumerate(pred_idx):
        r_id = int(cord[2])
        a_id = int(cord[3])
        bat = int(cord[0])
        cord[2] += int(pred_cen[bat,0,r_id,a_id])
        cord[3] += int(pred_cen[bat,1,r_id,a_id])
        pred_idx[cnt]= cord
        #print(int(8*pred_cen[bat,0,r_id,a_id]))

    return pred_idx

def metrics(grd_map,pred_map,mask,filter,cls,thresh,device):
    
    grd_intent,grd_idx = peaks_detect(grd_map,mask,filter,heat_thresh=0.8)
    grd_idx = distribute(grd_idx,device)
    #grd_idx,grd_intent= association(grd_intent,grd_idx,device)

    pred_intent,pred_idx = peaks_detect(pred_map,mask,filter,heat_thresh=0.1)
    pred_idx= distribute(pred_idx,device)
    pred_idx,pred_intent= association(pred_intent,pred_idx,device)

    cls= validation(grd_idx,pred_idx,pred_intent,cls,dist_thresh=thresh,device=device)

    
    return cls, grd_idx, pred_idx

def metrics_center(grd_map,pred_map,pred_c,mask,filter,tr_o,pred_o,rd_map,cls,thresh,device):
    
    grd_intent,grd_idx = peaks_detect(grd_map,mask,filter,heat_thresh=0.8)
    grd_idx = distribute(grd_idx,device)
    #grd_idx,grd_intent= association(grd_intent,grd_idx,device)

    pred_intent,pred_idx = peaks_detect(pred_map,mask,filter,heat_thresh=0.1)
    pred_idx= distribute(pred_idx,device)
    #pred_idx,pred_intent=offset_peaks(pred_c,pred_map,device,0.1)
    #pred_idx,pred_intent= association(pred_intent,pred_idx,device)
    #print(pred_idx,pred_intent,len(pred_intent),grd_idx)
    pred_idx = update_peaks(pred_c,pred_idx)
    pred_idx,pred_intent= association(pred_intent,pred_idx,device)
    #print(pred_idx_1,pred_intent_1,len(pred_intent_1))


    cls= validation(grd_idx,pred_idx,pred_intent,tr_o,pred_o,rd_map,cls,dist_thresh=thresh,device=device)

    
    return cls  











            

        





