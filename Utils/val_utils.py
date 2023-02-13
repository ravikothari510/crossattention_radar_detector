import torch
import torch.nn as nn
from tqdm import tqdm 
import numpy as np
from copy import deepcopy




def check_accuracy(loader, model, step, writer, 
                  args, loss_fn):

    model.eval()
    device = args.device

    with torch.no_grad():
        mse_loss =[]
        f_loss = []
        c_loss =[]
        o_loss = []
        for ra, rd, ad, tr_map, tr_c, tr_o, _ in loader:

            ra = ra.to(device)
            rd = rd.to(device)
            ad = ad.to(device)
            tr_map = tr_map.to(device)
            tr_c = tr_c.to(device)
            tr_o = tr_o.to(device)

            pred_center = 0
            pred_orent = 0
            if args.model =='RODNet':
                pred_mask = model(ra)
            else:
                if args.model == 'RAMP':
                    pred_mask = model(ra, rd, ad)
                else:
                    pred_mask, pred_center, pred_orent = model(ra, rd, ad)

            cls_loss = loss_fn['cls'](pred_mask, tr_map)
            
            center_loss = 0
            orent_loss = 0
            if args.co:
                center_loss = loss_fn['center'](pred_center, tr_c)

            if args.oren:
                orent_loss = loss_fn['oren'](pred_orent, tr_o)
        

            loss_fnk = nn.MSELoss(reduction='sum')

            loss = loss_fnk(torch.sigmoid(pred_mask),tr_map)
            #loss = loss_fn(preds,y)
            mse_loss.append(loss.item())
            f_loss.append(cls_loss.item())
            c_loss.append(center_loss.item())
            o_loss.append(orent_loss.item())


    mMSE = torch.mean(torch.Tensor(mse_loss))
    mfocal = torch.mean(torch.Tensor(f_loss))
    mcenter = torch.mean(torch.Tensor(c_loss))
    ml2orent = torch.mean(torch.Tensor(o_loss))
    
    writer.add_scalar('loss/mean MSE',mMSE,global_step= step )
    writer.add_scalar('loss/focal',mfocal,global_step =step)
    writer.add_scalar('loss/center',mcenter,global_step =step)  
    writer.add_scalar('loss/orent',ml2orent,global_step =step)  

    print(f"mean MSE : {mMSE}",f"mean focal : {mfocal}" ,f"mean center loss : {mcenter}",f"mean orent loss : {ml2orent}")

    model.train()
    return mMSE, mfocal



def save_checkpoint(state, filename):
    # print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def evaluation(loader, model, args, thresh=2):
    
    device = args.device

    model.eval()

    cls = dict()
    temp_dict = {'grd_no':0,'Conf':[],'O_P':[],'O_T':[],'TP':[],'P':[],'R':[],'AP':[],'asc':{'gt_r':[],'gt_a':[],'pd_r':[],'pd_a':[]}}
    cls['0'] = deepcopy(temp_dict)
    cls['1'] = deepcopy(temp_dict)
    cls['2'] = deepcopy(temp_dict)

    prev_size = 0

    with torch.no_grad():
        x_c = 0
        x_v = 0

        for idx, (ra, rd, ad, tr_map, tr_c, tr_o, name)\
                             in enumerate(tqdm(loader)):
            
            ra = ra.to(device)
            rd = rd.to(device)
            ad = ad.to(device)

            if args.model =='RODNet':
                x_map = model(ra)
            else:
                if args.model == 'RAMP':
                    x_map = model(ra, rd, ad)
                else:
                    x_map, x_c, x_o = model(ra, rd, ad)

            grd_map = tr_map.to(device=device)
            tr_o = tr_o.to(device=device)
        

            if ra.shape[0] != prev_size:
                mask, peak_cls = create_default(grd_map.size(), kernel_window=(3,5))
                mask = mask.to(device=device)
                peak_cls = peak_cls.to(device=device)
           
            prev_size = ra.shape[0]
    
            pred_map = torch.sigmoid(x_map)
            pred_c = 8*(torch.sigmoid(x_c)-0.5)
 
            cls = metrics_center(grd_map=grd_map, pred_map=pred_map,
                                 pred_c=pred_c, mask=mask, peaks_cls=peak_cls,
                                 tr_o=tr_o, pred_o=x_o, cls=cls,
                                 thresh=thresh, device=args.device)

        for idx in range(len(cls)):
            categ = cls[f"{idx}"]
            P =[]
            R=[]
            TP = 0
            GT = categ['grd_no']
            index = np.argsort(-categ['Conf'])
            for cnt, i in enumerate(index):
                TP += categ['TP'][i]
                P = np.append(P,TP/(cnt+1))
                R = np.append(R, TP/GT)
            categ['P']= P
            categ['R'] = R
            categ['AP'] = np.trapz(P,R)
            prec = np.sum(categ['TP'])
            rec =  len(categ['TP'])
            print(idx, 'AP:',categ['AP'],'P:' f"{prec/rec}" , 'R:',f"{prec/GT}", 'GT:'f" {GT}")
            #print('Heading Accuracy', cls[f"{idx}"]['O_T'],'\n',cls[f"{idx}"]['O_P'])
            cls[f"{idx}"] = categ
        
        model.train()
    return cls


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


def metrics_center(grd_map, pred_map, pred_c, mask, 
                  peaks_cls, tr_o, pred_o, cls,
                  thresh, device):
    
    grd_intent, grd_idx = peaks_detect(grd_map, mask, peaks_cls, heat_thresh=0.8)
    grd_idx = distribute(grd_idx, device)

    pred_intent, pred_idx = peaks_detect(pred_map, mask, peaks_cls, heat_thresh=0.1)
    pred_idx= distribute(pred_idx,device)

    if pred_c!=0:
        pred_idx = update_peaks(pred_c, pred_idx)
    pred_idx,pred_intent= association(pred_intent, pred_idx, device)

    cls= validation(grd_idx, pred_idx, pred_intent, tr_o, pred_o,
                    cls, dist_thresh=thresh, device=device)

    
    return cls  

def peaks_detect(map, mask, peaks_cls, heat_thresh):
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
    w = torch.linspace(-1,1,steps=256) # angular range from -1 radian to 1 radian
    angle_array = torch.arcsin(w)
    range = range_array[int(rng_idx)]
    angle = angle_array[int(agl_idx)]

    return range*torch.sin(angle), range*torch.cos(angle)

def distance (x1,y1,x2,y2):
    #print(f"{x1,y1,x2,y2}\n")
    dx = x1-x2
    dy = y1-y2

    return torch.sqrt(dx**2 + dy**2)

def validation(grd_idx, pred_idx, pred_int, tr_o, 
               pr_o, cls, dist_thresh=2, device='cpu'):

    foo =0

    for grd_cord in grd_idx:
        g_fr, g_cls = grd_cord[0], int(grd_cord[1])
        cls[f"{g_cls}"]['grd_no'] +=1

    while len(pred_int) != 0:
        cord = pred_idx[0]
        p_r, p_c= cord[2], cord[3]
        p_fr, p_cls = cord[0], int(cord[1])
        cls[f"{p_cls}"]['Conf'] = np.append(cls[f"{p_cls}"]['Conf'], pred_int[0].to('cpu').numpy())
        x1, y1 = pol2cord(p_r, p_c)
        dist = torch.Tensor().to(device =device)
        pred_int = pred_int[1:]
        pred_idx = pred_idx[1:]
        
        for grd_cord in grd_idx:
            g_fr, g_cls = grd_cord[0], int(grd_cord[1])
            if g_fr==p_fr and g_cls==p_cls:
                g_r, g_c = grd_cord[2], grd_cord[3]
                x2, y2 = pol2cord(g_r, g_c)
                dist = torch.cat([dist, torch.Tensor([distance(x1, y1, x2, y2)]).to(device=device)])
            else :
                dist = torch.cat([dist,torch.Tensor([100]).to(device=device)]) # dummy distance

        if len(dist)!=0:
            #print(f"{dist}\n")
            if torch.min(dist) < dist_thresh:
                min_idx = torch.argmin(dist)
                t_r = int(grd_idx[min_idx][2])
                grd_idx = torch.cat([grd_idx[0:min_idx], grd_idx[1+min_idx:]])
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'], 1)
                
                p_fr = int(p_fr)
                if pr_o != 0:
                        
                    tro_id = torch.nonzero(tr_o[p_fr, 1, ::], as_tuple=True)
                    if len(tro_id[0])>0 :
                        tro_min = torch.argmin(torch.abs(tro_id[0]-(t_r//4)))
                    else:
                        tro_id = torch.nonzero(tr_o[p_fr, 0, ::], as_tuple=True)
                        tro_min = torch.argmin(torch.abs(tro_id[0] - t_r//4))
    
                    cls[f"{p_cls}"]['O_T'].append(orent(tr_o[p_fr,::], tro_id[0][tro_min], tro_id[1][tro_min]))
                    cls[f"{p_cls}"]['O_P'].append(orent(pr_o[p_fr,::], int(p_r//4), int(p_c//4), pred=True))
        
            else:
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)
        else:
            cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)
    return cls


def orent (orent_map, r, a, velo=0, pred=False):

    s_t = orent_map[0, r, a]
    c_t = orent_map[1, r, a]
    s = s_t/((s_t**2 + c_t**2)**0.5)
    c = c_t/((s_t**2 + c_t**2)**0.5)
    angle = torch.arccos(c).rad2deg()
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
