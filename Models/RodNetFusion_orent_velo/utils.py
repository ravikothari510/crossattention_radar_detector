from numpy.core.fromnumeric import std
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from dataset import Carrada
from torch.utils.data import DataLoader
import os
from PIL import Image
from tqdm import tqdm
import os
from post_process_numpy_neu import metrics_center, create_default
#from post_process_numpy import metrics

def save_checkpoint(state,filename):
    # print("=> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_ra_dir,
    train_rd_dir,
    train_ad_dir,
    train_maskdir,
    train_center_dir,
    train_orent_dir,
    val_ra_dir,
    val_rd_dir,
    val_ad_dir,
    val_maskdir,
    val_center_dir,
    val_orent_dir,
    batch_size,
    ra_transform,
    rd_transform,
    ad_transform,
    num_workers = 2,
    pin_memory = True,):

    train_ds = Carrada(
        ra_map_dir=train_ra_dir,
        rd_map_dir=train_rd_dir,
        ad_map_dir=train_ad_dir,
        mask_dir= train_maskdir,
        center_dir=train_center_dir,
        orent_dir = train_orent_dir,
        ra_transform=ra_transform,
        rd_transform=rd_transform,
        ad_transform=ad_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Carrada(
        ra_map_dir=val_ra_dir,
        rd_map_dir=val_rd_dir,
        ad_map_dir=val_ad_dir,
        mask_dir= val_maskdir,
        center_dir=val_center_dir,
        orent_dir = val_orent_dir,
        ra_transform=ra_transform,
        rd_transform=rd_transform,
        ad_transform=ad_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model,step,writer,focal_loss,center_loss,loss_l2,device="cuda"):

    model.eval()

    with torch.no_grad():
        mse_loss =[]
        f_loss = []
        c_loss =[]
        o_loss = []
        for ra,rd,ad,tr_map,tr_c,tr_o,_ in loader:

            ra = ra.to(device)
            rd = rd.to(device)
            ad = ad.to(device)
            tr_map = tr_map.to(device)
            tr_c = tr_c.to(device)
            tr_o = tr_o.to(device)
            
            pred_map,pred_c,pred_o,pred_v = model(ra,rd,ad)

            focalloss_cust = focal_loss(pred_map,tr_map)
            centerloss = center_loss(pred_c,tr_c)

            loss_orent = loss_l2(pred_o,tr_o[:,:2,:,:],pred_v,tr_o[:,2,:,:])

            loss_fnk = nn.MSELoss(reduction='sum')

            loss = loss_fnk(torch.sigmoid(pred_map),tr_map)
            #loss = loss_fn(preds,y)
            mse_loss.append(loss.item())
            f_loss.append(focalloss_cust.item())
            c_loss.append(centerloss.item())
            o_loss.append(loss_orent.item())


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

def noise_add(r_map,device):
    noise_map = torch.normal(0,0.1,size=r_map.shape).to(device=device)
    return r_map+noise_map

def save_features(loader, model, folder="saved_feature/", device ="cuda"):
    dir = r'/home/ravikothari/dev/camera/val'
    model.eval()
    name_list = dict()

    for idx, (ra,rd,ad,tr_map,tr_c,name) in tqdm(enumerate(loader)):
        
        if idx>1 and idx<3:
            ra = ra.to(device)
            rd = rd.to(device)
            ad = ad.to(device)
            with torch.no_grad():
                name_list[f"{idx}"]={}
                pred_map,pred_c,rd_feature,ad_feature = model(ra,rd,ad)
                ra = ra.to("cpu").numpy()
                rd = rd.to("cpu").numpy()
                ad = ad.to("cpu").numpy()
                #ra_feature = ra_feature.to("cpu").numpy()
                rd_feature = rd_feature.to("cpu").numpy()
                ad_feature = ad_feature.to("cpu").numpy()
                #rd_b = rd_b.to("cpu").numpy()
                #ad_b = ad_b.to("cpu").numpy()
                #ra_attention = ra_attention.to("cpu").numpy()
                tr_map = tr_map.to("cpu").numpy()
                tr_c = tr_c.to("cpu").numpy()

                pred_map = torch.sigmoid(pred_map).to("cpu").numpy()
                pred_c = torch.sigmoid(pred_c).to("cpu").numpy()
                
                np.save(os.path.join(folder,f"{idx}_preds_map.npy"),pred_map)
                np.save(os.path.join(folder,f"{idx}_preds_c.npy"),pred_c)

                np.save(os.path.join(folder,f"{idx}_grd_map.npy"),tr_map)
                np.save(os.path.join(folder,f"{idx}_grd_c.npy"),tr_c)

                np.save(os.path.join(folder,f"{idx}_ra_map.npy"),ra)
                np.save(os.path.join(folder,f"{idx}_rd_map.npy"),rd)
                np.save(os.path.join(folder,f"{idx}_ad_map.npy"),ad)
                
                #np.save(os.path.join(folder,f"{idx}_ra_feature.npy"),ra_feature)
                np.save(os.path.join(folder,f"{idx}_rd_feature.npy"),rd_feature)
                np.save(os.path.join(folder,f"{idx}_ad_feature.npy"),ad_feature)
                #np.save(os.path.join(folder,f"{idx}_ra_attention.npy"),ra_attention)
                #np.save(os.path.join(folder,f"{idx}_rd_b.npy"),rd_b)
               # np.save(os.path.join(folder,f"{idx}_ad_b.npy"),ad_b)

                
                for cnt,images in enumerate(name):
                    seq,frame = images.split('_')
                    name_list[f"{idx}"][f"{cnt}"]=f"{images}"
                    path = os.path.join(dir,seq+frame.replace('.npy','.jpg'))
                    img=Image.open(path)
                    img.save(os.path.join(folder,f"{idx}_{cnt}.jpg"))
    np.save('saved_feature/name_list.npy',name_list)

    model.train()


def save_predictions_as_numpy(
    loader, model, folder="saved_output/", device ="cuda"):
    dir = r'/home/ravikothari/dev/camera/val'
    model.eval()
    name_list = dict()

    for idx, (ra,rd,ad,tr_map,tr_c,tr_o,name) in tqdm(enumerate(loader)):
        
        if idx>1 and idx<60:
            ra = ra.to(device)
            rd = rd.to(device)
            ad = ad.to(device)
            with torch.no_grad():
                name_list[f"{idx}"]={}
                pred_map,pred_c,x_o,x_v = model(ra,rd,ad)
                ra = ra.to("cpu").numpy()
                rd = rd.to("cpu").numpy()
                ad = ad.to("cpu").numpy()
                tr_map = tr_map.to("cpu").numpy()
                tr_c = tr_c.to("cpu").numpy()
                tr_o= tr_o.to("cpu").numpy()
                x_o = x_o.to("cpu").numpy()
                x_v = x_v.to("cpu").numpy()

                pred_map = torch.sigmoid(pred_map).to("cpu").numpy()
                pred_c = torch.sigmoid(pred_c).to("cpu").numpy()
                
                np.save(os.path.join(folder,f"{idx}_preds_map.npy"),pred_map)
                np.save(os.path.join(folder,f"{idx}_preds_c.npy"),pred_c)
                np.save(os.path.join(folder,f"{idx}_preds_o.npy"),x_o)
                np.save(os.path.join(folder,f"{idx}_preds_v.npy"),x_v)


                np.save(os.path.join(folder,f"{idx}_grd_map.npy"),tr_map)
                np.save(os.path.join(folder,f"{idx}_grd_c.npy"),tr_c)
                np.save(os.path.join(folder,f"{idx}_grd_o.npy"),tr_o)

                np.save(os.path.join(folder,f"{idx}_ra_map.npy"),ra)
                np.save(os.path.join(folder,f"{idx}_rd_map.npy"),rd)
                np.save(os.path.join(folder,f"{idx}_ad_map.npy"),ad)
                
                for cnt,images in enumerate(name):
                    seq,frame = images.split('_')
                    name_list[f"{idx}"][f"{cnt}"]=f"{images}"
                    path = os.path.join(dir,seq+frame.replace('.npy','.jpg'))
                    #img=Image.open(path)
                    #img.save(os.path.join(folder,f"{idx}_{cnt}.jpg"))
    np.save(folder+'name_list.npy',name_list)

    model.train()
def evaluation(
    loader, model,device ="cuda"):

    model.eval()

    cls = dict()
    cls['0']= {'grd_no':0,'Conf':[],'O_P':[],'O_T':[],'V_P':[],'V_T':[],'TP':[],'P':[],'R':[],'AP':[],'asc':{'gt_r':[],'gt_a':[],'pd_r':[],'pd_a':[]}}
    cls['1']= {'grd_no':0,'Conf':[],'O_P':[],'O_T':[],'V_P':[],'V_T':[],'TP':[],'P':[],'R':[],'AP':[],'asc':{'gt_r':[],'gt_a':[],'pd_r':[],'pd_a':[]}}
    cls['2']= {'grd_no':0,'Conf':[],'O_P':[],'O_T':[],'V_P':[],'V_T':[],'TP':[],'P':[],'R':[],'AP':[],'asc':{'gt_r':[],'gt_a':[],'pd_r':[],'pd_a':[]}}

    cal = True

    with torch.no_grad():

        for idx, (ra,rd,ad,tr_map,tr_c,tr_o,name) in enumerate(tqdm(loader)):
            if idx>0 and idx<76:
                ra = ra.to(device)
                rd = rd.to(device)
                ad = ad.to(device)
                x_map,x_c,x_o,x_v = model(ra,rd,ad)
                grd_map = tr_map.to(device=device)
                tr_o = tr_o.to(device=device)
            
                if cal :
                    cal = False
                    mask, peak_cls = create_default(grd_map.size(),(3,5))
                    mask = mask.to(device=device)
                    peak_cls = peak_cls.to(device=device)
                

            
            
                
                pred_map = torch.sigmoid(x_map)
                pred_c = 8*(torch.sigmoid(x_c)-0.5)
            #preds_npy = preds.to("cpu").numpy()
            #y_1 =y.to("cpu").numpy()
            #cls = metrics(y,preds,cls, thresh=3)
                x_o = torch.cat([x_o,x_v],dim=1)
                cls = metrics_center(grd_map,pred_map,pred_c,mask,peak_cls,tr_o,x_o,rd,cls,thresh=2,device=device)
            #pred_map=pred_map.to('cpu').numpy()
            #grd_map = grd_map.to('cpu').numpy()
            #cls = metrics(grd_map,pred_map,cls,thresh=2)
        
                    #print(name)
        for idx in range(len(cls)):
            categ = cls[f"{idx}"]
            P =[]
            R=[]
            TP = 0
            GT = categ['grd_no']
            index = np.argsort(-categ['Conf'])
            for cnt,i in enumerate(index):
                TP += categ['TP'][i]
                P = np.append(P,TP/(cnt+1))
                R = np.append(R, TP/GT)
            categ['P']= P
            categ['R'] = R
            categ['AP'] = np.trapz(P,R)
            prec = np.sum(categ['TP'])
            rec =  len(categ['TP'])
            print(idx, 'AP:',categ['AP'],'P:' f"{prec/rec}" , 'R:',f"{prec/GT}", 'GT:'f" {GT}")
            #print(cls[f"{idx}"]['O_T'],'\n',cls[f"{idx}"]['O_P'])
            cls[f"{idx}"] = categ
        
        model.train()
    return cls

class  focalloss(nn.Module):
    def __init__(self,weights,alpha,beta):
        super(focalloss,self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.beta = beta
    def forward(self,input,target):

        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.weights * (1-pt)**self.alpha * BCE_loss
        return F_loss.sum()

class centerloss(nn.Module):
    def __init__(self):
        super(centerloss,self).__init__()
    
    def forward(self,pred_map,target_map):

        target_map= target_map.flatten()
        idx = torch.nonzero(target_map)
        target_map = ((target_map/8)+0.5)
        pred_map = pred_map.flatten()
        c_loss = F.binary_cross_entropy_with_logits(pred_map[idx], target_map[idx], reduction='none')
        #c_loss = F.binary_cross_entropy_with_logits(pred_map[idx], target_map[idx], reduction='none')
        return c_loss.sum()

class l2loss(nn.Module):
    def __init__(self):
        super(l2loss,self).__init__()
   
    def forward(self,pred_o,target_o,pred_v,target_v):
        ep = 1e-10
        target_s= target_o[:,0,::].flatten()
        target_c= target_o[:,1,::].flatten()
        target_v= target_v.flatten()

        idx = torch.nonzero(target_c)

        target_v[idx] = torch.abs(target_v[idx])/(target_v[idx]+ep)

        #print(target_v[idx])
        
        pred_s= pred_o[:,0,::].flatten()
        pred_c= pred_o[:,1,::].flatten()
        pred_v= pred_v.flatten()

        loss = (F.mse_loss(pred_s[idx],target_s[idx],reduction='sum')
                +F.mse_loss(pred_c[idx],target_c[idx],reduction='sum')
                +F.mse_loss(pred_v[idx],target_v[idx],reduction='sum'))

        return loss.sum()






