from albumentations.pytorch import ToTensorV2
import albumentations as A
from numpy.core.fromnumeric import mean
import torch
import numpy as np
from torch._C import device
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_mid_fusion_5_velo import RODnet
import sys
from torch.utils.tensorboard import SummaryWriter

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_numpy,
    evaluation,
    focalloss,
    centerloss,
    noise_add,
    save_features,
    l2loss,
)

# Hyperparameters 
LEARNING_RATE = 1e-4
W1,W2,W3 = 1,1,10 #(focal loss, center loss, l2 loss)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEVICE ="cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 80
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_PICS = True
EVAL = False
SAVE_FEATURE = False

TRAIN_RA_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_ra_map'
TRAIN_RD_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_rd_map'
TRAIN_AD_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_ad_map'
TRAIN_MASK_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_mask'
TRAIN_CENTER_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_center'
TRAIN_ORENT_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/train_orent'

VAL_RA_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_ra_map'
VAL_RD_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_rd_map'
VAL_AD_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_ad_map'
VAL_MASK_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_mask'
VAL_CENTER_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_center'
VAL_ORENT_DIR = r'/home/ravikothari/dev/bi_variate_norm_latest/val_orent'
# TRAIN_RA_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/train_ra_map'
# TRAIN_RD_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/train_rd_map'
# TRAIN_AD_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/train_ad_map'
# TRAIN_MASK_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/train_mask'
# VAL_RA_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/val_ra_map'
# VAL_RD_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/val_rd_map'
# VAL_AD_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/val_ad_map'
# VAL_MASK_DIR = r'/home/ub273/radar_ravi/dev/bi_variate_norm/val_mask'




def train_fn(loader, model, optimizer, scaler,step,writer,loss_fn,loss_cn,loss_l2):
    loop = tqdm(loader)
    train_avg_loss =[]

    for batch_idx , (ra_map,rd_map,ad_map,target_map,target_center,target_orent,name) in enumerate(loop):
        
        ra_map = noise_add(ra_map.to(device=DEVICE),device=DEVICE)
        rd_map = noise_add(rd_map.to(device=DEVICE),device=DEVICE)
        ad_map = noise_add(ad_map.to(device=DEVICE),device=DEVICE)
        target_map = target_map.to(device=DEVICE)
        target_center = target_center.to(device=DEVICE)
        target_orent = target_orent.to(device=DEVICE)
        
        # forward pass
        with torch.cuda.amp.autocast():
            pred_mask,pred_center,pred_orent,pred_velo = model(ra_map,rd_map,ad_map)
            #print(targets.shape, predictions.shape)
            focal_loss = loss_fn(pred_mask, target_map)
            center_loss = loss_cn(pred_center, target_center)
            orent_loss = loss_l2(pred_orent,target_orent[:,:2,:,:],pred_velo,target_orent[:,2,:,:])
        

            final_loss = W1*focal_loss + W2*center_loss + W3*orent_loss

        # #backward with cpu
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


        # backward
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_avg_loss.append(final_loss.item())
        # update tqdm
        loop.set_postfix(loss=final_loss.item())

    print(f"Mean Training loss : {torch.mean(torch.Tensor(train_avg_loss))}") 
    writer.add_scalar('Train Loss',torch.mean(torch.Tensor(train_avg_loss)),global_step=step )   






def main():
    mMSE = 1e10
    mfocal = 1e10
    writer = SummaryWriter(f"runs/mid_fusion_velo_1_{LEARNING_RATE}_{W1}_{W2}_{W3}")
    step = 0
    torch.cuda.empty_cache() 
    ra_transform = transforms.Compose([
        transforms.Normalize(
            mean = [678.5],
            std = [1352.6]

        )])
    rd_transform = transforms.Compose([
        transforms.Normalize(
            mean = [34.1],
            std = [2.93]

        )])
             
    ad_transform = transforms.Compose([
        transforms.Normalize(
            mean = [35.3],
            std = [2.07]

        )])

    model = RODnet(in_channels=5,n_class=3).to(DEVICE)
    #weight= torch.tensor([1.2033898305084745, 2.969465648854962 ,0.9259259259259259])
    weight= torch.tensor([1.33, 2.81 ,0.86])
    weights = torch.zeros([3,256,256])
    weights[0,:,:] = weight[0] + weights[0,:,:]
    weights[1,:,:] = weight[1] + weights[1,:,:]
    weights[2,:,:] = weight[2] + weights[2,:,:]
    weights = weights.to(DEVICE)

    loss_fl = focalloss(weights,alpha=2,beta=0)
    loss_cn = centerloss()
    loss_l2 = l2loss()

    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4,factor=0.1,verbose=True)

    train_loader, val_loader= get_loaders(
        TRAIN_RA_DIR,
        TRAIN_RD_DIR,
        TRAIN_AD_DIR,
        TRAIN_MASK_DIR,
        TRAIN_CENTER_DIR,
        TRAIN_ORENT_DIR,
        VAL_RA_DIR,
        VAL_RD_DIR,
        VAL_AD_DIR,
        VAL_MASK_DIR,
        VAL_CENTER_DIR,
        VAL_ORENT_DIR,
        BATCH_SIZE,
        ra_transform,
        rd_transform,
        ad_transform,
        NUM_WORKERS,

    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(r'min_mMSE.pth.tar'),model)
        #load_checkpoint(torch.load('min_mMSE.pth.tar'),model)
        
        if EVAL:
            eval = evaluation(val_loader,model,device=DEVICE)
            np.save('data_3fr.npy',eval)
            sys.exit()
        
        
    # save some examples to folder
        if SAVE_PICS :
            save_predictions_as_numpy(
            val_loader, model, folder="saved_output/5fr/", device=DEVICE
            )
            sys.exit()
        
        if SAVE_FEATURE :
            save_features(
            val_loader, model, device=DEVICE)
            sys.exit()

        _=check_accuracy(val_loader,model, step,writer,loss_fl,loss_cn,loss_l2,device=DEVICE)

        sys.exit()
  
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        #print(1)
        #sys.exit()
        train_fn(train_loader, model, optimizer, scaler,step,writer,loss_fl,loss_cn,loss_l2)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        

        #check accuracy
        mMSE_new,mfocal_new = check_accuracy(val_loader,model, step,writer,loss_fl,loss_cn,loss_l2,device=DEVICE)
        
        writer.add_scalar('L Rate',optimizer.param_groups[0]['lr'],global_step=step )
        
        step += 1
        scheduler.step(mMSE_new)

        filename="last_epoch.pth.tar"
        save_checkpoint(checkpoint,filename)
        if mMSE_new<mMSE:
            mMSE=mMSE_new
            filename = "min_mMSE.pth.tar"
            save_checkpoint(checkpoint,filename)
        if mfocal_new<mfocal:
            mfocal = mfocal_new
            filename = "min_mfocal.pth.tar"
            save_checkpoint(checkpoint,filename)






if __name__ == "__main__":
    main()







