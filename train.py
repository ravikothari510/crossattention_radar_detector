
import os
import argparse
import yaml
from tqdm import tqdm 

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Utils.loss import focalloss, centerloss, l2loss
from Utils.dataloader import load_data
from Utils.transform import noise_add, data_transform
from Utils.val_utils import check_accuracy, save_checkpoint, evaluation
from Models.backbone import get_model

'''
args based (rodnet, ramp, cross attention with or wo bivar, with or with center offset, with or without orientation)
common dataloader
modular cross attention 
common evaluation
tensor board 
save weights dir
'''


def get_classweight(cfg):
    ped_w = float(cfg['CLASS_WT']['PED'])
    cyc_w = float(cfg['CLASS_WT']['CYC'])
    car_w = float(cfg['CLASS_WT']['CAR'])
    weight = torch.tensor([ped_w, cyc_w, car_w])
    weights = torch.zeros([3,256,256])

    weights[0,:,:] = weight[0] + weights[0,:,:]
    weights[1,:,:] = weight[1] + weights[1,:,:]
    weights[2,:,:] = weight[2] + weights[2,:,:]

    return weights


def train_fn(loader, model, optimizer, args, cfg,
            scaler, step, writer, loss_fn):
    loop = tqdm(loader)
    train_avg_loss =[]

    for batch_idx, (ra_map, rd_map, ad_map, 
                    target_map, target_center,
                    target_orent, name) in enumerate(loop):
        
        device = args.device

        ra_map = noise_add(ra_map.to(device=device),device=device)
        rd_map = noise_add(rd_map.to(device=device),device=device)
        ad_map = noise_add(ad_map.to(device=device),device=device)
        target_map = target_map.to(device=device)
        target_center = target_center.to(device=device)
        target_orent = target_orent.to(device=device)
        
        # forward pass
        with torch.cuda.amp.autocast():
            
            pred_center = 0
            pred_orent = 0
            if args.model =='RODNet':
                pred_mask = model(ra_map)
            else:
                if args.model == 'RAMP':
                    pred_mask = model(ra_map, rd_map, ad_map)
                else:
                    pred_mask, pred_center, pred_orent = model(ra_map, rd_map, ad_map)

            cls_loss = cfg['LOSS']['WEIGHT_GAUSS']*loss_fn['cls'](pred_mask, target_map)
            
            center_loss = 0
            orent_loss = 0
            if args.co:
                center_loss = cfg['LOSS']['WEIGHT_CENTER']*loss_fn['center'](pred_center, target_center)

            if args.oren:
                orent_loss = cfg['LOSS']['WEIGHT_OREN']*loss_fn['oren'](pred_orent, target_orent)
        

            final_loss = cls_loss + center_loss + orent_loss


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
    parser = argparse.ArgumentParser(description='Train radar detector')
    parser.add_argument('--model', help = 'model name', default='Crossatten',
                        choices=['RODNet', 'Crossatten', 'RAMP'])

    parser.add_argument('--gauss', help = 'Type of gauss', default='Bivar',
                        choices=['Gauss', 'Bivar'])
    
    parser.add_argument('--frame', help = 'Number of past frames (max 5)', type=int,
                         default=1, choices=range(1,6))
    parser.add_argument('--no_class', help='Number of classes', default=3)

    parser.add_argument('--co', help = 'Center offset loss activate', type=bool, default=1)
    parser.add_argument('--oren', help = 'Heading estimation', type=bool, default=0)
    parser.add_argument('--config', help='Config file path', default='config.yaml')
    parser.add_argument('--device', help='Either cuda:0 or CPU', default='cpu')
    parser.add_argument('--tag', help='an unique tag to save results',
                        default='exp1')
    parser.add_argument('--data_dir', help='Datset directory', default='dataset')

    args = parser.parse_args()

    with open(str(args.config),'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    torch.cuda.empty_cache() 
    # Initalise placeholder
    mMSE = 1e10 
    mfocal = 1e10
    writer = SummaryWriter(f"runs/{args.model}_{args.gauss}_{args.tag}")
    step = 0
    model = get_model(args=args)
    cls_weight = get_classweight(cfg=cfg).to(args.device)
    transforms= data_transform()

    lr = float(cfg['SOLVER']['LEARNING_RATE'])
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.1, verbose=True)

    loss = {}
    loss['cls'] = focalloss(cls_weight, alpha=2, beta=0)
    loss['center'] = centerloss()
    loss['oren'] = l2loss()

    train_loader, val_loader = load_data(cfg=cfg,
                                        args=args,
                                        transform=transforms)
    
    scaler = torch.cuda.amp.GradScaler() # mixed precision

    weight_dir = os.path.join('exp', f"{args.model}_{args.gauss}_{args.tag}")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    for epoch in range(cfg['SOLVER']['NUM_EPOCHS']):
        train_fn(loader=train_loader, 
                 model=model,
                 optimizer=optimizer,
                 args=args,
                 cfg=cfg,
                 scaler=scaler,
                 step=step,
                 writer=writer,
                 loss_fn=loss)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        #check accuracy
        mMSE_new, mfocal_new = check_accuracy(loader=val_loader, 
                                             model=model, step=step,
                                             writer=writer, args=args,
                                             loss_fn=loss)
        
        writer.add_scalar('L Rate',optimizer.param_groups[0]['lr'],global_step=step )
        
        step += 1
        scheduler.step(mMSE_new)

        
        
        filename = os.path.join(weight_dir, 'last_epoch.pth.tar')
        save_checkpoint(checkpoint, filename)

        if mMSE_new < mMSE:
            mMSE=mMSE_new
            filename = os.path.join(weight_dir, 'min_mMSE.pth.tar')
            save_checkpoint(checkpoint,filename)
        
        if mfocal_new < mfocal:
            mfocal = mfocal_new
            filename = os.path.join(weight_dir, 'min_mfocal.pth.tar')
            save_checkpoint(checkpoint,filename)
        
        if epoch%5==0:
            _ = evaluation(loader=val_loader, 
                        model=model, args=args)


if __name__ == '__main__':
    main()