import torch
import torch.nn as nn




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

def get_metric(grd_map, pred_map,
               filter, cls, thresh, device):
    
