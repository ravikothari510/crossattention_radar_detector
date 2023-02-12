
import torch
import torch.nn as nn
import torch.nn.functional as F



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
   
    def forward(self,pred_o,target_o):
        ep = 1e-10
        target_s= target_o[:,0,::].flatten() # sine 
        target_c= target_o[:,1,::].flatten() # cosine

        idx = torch.nonzero(target_c)

        
        pred_s= pred_o[:,0,::].flatten()
        pred_c= pred_o[:,1,::].flatten()

        loss = (F.mse_loss(pred_s[idx],target_s[idx],reduction='sum')
                +F.mse_loss(pred_c[idx],target_c[idx],reduction='sum'))

        return loss.sum()
