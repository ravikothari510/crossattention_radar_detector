# The code is developed by Ali Kariminezhad on 06.08.2019
# The code finds the peaks within a sequence with OS-CFAR (Ordered-statistics Cell Averaging Constant False Alarm Rate) Algorithm
# Open question: what would be the dimentions of Training and Guard cells. How should they be determined?
# open question: how should we determine the alpha parameter (thereshold scaling factor)?
import numpy as np

def DetectPeaksOSCFAR_2D(x,TrainCell_R,TrainCell_V,GuardCell_R,GuardCell_V,FA_rate):
    
    # x: a nparray of two dimensions. Dimension 1 consists of the range information and dimension 2 consists of the doppler information
    Xshape=x.shape
    X_R_Dim=Xshape[1]
    X_V_Dim=Xshape[0]
    
    Side_Train_R=round(TrainCell_R/2)
    Side_Guard_R=round(GuardCell_R/2)
    
    Side_Train_V=round(TrainCell_V/2)
    Side_Guard_V=round(GuardCell_V/2)
    
    Side_R=Side_Train_R+Side_Guard_R
    Side_V=Side_Train_V+Side_Guard_V
    
    
    alpha_R = TrainCell_R*(FA_rate**(-1/TrainCell_R) - 1) # threshold factor
    alpha_V = TrainCell_V*(FA_rate**(-1/TrainCell_V) - 1) # threshold factor

    
    peaks_ind=list()
    peaks_val=list()
    Threshold=list()

    for j in range(Side_V,X_V_Dim-Side_V): 
        for i in range(Side_R,X_R_Dim-Side_R):
            if j!=j-Side_V+np.argmax(x[j-Side_V:j+Side_V+1,i]):
                continue
            
            if i!=i-Side_R+np.argmax(x[j,i-Side_R:i+Side_R+1]):
                continue
            SortTrainSeq_R = np.msort(x[j,i-Side_R:i+Side_R+1])
            SortTrainSeq_V = np.msort(x[j-Side_V:j+Side_V+1,i])

            Pnoise = ( SortTrainSeq_R[(SortTrainSeq_R.size)//2] +  SortTrainSeq_V[(SortTrainSeq_V.size)//2] ) / 2
             
            Threshold.append(Pnoise*(alpha_R+alpha_V)/2)
            ThresholdA=Pnoise*(alpha_R+alpha_V)/2
            if x[j,i] > ThresholdA:
                peaks_ind.append([j,i])
                peaks_val.append(x[j,i])  
             
    
    peaks_ind = np.array(peaks_ind, dtype=int) 
    Threshold = np.array(Threshold)             
    return peaks_ind