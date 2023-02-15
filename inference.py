import argparse
import os
import numpy as np

import torch

from Utils.transform import data_transform
from Models.cross_atten import RadarCross
from Utils.val_utils import load_checkpoint, create_default,\
                        distribute, association, peaks_detect,\
                        update_peaks, orent

'''
Currently works with Cross ateention model with orentation and center offset 
'''


def main():
    parser = argparse.ArgumentParser(description='Infer radar detector')

    parser.add_argument('ckpt', help='model weights in .pth form')
    parser.add_argument('frame_name', help='name of the frame located in demo folder',
                        default=r'2019-09-16-13-25-35_000048')

 
    parser.add_argument('--device', help='Either cuda:0 or CPU', default='cpu')

    args = parser.parse_args()

    model = RadarCross(in_channels=5, n_class=3,
                   center_offset=True, orentation=True)

    load_checkpoint(torch.load(args.ckpt), model=model)
    transform= data_transform()

    ra_map = torch.from_numpy(np.load(os.path.join('demo',\
            'fft_maps', 'ra_map', args.frame_name + '.npy' )))
    ra_map = transform['ra_map'](ra_map)
    ra_map = ra_map.unsqueeze(0)

    rd_map = torch.from_numpy(np.load(os.path.join('demo',\
                'fft_maps', 'rd_map', args.frame_name + '.npy' )))
    rd_map = transform['rd_map'](rd_map)
    rd_map = rd_map.unsqueeze(0)

    ad_map = torch.from_numpy(np.load(os.path.join('demo',\
                'fft_maps', 'ad_map', args.frame_name + '.npy' )))
    ad_map = transform['ad_map'](ad_map)
    ad_map = ad_map.unsqueeze(0)

    model.eval()
    device = args.device

    ra = ra_map.to(device)
    rd = rd_map.to(device)
    ad = ad_map.to(device)

    x_map, x_c, x_o = model(ra, rd, ad)

    pred_map = torch.sigmoid(x_map)
    pred_c = 8*(torch.sigmoid(x_c)-0.5)

    mask, peak_cls = create_default(pred_map.size(), kernel_window=(3,5))
    mask = mask.to(device=device)
    peak_cls = peak_cls.to(device=device)

    pred_intent, pred_idx = peaks_detect(pred_map, mask, peak_cls, heat_thresh=0.1)
    pred_idx= distribute(pred_idx,device)
    pred_idx = update_peaks(pred_c, pred_idx)
    pred_idx, pred_intent= association(pred_intent, pred_idx, device)

    for cnt, cord in enumerate(pred_idx):
        p_r, p_c = cord[2], cord[3]
        p_fr, p_cls = cord[0], int(cord[1])

        print(f"Class: {p_cls}, Confidence: {pred_intent[cnt]},\
            Range: {50*p_r/256}, Angle(deg): {p_c*57.29*2/256 - 57.29},\
            Heading: {orent(x_o[p_r,::],int(p_r//4), int(p_c//4), pred=True)}")

if __name__ == '__main__':
    main()