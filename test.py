import argparse
import yaml

import torch

from Utils.dataloader import load_data
from Utils.transform import data_transform
from Utils.val_utils import evaluation, load_checkpoint
from Models.backbone import get_model



def main():
    parser = argparse.ArgumentParser(description='Evaluate radar detector')
    parser.add_argument('--model', help = 'model name', default='Crossatten',
                        choices=['RODNet', 'Crossatten', 'RAMP'])

    parser.add_argument('--gauss', help = 'Type of gauss', default='Bivar',
                        choices=['Gauss', 'Bivar'])
    parser.add_argument('ckpt', help='model weights in .pth form')
    
    parser.add_argument('--frame', help = 'Number of past frames (max 5)', type=int,
                         default=1, choices=range(1,6))
    parser.add_argument('--no_class', help='Number of classes', default=3)
    parser.add_argument('--thresh', help='Distance threshold for GT association',
                        default=2)

    parser.add_argument('--co', help = 'Center offset loss activate', type=bool, default=1)
    parser.add_argument('--oren', help = 'Heading estimation', type=bool, default=0)
    parser.add_argument('--config', help='Config file path', default='config.yaml')
    parser.add_argument('--device', help='Either cuda:0 or CPU', default='cpu')
    parser.add_argument('--data_dir', help='Datset directory', default='dataset')

    args = parser.parse_args()

    with open(str(args.config),'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    torch.cuda.empty_cache() 
    # Initalise placeholder
    model = get_model(args=args)
    transforms= data_transform()

    _, val_loader = load_data(cfg=cfg,
                              args=args,
                              transform=transforms)

    load_checkpoint(torch.load(args.ckpt), model=model)
    _ = evaluation(loader=val_loader, 
                model=model, args=args, thresh=args.thresh)


if __name__ == '__main__':
    main()