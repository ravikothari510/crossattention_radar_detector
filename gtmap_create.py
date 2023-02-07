'''
1. create gt maps of plain gaussian and bi var gaussian
2. additional gt maps for orientation, center offset loss
'''
import os
from tqdm import tqdm
import argparse
import numpy as np
import json
from tqdm import tqdm

from Utils.bi_gauss_update import bi_var_gauss, plain_gauss


def get_co_vec():
    '''
    create a dummy center offset mask of size 2x9x9
    '''
    vect = np.zeros((2,9,9))
    for i in range(9):
        for j in range(9):
            vect[0,i,j] = 4-i
            vect[1,i,j] = 4-j
    return vect




def main():
    parser = argparse.ArgumentParser(description='Create gt maps (bi var gauss,\
                                    normal gauss, orientaion and centeroffset')
    parser.add_argument('--gt_path', help = 'path to preprocessed \
                                         gt_annotation', default='gt_anno.json')
    parser.add_argument('--carrada_dir', help = 'dir of carrada dataset is located',default = 'Carrada' )
    
    args = parser.parse_args()

    with open(args.gt_path, 'r') as f:
        annotate = json.load(f)

    with open(os.path.join(args.carrada_dir, 'validated_seqs.txt'), 'r') as f:
        seqs = [line.strip() for line in f.readlines()]

    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    for seq in tqdm(seqs):
        seq_dir = os.path.join('dataset', 'gt_maps', seq)
        
        if not os.path.exists(seq_dir):
            os.makedirs(os.path.join(seq_dir, 'bivar_gauss'))
            os.makedirs(os.path.join(seq_dir, 'orent'))
            os.makedirs(os.path.join(seq_dir, 'gauss'))
            os.makedirs(os.path.join(seq_dir, 'center'))


        tracks = annotate[seq]['tracks']

        for track in tracks:
            frame = str(track[-1]).zfill(6)

            bi_gauss_map, center_offset, orient_map = bi_var_gauss(annotate[seq][frame],\
                                                                vect=get_co_vec())
                                                            
            std_gauss = plain_gauss(annotate[seq][frame], s_r=15, s_a=15)
            
            np.save(os.path.join(seq_dir, 'bivar_gauss', frame + '.npy'), bi_gauss_map)
            np.save(os.path.join(seq_dir, 'orent', frame + '.npy'), orient_map)
            np.save(os.path.join(seq_dir, 'gauss', frame + '.npy'), std_gauss)
            np.save(os.path.join(seq_dir, 'center', frame + '.npy'), center_offset)



if __name__ == '__main__':
    '''
        1. create gt maps of plain gaussian and bi var gaussian
        2. additional gt maps for orientation, center offset loss
    '''
    main()







