import os
import argparse
import json
from tqdm import tqdm
import numpy as np



def get_admap(rad_file_path):
    rad_map = np.load(rad_file_path)
    ad_map = np.fft.ifftshift(rad_map, axes=0)
    ad_map = np.fft.ifft(ad_map, axis=0)
    ad_map = pow(np.abs(ad_map), 2)
    ad_map = np.sum(ad_map, axis=0)
    ad_map = 10*np.log10(ad_map + 1)
    ad_map = np.transpose(ad_map)
    ad_map[31:34,:]=0

    return np.float32(ad_map)


def main():
    parser = argparse.ArgumentParser(description='CARRADA Dataset preprocessing')
    parser.add_argument('--carrada_dir', help = 'dir of carrada dataset is located',default = 'Carrada' )
    parser.add_argument('--raw_carrada', help='RAD tensors of carrada', default='Carrada_RAD')
    parser.add_argument('--gt_path', help = 'path to preprocessed \
                                         gt_annotation', default='gt_anno.json')

    args = parser.parse_args()
    
    with open(args.gt_path, 'r') as f:
        annotate = json.load(f)
    
    with open(os.path.join(args.carrada_dir, 'validated_seqs.txt'), 'r') as f:
        seqs = [line.strip() for line in f.readlines()]
    
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    for seq in tqdm(seqs):
        seq_dir = os.path.join('dataset', 'fft_maps', seq)
        
        if not os.path.exists(seq_dir):
            os.makedirs(os.path.join(seq_dir, 'ra_map'))
            os.makedirs(os.path.join(seq_dir, 'rd_map'))
            os.makedirs(os.path.join(seq_dir, 'ad_map'))

        
        tracks = annotate[seq]['tracks']

        for track in tracks:
            ra_map = np.zeros((5, 256, 256))
            rd_map = np.zeros((5, 256, 64))
            ad_map = np.zeros((5, 64, 256))
            for i in range(5):
                frame = str(track[i]).zfill(6)
                ra_map[i, ::] = np.load(os.path.join(args.carrada_dir, seq,\
                                         'range_angle_numpy', frame + '.npy'))
                
                rd_map_temp = np.load(os.path.join(args.carrada_dir, seq,\
                                        'range_doppler_numpy', frame + '.npy'))

                # flip the axis in the order of increasing velocity (L->R)
                rd_map_temp = np.flip(rd_map_temp, 0)
                rd_map_temp = np.flip(rd_map_temp, 1)

                rd_map_temp[:, 31:34]=0 # removing dc component
                rd_map[i, ::] = np.float32(rd_map_temp)

                rad_file_path = os.path.join(args.raw_carrada, seq, 'RAD_numpy'\
                                            ,frame + '.npy')
                ad_map[i, ::] = get_admap(rad_file_path=rad_file_path)
            
            np.save(os.path.join(seq_dir, 'ra_map', str(track[-1]).zfill(6) + '.npy'), ra_map)
            np.save(os.path.join(seq_dir, 'rd_map', str(track[-1]).zfill(6) + '.npy'), rd_map)
            np.save(os.path.join(seq_dir, 'ad_map', str(track[-1]).zfill(6) + '.npy'), ad_map)

if __name__ =='__main__':
    '''
        create dataset based on gt annotation tracks
        stack ra maps
        create rd and ad maps and then stack them
        save the npy files in a dataset
    '''
    main()
