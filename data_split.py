
import os
import random



def write_frame(frame_list, txt_file):
        with open(txt_file, 'a') as f:
            for frame in frame_list:
                f.write(frame)
                f.write('\n')


def main(gt_dir):
    
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15

    frame_list = []

    for seq in os.listdir(gt_dir):
        for frame in os.listdir(os.path.join(gt_dir, seq, 'center')):
            frame_list.append(seq + '_' + frame.removesuffix('.npy'))

    random.shuffle(frame_list)

    train_idx = int(TRAIN_SPLIT*len(frame_list))
    val_idx = train_idx + int(VAL_SPLIT*len(frame_list))

    train_labels = frame_list[:train_idx]
    val_labels = frame_list[train_idx:val_idx]
    test_labels = frame_list[val_idx:]

    write_frame(frame_list=train_labels, txt_file='train_frames.txt')
    write_frame(frame_list=val_labels, txt_file='val_frames.txt')
    write_frame(frame_list=test_labels, txt_file='test_frames.txt')

if __name__ =='__main__':
    main(gt_dir=r'dataset\gt_maps')
    random.seed(1)