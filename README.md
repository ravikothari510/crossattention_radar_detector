# Raw radar based object detector

This repo is an implementation of the paper [Raw Radar data based Object Detection and Heading estimation using Cross Attention](https://arxiv.org/abs/2205.08406). It contains the refined annotations from [CARRADA Dataset](https://github.com/valeoai/carrada_dataset). 

## Installation 
The code has been tested in ubuntu-18 framework with Nvidia GPU. Create and activate virtual environment in python and intall the requirements.  
```
python3 venv -m  path/to/env/radar_cross

source */radar_cross/bin/activate

pip install -r requirements.txt
```
## Dataset

Download the CARRADA dataset from [here](https://arthurouaknine.github.io/codeanddata/carrada), as we use FFT maps as well as the raw data both the folders **Carrada.tar.gz** and **Carrada_RAD.tar.gz** should be downloaded in the working directory. 

The updated annotation are saved in gt_anno.json file. The file structure is as follows:
- scenes
    - frame
        - cls (classes 1=pedestrian, 2=cyclist, 3=car)
        - idx (BBox in RA plane, a1, r1, a2, r2)
        - mu_r (gauss center in range bin)
        - mu_a (gauss center in angle bin)
        - sigma_r (range variance)
        - sigma_a (angle variance)
        - sigma_cov (covariance)
        - velo (velocity m/s)
        - orient (orientaion in degrees)
    - tracks (list of 5 consecutive frames)

Some of the frames are empty as either the objects are too far away (>50m), or too close (<3m) or out of camera frame (thus difficult to verify object position). 

### Dataset preprocess

To pre-process the data run the ```dataset_process.py```. It extracts ra and rd fft maps and stack them in 5 frames, also we compute ad fft from raw tensors and further stack them in dataset folder.

```
python  dataset_process.py --gt_path 'gt_anno.json'\
                           --carrada_dir 'Carrada'\
                           --raw_carrada 'Carrada_RAD'   
```

###  Groundtruth maps
The model needs groundtruth maps such as:
 - Gaussian (3x256x256)
 - heading(2x64x64)
 - center-offset(2x256x256)
 The script saves the maps in ```.npy``` format in dataset folder

```
python gtmap_create.py
```

At the end, the folder structure should look like
```
dataset
    |- fft_maps
        |- sequence_x
            |- ra_map
            |- rd_map
            |- ad_map
    |- gt_maps
        |- sequence_x
            |- bivar_gauss
            |- center
            |- gauss
            |- orent
```


