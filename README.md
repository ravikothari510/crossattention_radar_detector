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
To split the dataset in 70% Train, 15% Validation and 15% Test set, run the script ```data_split.py```. This saves the frames list per set in the data directory. 

## Train
The train script ```train.py``` requires the following arguments:
```
  --model {RODNet,Crossatten,RAMP}
                        model name
  --gauss {Gauss,Bivar}
                        Type of gauss
  --frame {1,2,3,4,5}   Number of past frames (max 5)
  --no_class NO_CLASS   Number of classes
  --co CO               Center offset loss activate
  --oren OREN           Heading estimation
  --config CONFIG       Config file path
  --device DEVICE       Either cuda:0 or CPU
  --tag TAG             an unique tag to save results
  --data_dir DATA_DIR   Datset directory
```
Depending on the model type and number of frames used, the training time varies from 3-4 hours on NVIDIA V100 for 80 epochs. 
For custom dataset, make sure to update class imbalance weights and FFT maps normalizing values. The script for helping functions are located in ```Utils/dataset ```.
The training can be tracked with tensorboard. 

## Evaluation 
Evaluation script ```test.py``` prints the class based AP for a given distance threshold (default=2m). These are following results when tested with the above training schedule. 

| Model | Input | GT | AP_2m Ped | AP_1m Ped | AP_2m Cyc | AP_1m Cyc | AP_2m Car | AP_1m Cyc | 
| :---: | :---: | :---: | :---: | :---: | :---: |:---: | :---: |:---: |
| [RODNet-CDC](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.pdf) | RA | Gauss | 0.88 | 0.80 | 0.84 | 0.80 | 0.93 | 0.87 |
| [RAMP-CNN](https://arxiv.org/abs/2011.08981) | RAD | Gauss | 0.89 | 0.82| 0.86 | 0.79 | 0.93 | 0.88 |
| Cross Atten. | RAD | Gauss | 0.92 | 0.81 | 0.85 | 0.82 | 0.96 | 0.86 |
| Cross Atten. + offset loss | RAD | Gauss | 0.91 | 0.82 | 0.86 | 0.84 | 0.94 | 0.86 |
| Bivar Cross Atten. + offset loss | RAD | Bivariate | 0.91 | 0.8 | 0.93 | 0.88 | 0.97 | 0.91 |

## Inference
`inference.py` can be run to check out few examples located in `demo/`. Currently it only supports the cross attention model with orentation and center-offset. The output is the predicted class with location and orentation. 