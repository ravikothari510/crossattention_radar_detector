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

Download the CARRADA dataset form [here](https://arthurouaknine.github.io/codeanddata/carrada), as we use only the FFT maps **Carrada.tar.gz** shoud be sufficient. 