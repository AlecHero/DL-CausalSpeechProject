# Dataloader

## Description
A brief description of how to go from nothing to everything (data ready for model)

## Table of Contents
- [Description](#Description)
- [Download Raw Data](#Download Raw Data)
- [Train, Test, Val Split](#Train, Test, Val Split)
- [Resample](#Resample)
- [Dataloader](#Dataloader)


## Download Raw Data
https://github.com/sp-uhh/ears_benchmark

## Train, Test, Val Split
Run the generate_ears_wham.py script, to generate EARS-WHAM dataset
This will also split the data in Train, Test, Val
python generate_ears_wham.py --data_dir <data_dir> --copy_clean

## Resample
Run Resample.py with correct data paths. Do:
python Resample.py --clean_data_dir /path/to/EARS --noise_data_dir /path/to/WHAM48kHz/high_res_wham/audio

## Dataloader
Import Dataloader.py in code and use as specified in the bottom of the file
