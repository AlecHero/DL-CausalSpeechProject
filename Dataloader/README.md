# Dataloader

## Description
A brief description of how to go from nothing to everything (data ready for model)

## Table of Contents
- [Description](#Description)
- [Download Raw Data](#Download_Raw_Data)
- [Resample](#Resample)
- [Train, Test, Val Split](#Train_Test_Val_Split)
- [Dataloader](#Dataloader)


## Download Raw Data
https://github.com/sp-uhh/ears_benchmark

## Resample
Run Resample.py with correct data paths. Do:
```bash
python Resample.py --clean_data_dir /path/to/EARS --noise_data_dir /path/to/WHAM48kHz/high_res_wham/audio --output_dir /path/to/output/dir
```

## Train, Test, Val Split
Run the generate_ears_wham.py script, to generate EARS-WHAM dataset.
Use --output_dir as data_dir here, if you want to use the resampled data.
This will also split the data in Train, Test, Val.
```bash
python generate_ears_wham.py --data_dir <data_dir> --copy_clean
```


## Dataloader
Import Dataloader.py in code and use as specified in the bottom of the file
