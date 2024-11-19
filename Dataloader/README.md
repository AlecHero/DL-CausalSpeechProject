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

# example
mkdir /dtu/blackhole/0b/187019/resampled_data
python Dataloader/Resample.py --clean_data_dir /dtu/blackhole/0b/187019/EARS --noise_data_dir /dtu/blackhole/0b/187019/WHAM48kHz/high_res_wham/audio --output_dir /dtu/blackhole/0b/187019/resampled_data
```

## Train, Test, Val Split
Run the generate_ears_wham.py script, to generate EARS-WHAM dataset.
data_dir should be a --output_dir from resample.py
This will also split the data in Train, Test, Val.
```bash
python generate_ears_wham.py --data_dir <data_dir> --copy_clean --sr 16000
```
If you get an AssertionError: sr == args.sr, then you are pointing at the old dataset, not the resampled one. Make sure --data_dir points to the resampled dataset


## Dataloader
Import Dataloader.py in code and use as specified in the bottom of the file
