import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np

class ConvTasNetDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ConvTasNetDataLoader, self).__init__(*args, **kwargs)

class EarsDataset(Dataset):
    def __init__(self, data_dir, subset="train", fixed_length="shortest", normalize=True, sample_rate=16000):
        self.noisy_dir = os.path.join(data_dir, subset, "noisy")
        self.clean_dir = os.path.join(data_dir, subset, "clean")
        
        # Collect all .wav files stored in subdirectories (p001, p002...)
        self.noisy_files = sorted(glob(os.path.join(self.noisy_dir, "**", "*.wav"), recursive=True))
        self.clean_files = sorted(glob(os.path.join(self.clean_dir, "**", "*.wav"), recursive=True))
        
        # Noisy data and corresponding clean data should match in length 
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch between noisy and clean files"
        
        self.sample_rate = sample_rate
        self.normalize = normalize

        # Length of the data varies a lot, so here we collect all the lengths 
        self.length_distribution = [torchaudio.load(f)[0].shape[1] for f in self.clean_files]
        
        # Determine fixed length
        # 'shortest' will look for the shortest length of all data points and then crop all data at this length
        # 'average' using both crop and padding to fix the length to averge of all data. padding will add 0's if data to short
        # we can also use 'int' to fix the length at any integer
        if fixed_length == "shortest":
            self.fixed_length = min(self.length_distribution)
        elif fixed_length == "average":
            self.fixed_length = int(np.mean(self.length_distribution))
        elif isinstance(fixed_length, int):
            self.fixed_length = fixed_length
        else:
            raise ValueError("fixed_length must be 'shortest', 'average', or an integer")

    def __len__(self): # used in dataloader
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # Load noisy and clean audio
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]
        noisy_waveform, _ = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)

        # option to normalize data using x-mean/var
        if self.normalize:
            noisy_waveform = self._normalize_waveform(noisy_waveform)
            clean_waveform = self._normalize_waveform(clean_waveform)

        # Pad or crop to fixed length
        noisy_waveform = self._pad_or_crop(noisy_waveform, self.fixed_length)
        clean_waveform = self._pad_or_crop(clean_waveform, self.fixed_length)

        return noisy_waveform, clean_waveform

    def _normalize_waveform(self, waveform):
        mean = waveform.mean()
        std = waveform.std()
        normalized_waveform = (waveform - mean) / std
        return normalized_waveform

    def _pad_or_crop(self, waveform, length):
        # Crop if too long
        if waveform.shape[1] > length:
            waveform = waveform[:, :length]
        # Padding if too short
        elif waveform.shape[1] < length:
            padding = length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

# Usage
data_dir = "/Users/benyla/Desktop/Ears_data_resampled/EARS-WHAM"
batch_size = 8
train_dataset = EarsDataset(data_dir, subset="train", fixed_length="average")
train_loader = ConvTasNetDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# for noisy, clean in train_loader:
#     pass