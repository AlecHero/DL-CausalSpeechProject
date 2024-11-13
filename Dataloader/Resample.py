import os
import torchaudio
from glob import glob
from tqdm import tqdm
import argparse

def resample_directory(input_dir, output_dir, target_sr=16000):
    # If the specified output_dir does not already exist, it is created
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if the input directory contains subdirectories or files directly
    # The clean and noise data has different structure, so we need this to handle both
    has_subdirs = any(os.path.isdir(os.path.join(input_dir, entry)) for entry in os.listdir(input_dir))

    if has_subdirs:
        for speaker_folder in os.listdir(input_dir):
            speaker_dir = os.path.join(input_dir, speaker_folder)
            if os.path.isdir(speaker_dir):
                output_speaker_dir = os.path.join(output_dir, speaker_folder)
                os.makedirs(output_speaker_dir, exist_ok=True)
                for file_path in tqdm(glob(os.path.join(speaker_dir, "*.wav")), desc=f"Resampling {speaker_folder}"):
                    resample_and_save(file_path, output_speaker_dir, target_sr)
    else:
        for file_path in tqdm(glob(os.path.join(input_dir, "*.wav")), desc="Resampling noise data"):
            resample_and_save(file_path, output_dir, target_sr)

def resample_and_save(file_path, output_dir, target_sr): # resample using torchaudio
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Save resampled file and keep the same file names
    file_name = os.path.basename(file_path)
    torchaudio.save(os.path.join(output_dir, file_name), waveform, target_sr)

def main():
    parser = argparse.ArgumentParser(description="Resample audio files in specified directories.")
    parser.add_argument("--clean_data_dir", type=str, required=True, help="Path to the clean data directory.")
    parser.add_argument("--noise_data_dir", type=str, required=True, help="Path to the noise data directory.")
    args = parser.parse_args()

    # Define output directories based on the current directory
    base_output_dir = os.path.join(os.getcwd(), "Ears_data_resampled")
    resampled_clean_data_dir = os.path.join(base_output_dir, "EARS")
    resampled_noise_data_dir = os.path.join(base_output_dir, "WHAM48kHz/high_res_wham/audio")

    # Resample the clean and noise data
    resample_directory(args.clean_data_dir, resampled_clean_data_dir, target_sr=16000)
    resample_directory(args.noise_data_dir, resampled_noise_data_dir, target_sr=16000)

if __name__ == "__main__":
    main()
