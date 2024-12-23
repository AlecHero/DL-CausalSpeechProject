import pickle
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import librosa

def load_audio_files(file_paths):
    audio_data = {}
    for file_path in file_paths:
        try:
            data, sr = librosa.load(file_path, sr=None)
            audio_data[file_path] = {"data": data, "sample_rate": sr}
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return audio_data

dataset_TRN = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'train', normalize = False, max_samples=100)
dataset_VAL = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'valid', normalize = False, max_samples=100)

clean_trn_files = dataset_TRN.clean_files
clean_val_files = dataset_VAL.clean_files
noisy_trn_files = dataset_TRN.noisy_files
noisy_val_files = dataset_VAL.noisy_files

clean_trn_data = load_audio_files(clean_trn_files)
clean_val_data = load_audio_files(clean_val_files)
noisy_trn_data = load_audio_files(noisy_trn_files)
noisy_val_data = load_audio_files(noisy_val_files)

print(f"Length of clean_trn_files: {len(clean_trn_files)}")
print(f"Length of clean_val_files: {len(clean_val_files)}")
print(f"Length of noisy_trn_files: {len(noisy_trn_files)}")
print(f"Length of noisy_val_files: {len(noisy_val_files)}")

# Save the file lists to a pickle file
file_data = {
    "clean_trn_files": clean_trn_files,
    "clean_val_files": clean_val_files,
    "noisy_trn_files": noisy_trn_files,
    "noisy_val_files": noisy_val_files
}

# Save to a file
with open("sound_file_lists_new.pkl", "wb") as f:
    pickle.dump(file_data, f)
print("Sound file lists saved to 'sound_file_lists.pkl'.")