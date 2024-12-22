from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import os

blackhole_path = os.getenv('BLACKHOLE')

if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

print(blackhole_path)
print("Data dir: ", os.path.join(blackhole_path, "EARS-WHAM"))

dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
#train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=batch_size, shuffle=True)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False)
#val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=batch_size, shuffle=True)
print("Both should be larger than 0")
print(len(dataset_TRN))
print(len(dataset_VAL))
