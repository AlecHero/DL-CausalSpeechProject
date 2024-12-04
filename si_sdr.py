from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import os

blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False)

print(dataset_TRN[0])

print(ScaleInvariantSignalDistortionRatio)