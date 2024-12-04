from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from Dataloader.Dataloader import EarsDataset
import os

blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False)

noisy_data = []
clean_data = []
for i in range(len(dataset_TRN)):
    noisy, clean = dataset_TRN[i]
    noisy_data.append(noisy)
    clean_data.append(clean)

si_sdr = ScaleInvariantSignalDistortionRatio()
si_sdr(noisy_data, clean_data)