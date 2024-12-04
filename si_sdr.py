from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from Dataloader.Dataloader import EarsDataset
import os
from tqdm import tqdm

blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False)

noisy_data = []
clean_data = []
for i in tqdm(range(len(dataset_TRN))):
    noisy, clean = dataset_TRN[i]
    noisy_data.append(noisy)
    clean_data.append(clean)

# save noisy and clean data in files called /tmp/noisy_data and /tmp/clean_data in a format that is easily readable by the user
import numpy as np
np.save('/tmp/noisy_data', np.array(noisy_data))
np.save('/tmp/clean_data', np.array(clean_data))

# si_sdr = ScaleInvariantSignalDistortionRatio()
# si_sdr(noisy_data, clean_data)