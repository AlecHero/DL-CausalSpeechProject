import torch
from torch.utils.data import DataLoader, Dataset

## Output size: (BATCH, CHANNELS, LENGTH/Hz)
## Downsampling: https://pytorch.org/audio/main/tutorials/audio_resampling_tutorial.html

class CausalDataloader(DataLoader):
    def __init__(self, causal : bool, *args, **kwargs):
        self.causal = causal
        super(CausalDataloader, self).__init__(*args, **kwargs)

