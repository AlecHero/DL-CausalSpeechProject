import torchaudio
import torch

model = torchaudio.models.conv_tasnet(
    num_sources=2,
    num_blocks=8,
    num_layers=8,
    kernel_size=20,
    num_features=512,
    encoder_activation=torch.nn.functional.relu,
    decoder_activation=torch.nn.functional.relu,
)

