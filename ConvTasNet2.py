import torchaudio
import torch

if __name__ == "__main__":

        # num_sources (int, optional): The number of sources to split.
        # enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        # enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        # msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        # msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        # msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        # msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        # msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        # msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
    
    model = torchaudio.models.conv_tasnet.ConvTasNet(
        num_sources=2,
        enc_kernel_size=3,  # Reduced from 20 to avoid size mismatch
        enc_num_feats=512,
        msk_kernel_size=3,  # Reduced from 20 to match encoder kernel size
        msk_num_feats=512,
        msk_num_hidden_feats=512,
        msk_num_layers=2,
        msk_num_stacks=8,
        msk_activate="sigmoid"
    )

    # Create fake input data
    # Batch size of 2, 1 channel, 16000 samples (1 second at 16kHz)
    fake_input = torch.randn(4, 1, 16000)
    
    # Pass through model
    separated_sources = model(fake_input)
    
    print(f"Input shape: {fake_input.shape}")
    print(f"Output shape: {separated_sources.shape}")
