"""Implements Conv-TasNet with building blocks of it.

Based on https://github.com/naplab/Conv-TasNet/tree/e66d82a8f956a69749ec8a4ae382217faa097c5c
"""

from typing import Optional, Tuple

import torch


class causal_index(torch.nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding
    
    def forward(self, x):
        return x[..., :-self.padding]


# class CumulativeLayerNorm(torch.nn.LayerNorm):
#     def __init__(self, normalized_shape):
#         super().__init__(normalized_shape=normalized_shape, elementwise_affine=True)

#     def forward(self, x):
#         x = torch.transpose(x, 1, 2)
#         x = super().forward(x)
#         x = torch.transpose(x, 1, 2)
#         return x

class ConvBlock(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int, optional): Dilation value of the convolution in the middle layer.
        no_redisual (bool, optional): Disable residual block/output.
        dropout (float, optional): Dropout probability.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int = 1,
        no_residual: bool = False,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation if causal else (kernel_size - 1) // 2  # causal and non causal padding
    
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            (causal_index(padding=padding) if causal else torch.nn.Identity()), # Do nothing here if not causal
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Dropout(p=dropout),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.
        dropout (float): Dropout probability.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
        causal: bool = True,
        save_intermediate_values: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources
        self.save_intermediate_values = save_intermediate_values

        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                        causal=causal,
                        dropout=dropout
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_dim * num_sources,
            kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        intermediate_values = []  # Collect intermediate values if enabled
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
            if self.save_intermediate_values:  # Save the skip connections for loss
                intermediate_values.append(skip)
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        if self.save_intermediate_values:
            return output.view(batch_size, self.num_sources, self.input_dim, -1), intermediate_values
        else:
            return output.view(batch_size, self.num_sources, self.input_dim, -1)


class ConvTasNet(torch.nn.Module):
    """Conv-TasNet architecture introduced in
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    :cite:`Luo_2019`.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
        dropout (float, optional): Dropout probability (Default: 0.0).
    """

    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
        causal: bool = True,
        save_intermediate_values: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.save_intermediate_values = save_intermediate_values

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            causal=causal,
            save_intermediate_values=save_intermediate_values,
            dropout=dropout
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        padded, num_pads = self._align_num_frames_with_strides(input)
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)
        
        if self.save_intermediate_values:
            masked, intermediate_values = self.mask_generator(feats)
        else:
            masked = self.mask_generator(feats)
        
        masked = masked.view(batch_size * self.num_sources, self.enc_num_feats, -1)
        decoded = self.decoder(masked)
        output = decoded.view(batch_size, self.num_sources, num_padded_frames)
        if num_pads > 0:
            output = output[..., :-num_pads]
        if self.save_intermediate_values:
            return output, intermediate_values
        else:
            return output


def conv_tasnet_base(num_sources: int = 2) -> ConvTasNet:
    r"""Builds non-causal version of :class:`~torchaudio.models.ConvTasNet`.

    The parameter settings follow the ones with the highest Si-SNR metirc score in the paper,
    except the mask activation function is changed from "sigmoid" to "relu" for performance improvement.

    Args:
        num_sources (int, optional): Number of sources in the output.
            (Default: 2)
    Returns:
        ConvTasNet:
            ConvTasNet model.
    """
    return ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=16,
        enc_num_feats=512,
        msk_kernel_size=3,
        msk_num_feats=128,
        msk_num_hidden_feats=512,
        msk_num_layers=8,
        msk_num_stacks=3,
        msk_activate="relu",
    )

if __name__ == "__main__":
    import torch

    # Initialize model parameters
    num_sources = 2  # Number of sources to separate
    enc_kernel_size = 16  # Encoder kernel size
    enc_num_feats = 512  # Number of encoder feature dimensions
    msk_kernel_size = 3  # Mask generator kernel size
    msk_num_feats = 128  # Mask generator input/output feature dimension
    msk_num_hidden_feats = 512  # Mask generator internal feature dimensions
    msk_num_layers = 8  # Number of layers in each mask generator stack
    msk_num_stacks = 3  # Number of stacks in the mask generator
    msk_activate = "relu"  # Activation function for the mask generator

    # Instantiate the ConvTasNet model
    model = ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate,
        causal=True,
        dropout=0.1  # Example dropout value
    )

    # Print the model architecture
    # print(model)

    # Generate dummy input data for testing
    batch_size = 4
    
    input_length = 16000  # Example input signal length (frames)
    dummy_input = torch.randn(batch_size, 1, input_length)

    # Pass the dummy input through the model
    output = model(dummy_input)

    # Print the output shape
    print(f"Output shape: {output.shape}")
    print(batch_size, num_sources, input_length)
    # Expected output shape: [batch_size, num_sources, output_length]