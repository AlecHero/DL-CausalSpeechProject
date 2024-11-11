import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, num_filters, kernel_size, stride):
        """
        Parameters:
        - num_filters (int): (Number of features)
        - kernel_size (int): Length of each filter, L
        - stride (int): Controls the overlap, we use 50% overlap? : kernel_size//2
        """
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters,
                                kernel_size=kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor: 
        # This function returns w = H(xU) as defined in the paper
        # We apply the 1D convolution operation to input x.
        # xU is the output of this convolution
        # H is the optional nonlinear function, in paper ReLu
        return F.relu(self.conv1d(x)) # Output shape: [batch_size, num_filters, output_length]
    

    
class Decoder(nn.Module):
    def __init__(self, num_filters, kernel_size, stride):
        """
        Parameters:
        - num_filters (int): Should match the number of encoder filters
        - kernel_size (int): Length of each feature in the decoder filters, same as in the encoder
        - stride (int): Use same value from the encoder
        """
        super(Decoder, self).__init__()
        self.deconv1d = nn.ConvTranspose1d(in_channels=num_filters, out_channels=1,
                                           kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # output_length corresponds to the original signal length due to
        # the transposed convolution with stride and kernel matching the encoder.
        return self.deconv1d(x) # Output shape: [batch_size, 1, output_length]


class TCNBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, dilation, skip_channels, residual_channels):
        """
        Parameters:
        - num_channels (int): Input channels for the main convolution in the TCN block.        
        - kernel_size (int): Only used in the dilated convolution. kernel_size = 1 by default
          for all other convolutions in the TCN block
        - dilation (int): The dilation rate for the convolution in this block
        - skip_channels (int): The number of channels in the skip connection output
        - residual_channels (int): The number of output channels for the residual connection.
          Should always math input channels, so we can add origional input to the residual convolution
        """
        super(TCNBlock, self).__init__()

        # 1D convolution, PReLu and Norm
        self.conv1x1_in = nn.Conv1d(num_channels, residual_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(residual_channels)
        self.prelu1 = nn.PReLU()
        
        # 1D convolution with dilation, PReLu and Norm
        # For long-term dependencies
        # padding = (kernel_size - 1) * dilation // 2
        # self.dilated_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size, 
        #     padding=padding, dilation=dilation)
        self.dilated_conv = nn.Conv1d(
            residual_channels, residual_channels, kernel_size,
            padding='same', dilation=dilation)
        self.norm2 = nn.BatchNorm1d(residual_channels)
        self.prelu2 = nn.PReLU()

        # keeping the shape and channel dimensions intact, so we can add the origional input in forward()
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

        # computes skip features. 
        # This is done for each TCN block, so features from each block 
        # can contribute to the final output
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]: # Figure C in the paper
        out = self.conv1x1_in(x)
        out = self.prelu1(self.norm1(out))
        out = self.dilated_conv(out)
        out = self.prelu2(self.norm2(out))
        residual = self.res_conv(out) + x 
        skip = self.skip_conv(out)
        return residual, skip


class TCN(nn.Module):
    def __init__(self, num_channels, num_blocks, kernel_size, skip_channels, residual_channels):
        """
        Parameters:
        - num_channels (int):
        - num_blocks (int): Controls how many TCN Blocks we include in the model
        - kernel_size (int):
        - skip_channels (int): The number of channels in the skip connection output
        - residual_channels (int): The number of output channels for the residual connection.
          Should always math input channels, so we can add origional input to the residual convolution
        """
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList(
            # 2**i makes the dilation double for each TCN block
            [TCNBlock(num_channels, kernel_size, 2**i, skip_channels, residual_channels) for i in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []
        for block in self.blocks:
            # x is the updated representation after passing through the current block, 
            # which will then be fed into the next block.
            x, skip = block(x)
            skip_connections.append(skip)
        return torch.sum(torch.stack(skip_connections), dim=0)



class ConvTasNet(nn.Module): # final model
    def __init__(self, num_speakers, num_filters, kernel_size, stride, num_channels, num_blocks, skip_channels, residual_channels):
        """
        Parameters:
        - num_speakers (int): Number of speakers to separate (2 for two speakers and so on).
        - num_filters (int): Number of filters in the encoder and decoder
        - kernel_size (int): Length of each filter in the encoder and decoder.
        - stride (int): Stride length in the encoder and decoder
        - num_channels (int): Number of channels in each TCN block
        - num_blocks (int): Number of TCN blocks stacked together
        - skip_channels (int): The number of channels in the skip connection output
        - residual_channels (int): The number of output channels for the residual connection.
        """
        super(ConvTasNet, self).__init__()
        self.encoder = Encoder(num_filters, kernel_size, stride)

        # Encoder outputs [batch_size, num_filters, sequence_length]
        # TCN expects [batch_size, num_channels, sequence_length]
        # Either force num_filters and num_channels to be the same number
        # or make another layer, that brings to correct shape
        # Dont know if this causes any preformance drop?
        self.bottleneck = nn.Conv1d(num_filters, num_channels, kernel_size=1)

        self.tcn = TCN(num_channels, num_blocks, kernel_size, skip_channels, residual_channels)
        self.decoder = Decoder(num_filters, kernel_size, stride)
        self.mask_conv = nn.Conv1d(skip_channels, num_speakers * num_filters, kernel_size=1)
        self.num_speakers = num_speakers
        self.num_filters = num_filters

    def forward(self, x):
        enc_out = self.encoder(x)  
        bottleneck_out = self.bottleneck(enc_out)  
        tcn_out = self.tcn(bottleneck_out)
        masks = torch.sigmoid(self.mask_conv(tcn_out))
        # Reshape from [batch_size, num_sources * num_filters, sequence_length]
        # To [batch_size, num_sources, num_filters, sequence_length]
        # As descibed in section C of the paper
        masks = masks.view(x.size(0), self.num_speakers, self.num_filters, -1)
        masked_speakers = enc_out.unsqueeze(1) * masks  # Section C (4)
        # we then decode each speaker
        decoded_speakers = [self.decoder(masked_speakers[:, i]) for i in range(self.num_speakers)] # Section C (5)
        return torch.cat(decoded_speakers, dim=1)


# Initialize model parameters
num_speakers = 2
num_filters = 256
kernel_size = 7 # Avoid even kernel_sizes. Odd dilation (1) and even kernel_size gives warning, but still seems to work
stride = 10
num_channels = 512
num_blocks = 8
skip_channels = 128
residual_channels = 512  

model = ConvTasNet(
    num_speakers=num_speakers,
    num_filters=num_filters,
    kernel_size=kernel_size,
    stride=stride,
    num_channels=num_channels,         
    num_blocks=num_blocks,
    skip_channels=skip_channels,
    residual_channels=residual_channels)

# dummy input tensor
batch_size = 4
input_length = 32000
x = torch.randn(batch_size, 1, input_length)

output = model(x)
print(f"Output shape: {output.shape}")  # Should be [batch_size, num_speakers, output_length]
