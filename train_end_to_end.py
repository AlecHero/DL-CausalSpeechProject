import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from ConvTasNet import ConvTasNet
from tqdm import tqdm

## CONSTANTS
num_speakers = 2
num_filters = 256
kernel_size = 7 # Avoid even kernel_sizes. Odd dilation (1) and even kernel_size gives warning, but still seems to work
stride = 10
num_channels = 512
num_blocks = 8
skip_channels = 128
residual_channels = 512  
batch_size = 4
sr = 16000

inputs, labels = torch.randn(batch_size, 1, sr), torch.randint(0, 2, (batch_size, 1, sr - 3))
model = ConvTasNet(num_speakers=num_speakers, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                   num_channels=num_channels, num_blocks=num_blocks, skip_channels=skip_channels, residual_channels=residual_channels)
teacher = ConvTasNet(num_speakers=num_speakers, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                     num_channels=num_channels, num_blocks=num_blocks, skip_channels=skip_channels, residual_channels=residual_channels)
alpha = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in tqdm(range(20)):
    optimizer.zero_grad()

    with torch.no_grad():
        soft_target = nn.functional.softmax(teacher(inputs), dim=1)

    hard_target = torch.zeros_like(soft_target)
    hard_target.scatter_(1, labels, 1)

    target = alpha * soft_target + (1 - alpha) * hard_target

    outputs = model(inputs)
    logprobs = nn.functional.log_softmax(outputs, dim=1)

    loss = nn.functional.kl_div(logprobs, target, reduction='batchmean')

    print(f"Iteration {i}: Loss = {loss.item()}")
    loss.backward()
    optimizer.step()

