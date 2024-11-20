import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from ConvTasNet import ConvTasNet
from tqdm import tqdm
from eval import Loss
from neptuneLogger import NeptuneLogger

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
sr = 4000

inputs, labels = torch.randn(batch_size, 1, sr), torch.randint(0, 2, (batch_size, 2, sr))
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
teacher = torchaudio.models.conv_tasnet.ConvTasNet(
        num_sources=2,
        enc_kernel_size=3,  # Reduced from 20 to avoid size mismatch
        enc_num_feats=256,
        msk_kernel_size=3,  # Reduced from 20 to match encoder kernel size
        msk_num_feats=256,
        msk_num_hidden_feats=256,
        msk_num_layers=2,
        msk_num_stacks=4,
        msk_activate="sigmoid"
)
alpha = 0.5

logger = NeptuneLogger()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 10)

loss_func = Loss()

scaler = torch.amp.GradScaler()

for i in tqdm(range(100)):
    optimizer.zero_grad()
    with torch.amp.autocast():
        soft_target = nn.functional.softmax(teacher(inputs), dim=1)
        target = alpha * soft_target + (1 - alpha) * labels
        outputs = model(inputs)
        loss = loss_func.compute_loss(outputs, target)
    
    print(f"Iteration {i}: Loss = {loss.item()}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step(loss.item())

