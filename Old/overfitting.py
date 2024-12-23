import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from conv_tasnet_causal import ConvTasNet
from tqdm import tqdm
from eval import Loss
from neptuneLogger import NeptuneLogger
from wav_generator import save_to_wav
from Dataloader.Dataloader import EarsDataset
from asteroid_loss import PairwiseNegSDR
import os
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from Old.criterion import cal_loss

## CONSTANTS
num_sources = 2
enc_kernel_size = 16
enc_num_feats = 512
msk_kernel_size = 3
msk_num_feats = 128
msk_num_hidden_feats = 512
msk_num_layers = 8
msk_num_stacks = 3
msk_activate = 'sigmoid'
batch_size = 1
sr = 2000

blackhole_path = os.getenv('BLACKHOLE')

if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

overfit_idx = 1
dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False, max_samples=10)
print("Dataset imported")
inputs, labels = dataset_TRN.__getitem__(overfit_idx)

print("SNR loss inputs vs labels", Loss().snr(inputs, labels))
print("SNR loss labels vs labels", Loss().snr(labels, labels))

inputs = inputs.unsqueeze(0)[:, :, :16000]
labels = labels.unsqueeze(0)[:, :, :16000]
# save_to_wav(inputs[:, :, :].detach().numpy(), output_filename="test_inputs.wav")
# save_to_wav(labels[:, :, :].detach().numpy(), output_filename="test_labels.wav")
# print(inputs.unsqueeze(0).shape)
# print(labels.unsqueeze(0).shape)
# raise ValueError()
print("Getitem worked")
# inputs, labels = torch.randn(batch_size, 1, sr), torch.randint(0, 2, (batch_size, 2, sr))

# inputs = ...
# labels = ...

# model = torchaudio.models.conv_tasnet.ConvTasNet(
#         num_sources=2,
#         enc_kernel_size=3,  # Reduced from 20 to avoid size mismatch
#         enc_num_feats=512,
#         msk_kernel_size=3,  # Reduced from 20 to match encoder kernel size
#         msk_num_feats=512,
#         msk_num_hidden_feats=512,
#         msk_num_layers=2,
#         msk_num_stacks=8,
#         msk_activate="sigmoid"
# )

teacher = torchaudio.models.conv_tasnet.ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,  # Reduced from 20 to avoid size mismatch
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,  # Reduced from 20 to match encoder kernel size
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate
)

print("Models created")

alpha = 0.5
lr = 1e-3
epochs = 2000

logger = NeptuneLogger(test=True)
# optimizer = torch.optim.Adam(model.parameters())
teacher_optim = torch.optim.Adam(teacher.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(teacher_optim, mode='min', factor = 0.5, patience = 100)

print("Logger started")

logger.log_metadata({
    "lr": lr,
    "optimizer": "adam",
    "scheduler": "Reduce on Plateau",
    "epoch": epochs,
    "batch_size": batch_size,
    "desc": "loss with noise "
})

loss_func = Loss()
# loss_func = PairwiseNegSDR()

# scaler = torch.amp.GradScaler()

# Missing causal teacher and non-causal student
# Loss is fucked

for i in tqdm(range(epochs), desc="Training..."):
    teacher_optim.zero_grad()
    teacher_output = teacher(inputs)
    clean_sound_teacher_output = teacher_output[:, 0:1, :]
    # noise = inputs - labels
    # labels_with_noise = torch.cat((labels, noise), dim=1)
    
    # Ensure loss is a scalar by taking the mean
    
    # loss = cal_loss(labels, clean_sound_teacher_output, torch.tensor([16000]))
    loss = -loss_func.sisnr(labels, clean_sound_teacher_output)
    logger.log_metric("SNR", loss_func.snr(clean_sound_teacher_output, labels))
    
    loss.backward()
    teacher_optim.step()
    scheduler.step(loss.item())
    
    logger.log_metric("train_loss", loss.item())
    for param_group in teacher_optim.param_groups:
        current_lr = param_group['lr']
    logger.log_metric("lr", current_lr)

    if i % 20 == 0:
        # save_to_wav(teacher_output[0:1, 0:1, :].detach().numpy(), output_filename="train_sound_1.wav")
        # save_to_wav(teacher_output[0:1, 1:2, :].detach().numpy(), output_filename="train_sound_2.wav")

        # logger.log_train_soundfile("train_sound_1.wav", speaker=1, idx=i)
        # logger.log_train_soundfile("train_sound_2.wav", speaker=2, idx=i)

        desired_rms = 0.03  # can change this, not a constant
        rms = torch.sqrt(torch.mean(teacher_output**2))
        teacher_output = teacher_output * (desired_rms / (rms + 1e-9))
        teacher_output = torch.clamp(teacher_output, -0.9, 0.9)

        save_to_wav(teacher_output[0:1, 0:1, :].detach().numpy(), output_filename="teacher_train_clean.wav")
        save_to_wav(teacher_output[0:1, 1:2, :].detach().numpy(), output_filename="teacher_train_noicy.wav")
        save_to_wav(labels[0:1, 0:1, :].detach().numpy(), output_filename="train_true_label.wav")

        logger.log_custom_soundfile("teacher_train_clean.wav", f"train/teacher_clean_index{i}.wav")
        logger.log_custom_soundfile("train_true_label.wav", "train/true_label.wav")

# save models to neptune please
# torch.save(model.state_dict(), PATH)

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, weights_only=True))
# model.eval()