import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
# from ConvTasNet import ConvTasNet
from tqdm import tqdm
from eval import Loss, Accuracy
from neptuneLogger import NeptuneLogger
from wav_generator import save_to_wav
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import pickle
import time
from conv_tasnet_causal import ConvTasNet

print("Torch is available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# sr = 2000
j = 0

# All of this should be preloaded in, not lazy. Laziness is time consuming here.
# If laze then should save outputs! in memory??
# ALso check if .compile is actually faster...
dataset_TRN = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'train', normalize = False)
print(len(dataset_TRN))
train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=batch_size, shuffle=True)
dataset_VAL = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'valid', normalize = False)
print(len(dataset_VAL))
val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=batch_size, shuffle=True)

print("Dataloader imported")

teacher = torch.compile(ConvTasNet(
    num_sources=num_sources,
    enc_kernel_size=enc_kernel_size,
    enc_num_feats=enc_num_feats,
    msk_kernel_size=msk_kernel_size,
    msk_num_feats=msk_num_feats,
    msk_num_hidden_feats=msk_num_hidden_feats,
    msk_num_layers=msk_num_layers,
    msk_num_stacks=msk_num_stacks,
    msk_activate=msk_activate,
))

teacher = teacher.to(device)

print("Should say cuda:0: ", next(teacher.parameters()).device)
print(len(train_loader))

print("Models created and compiles")

## !! CHANGE THESE !!
alpha = 0.5
lr = 1e-3
epochs = 200

logger = NeptuneLogger()
teacher_optimizer = torch.optim.Adam(teacher.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(teacher_optimizer, mode='min', factor = 0.5, patience = 3)

print("Logger started")

logger.log_metadata({
    "lr": lr,
    "optimizer": "adam",
    "scheduler": "Reduce on Plateau",
    "epoch": epochs,
    "batch_size": batch_size,
    "desc": "Student training directly from labels, no teacher"
})

loss_func = Loss()

@torch.compile
def eval():
    teacher.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs[:, :, :]
            labels = labels[:, :, :]
            inputs, labels = inputs.to(device), labels.to(device)

            clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
            loss = -loss_func.sisnr(clean_sound_teacher_output, labels)
            losses.append(loss)
    return sum(losses)/len(losses)

@torch.compile
def forward_and_back(inputs, labels):
    teacher.train()
    teacher_optimizer.zero_grad() 
    clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
    loss = -loss_func.compute_loss(clean_sound_teacher_output, labels)
    loss.backward()
    teacher_optimizer.step()
    return loss, clean_sound_teacher_output 

for i in range(epochs):
    start_time = time.time()
    losses = []
    for batch in train_loader:
        j += 1
        batched_inputs, batched_labels = batch
        inputs = batched_inputs[:, :, :]
        labels = batched_labels[:, :, :]
        inputs, labels = inputs.to(device), labels.to(device)

        loss, clean_sound_teacher_output = forward_and_back(inputs, labels)
        losses.append(loss.item())
        if j % 1000 == 0:
            print(f"at {j} out of {len(train_loader)}")
            avg_loss = sum(losses)/len(losses)
            losses = []
            logger.log_metric("train_loss", avg_loss, step=j)
            for param_group in teacher_optimizer.param_groups:
                current_lr = param_group['lr']
            logger.log_metric("lr_teacher", current_lr, step=j)
    print(f"done with epoch {i}")
    logger.log_metric("epoch_time", time.time() - start_time) 
    save_to_wav(clean_sound_teacher_output[0:1, 0:1, :].cpu().detach().numpy(), output_filename="teacher_train_clean.wav")
    save_to_wav(labels[0:1, 0:1, :].cpu().detach().numpy(), output_filename="train_true_label.wav")
    logger.log_custom_soundfile("teacher_train_clean.wav", f"train/teacher_clean_index{i}.wav")
    logger.log_custom_soundfile("train_true_label.wav", "train/true_label.wav")

    val_loss = eval()
    scheduler.step(val_loss)
    logger.log_metric("val_loss", val_loss)
    torch.save(teacher.state_dict(), "teacher.pth")
    logger.log_model("teacher.pth", "artifacts/teacher_latest.pth")