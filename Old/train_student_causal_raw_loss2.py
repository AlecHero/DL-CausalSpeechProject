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
import os

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

blackhole_path = os.getenv('BLACKHOLE')

if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

overfit_idx = 1
dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False, max_samples=1)
train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=batch_size, shuffle=True)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False, max_samples=1)
val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=batch_size, shuffle=True)

# All of this should be preloaded in, not lazy. Laziness is time consuming here.
# If laze then should save outputs! in memory??
# ALso check if .compile is actually faster...

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

logger = NeptuneLogger(test=True)
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
    losses1 = []
    losses2 = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs[:, :, :]
            labels = labels[:, :, :]
            inputs, labels = inputs.to(device), labels.to(device)

            clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
            loss1 = -loss_func.sisnr(clean_sound_teacher_output, labels)
            losses1.append(loss1)
            loss2 = -loss_func.compute_loss(clean_sound_teacher_output, labels)
            losses2.append(loss2)
    return sum(losses1)/len(losses1), sum(losses2)/len(losses2) 

@torch.compile
def forward_and_back(inputs, labels):
    teacher.train()
    teacher_optimizer.zero_grad() 
    clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
    loss = -loss_func.sisnr(clean_sound_teacher_output, labels)
    loss.backward()
    teacher_optimizer.step()
    return loss, clean_sound_teacher_output

for i in tqdm(range(epochs)):
    start_time = time.time()
    losses = []
    for batch in tqdm(train_loader, desc=f"Epoch {i}"):
        j += 1
        batched_inputs, batched_labels = batch
        inputs = batched_inputs[:, :, :16000]
        labels = batched_labels[:, :, :16000]
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
    #logger.log_custom_soundfile("teacher_train_clean.wav", f"train/teacher_clean_index{i}.wav")
    #logger.log_custom_soundfile("train_true_label.wav", "train/true_label.wav")

    #val_loss, val_loss2 = eval()
    #scheduler.step(val_loss)
    #logger.log_metric("val_loss_sisnr", val_loss)
    #logger.log_metric("val_loss_old_loss", val_loss2)
    #torch.save(teacher.state_dict(), "teacher.pth")
    #logger.log_model("teacher.pth", "artifacts/teacher_latest.pth")