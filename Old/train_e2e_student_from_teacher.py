import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from conv_tasnet_causal import ConvTasNet
from tqdm import tqdm
from eval import Loss, Accuracy
from neptuneLogger import NeptuneLogger
from wav_generator import save_to_wav
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import pickle
import os
import time

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
sr = 2000
j = 0

blackhole_path = os.getenv('BLACKHOLE')

if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

overfit_idx = 1
dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=batch_size, shuffle=True)
dataset_VAL = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'valid', normalize = False)
val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=batch_size, shuffle=True)

print("Dataloader imported")

student = torch.compile(ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,  # Reduced from 20 to avoid size mismatch
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,  # Reduced from 20 to match encoder kernel size
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate,
        causal = True,
        save_intermediate_values = False
))

teacher = torch.compile(ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,  # Reduced from 20 to avoid size mismatch
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,  # Reduced from 20 to match encoder kernel size
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate,
        causal = True,
        save_intermediate_values = False
))

teacher = teacher.to(device)
student = student.to(device)

print("Models created and loaded")

alpha = 0.5
lr = 1e-3
epochs = 200

logger = NeptuneLogger()
student_optimizer = torch.optim.Adam(student.parameters())
teacher_optimizer = torch.optim.Adam(teacher.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(student_optimizer, mode='min', factor = 0.5, patience = 3)

print("Logger started")

logger.log_metadata({
    "lr": lr,
    "optimizer": "adam",
    "scheduler": "Reduce on Plateau",
    "epoch": epochs,
    "batch_size": batch_size,
    "desc": "Train student from only teacher End 2 End"
})

loss_func = Loss()

@torch.compile
def eval_student():
    teacher.eval()
    student.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs[:, :, :]
            labels = labels[:, :, :]
            inputs, labels = inputs.to(device), labels.to(device)

            clean_sound_student_output = student(inputs)[:, 0:1, :]
            loss = -loss_func.sisnr(clean_sound_student_output, labels)
            losses.append(loss)
    return sum(losses)/len(losses)

@torch.compile
def eval_teacher():
    teacher.eval()
    student.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs[:, :, :]
            labels = labels[:, :, :]
            inputs, labels = inputs.to(device), labels.to(device)

            clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
            loss = -loss_func.compute_loss(clean_sound_teacher_output, labels)
            losses.append(loss)
    return sum(losses)/len(losses)

@torch.compile
def forward_and_back(inputs, labels):
    teacher.train()
    teacher_optimizer.zero_grad()
    teacher_out = teacher(inputs)[:, 0:1, :]
    teacher_loss = -loss_func.sisnr(teacher_out, labels)
    teacher_loss.backward()
    teacher_optimizer.step()

    teacher.eval()
    student.train()
    student_optimizer.zero_grad()
    with torch.no_grad():
        teacher_out = teacher(inputs)[:, 0:1, :]
    student_output = student(inputs)[:, 0:1, :]
    loss = -loss_func.sisnr(student_output, teacher_out)
    loss.backward()
    student_optimizer.step()
    return loss, student_output, teacher_loss, teacher_out

for i in range(epochs):
    start_time = time.time()
    losses = []
    teacher_losses = []
    for batch in train_loader:
        j += 1
        batched_inputs, batched_labels = batch
        inputs = batched_inputs[:, :, :]
        labels = batched_labels[:, :, :]
        inputs, labels = inputs.to(device), labels.to(device)

        loss, clean_sound_student_output, teacher_loss, clean_sound_teacher_output = forward_and_back(inputs, labels)
        losses.append(loss.item())
        teacher_losses.append(teacher_loss.item())
        if j % 1000 == 0:
            print(f"at {j} out of {len(train_loader)}")
            avg_loss = sum(losses)/len(losses)
            avg_teacher_loss = sum(teacher_losses)/len(teacher_losses)
            losses = []
            teacher_losses = []
            logger.log_metric("train_loss", avg_loss, step=j)
            logger.log_metric("teacher_train_loss", avg_teacher_loss, step=j)
            for param_group in student_optimizer.param_groups:
                current_lr = param_group['lr']
            logger.log_metric("lr_student", current_lr, step=j)
    print(f"done with epoch {i}")
    logger.log_metric("epoch_time", time.time() - start_time) 
    save_to_wav(clean_sound_student_output[0:1, 0:1, :].cpu().detach().numpy(), output_filename="student_train_clean.wav")
    save_to_wav(labels[0:1, 0:1, :].cpu().detach().numpy(), output_filename="train_true_label.wav")
    save_to_wav(clean_sound_teacher_output[0:1, 0:1, :].cpu().detach().numpy(), output_filename="teacher_train_clean.wav")
    logger.log_custom_soundfile("student_train_clean.wav", f"train/student_clean_index{i}.wav")
    logger.log_custom_soundfile("train_true_label.wav", "train/true_label.wav")
    logger.log_custom_soundfile("teacher_train_clean.wav", f"train/teacher_clean_index{i}.wav")

    val_loss = eval_student()
    val_teacher_loss = eval_teacher()
    scheduler.step(val_loss)
    logger.log_metric("val_loss", val_loss)
    logger.log_metric("val_teacher_loss", val_teacher_loss)
    torch.save(student.state_dict(), "student.pth")
    logger.log_model("student.pth", "artifacts/student_latest.pth")
    torch.save(teacher.state_dict(), "teacher.pth")
    logger.log_model("teacher.pth", "artifacts/teacher_latest.pth")

