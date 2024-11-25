import math
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from ConvTasNet import ConvTasNet
from tqdm import tqdm
from eval import Loss, Accuracy
from neptuneLogger import NeptuneLogger
from wav_generator import save_to_wav
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import pickle

## CONSTANTS
num_sources=2
enc_kernel_size=3  # Reduced from 20 to avoid size mismatch
enc_num_feats=256
msk_kernel_size=3  # Reduced from 20 to match encoder kernel size
msk_num_feats=256
msk_num_hidden_feats=256
msk_num_layers=2
msk_num_stacks=4
msk_activate="sigmoid"
batch_size = 1
sr = 2000
_LOCAL = False

overfit_idx = 1
dataset_TRN = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'train', normalize = False, max_samples=100)
# Limit to 100 samples
train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=batch_size, shuffle=True)
dataset_VAL = EarsDataset(data_dir="/dtu/blackhole/0b/187019/EARS-WHAM", subset = 'valid', normalize = False, max_samples=100)
# Limit to 100 sample
val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=batch_size, shuffle=True)

if _LOCAL:
    sound_files = pickle.load("sound_file_lists.pkl", "rb")
    dataset_TRN.clean_files = sound_files["clean_trn_files"]
    dataset_VAL.clean_files = sound_files["clean_val_files"] 
    dataset_TRN.noisy_files = sound_files["noisy_trn_files"]
    dataset_VAL.noisy_files = sound_files["noisy_val_files"]


print("Dataloader imported")
# inputs, labels = next(train_iter)
# inputs = inputs.unsqueeze(0)[:, :, :16000]
# labels = labels.unsqueeze(0)[:, :, :16000]

# save_to_wav(inputs[:, :, :].detach().numpy(), output_filename="test_inputs.wav")
# save_to_wav(labels[:, :, :].detach().numpy(), output_filename="test_labels.wav")
# print(inputs.unsqueeze(0).shape)
# print(labels.unsqueeze(0).shape)
# raise ValueError()
# print("Getitem worked")
# inputs, labels = torch.randn(batch_size, 1, sr), torch.randint(0, 2, (batch_size, 2, sr))

# inputs = ...
# labels = ...

# student = torchaudio.models.conv_tasnet.ConvTasNet(
#         num_sources=num_sources,
#         enc_kernel_size=enc_kernel_size,  # Reduced from 20 to avoid size mismatch
#         enc_num_feats=enc_num_feats,
#         msk_kernel_size=msk_kernel_size,  # Reduced from 20 to match encoder kernel size
#         msk_num_feats=msk_num_feats,
#         msk_num_hidden_feats=msk_num_hidden_feats,
#         msk_num_layers=msk_num_layers,
#         msk_num_stacks=msk_num_stacks,
#         msk_activate=msk_activate
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
epochs = 200

logger = NeptuneLogger()
# student_optimizer = torch.optim.Adam(student.parameters())
teacher_optimizer = torch.optim.Adam(teacher.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(teacher_optimizer, mode='min', factor = 0.5, patience = 600)

print("Logger started")

logger.log_metadata({
    "lr": lr,
    "optimizer": "adam",
    "scheduler": "Reduce on Plateau",
    "epoch": epochs,
    "batch_size": batch_size,
    "desc": "Only clean soundfile loss calc, only teacher training, ReduceLROnPlateau with patience=50"
})

loss_func = Loss()

# Missing causal teacher and non-causal student

def eval(i: int):
    # student is compared to calc target. Is this correct?
    # student.eval()
    teacher.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating..."):
            inputs, labels = batch
            inputs = inputs[:, :, :16000]
            labels = labels[:, :, :16000]

            clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
            loss = loss_func.compute_loss(clean_sound_teacher_output, labels)
            # target = alpha * clean_sound_teacher_output + (1 - alpha) * labels
            # clean_sound_student_output = student(inputs)[:, 0:1, :]
            # loss = loss_func.compute_loss(clean_sound_student_output, target)
            losses.append(loss)
            accuracies.append(Accuracy.mse(clean_sound_teacher_output, labels))

        save_to_wav(clean_sound_teacher_output[0:1, 0:1, :].detach().numpy(), output_filename="teacher_val_clean.wav")
        # save_to_wav(clean_sound_student_output[0:1, 0:1, :].detach().numpy(), output_filename="student_val_clean.wav")
        save_to_wav(labels[0:1, 0:1, :].detach().numpy(), output_filename="val_true_label.wav")

        logger.log_custom_soundfile("teacher_val_clean.wav", f"val/teacher_clean_index{i}.wav")
        # logger.log_custom_soundfile("student_val_clean.wav", f"val/student_clean_index{i}.wav")
        logger.log_custom_soundfile("val_true_label.wav", "val/true_label.wav")

    # student.train()
    teacher.train()

    logger.log_metric("val_loss", sum(losses)/len(losses))
    logger.log_metric("val_acc_mse", sum(accuracies)/len(accuracies))
    return sum(losses)/len(losses)

for i in range(epochs):
    for batch in tqdm(train_loader, desc=f"Epoch {i} out of {epochs}"):
        batched_inputs, batched_labels = batch
        inputs = batched_inputs[:, :, :16000]
        labels = batched_labels[:, :, :16000]

        # student_optimizer.zero_grad()
        teacher_optimizer.zero_grad()

        clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
        loss = loss_func.compute_loss(clean_sound_teacher_output, labels)
        # target = alpha * clean_sound_teacher_output + (1 - alpha) * labels
        # clean_sound_student_output = student(inputs)[:, 0:1, :]
        # loss = loss_func.compute_loss(clean_sound_student_output, target)

        
        # print(f"Iteration {i}: Loss = {loss.item()}")
        # loss.backward(retain_graph=True)
        # student_optimizer.step()

        loss.backward()
        teacher_optimizer.step()
        scheduler.step(loss.item()) # This is maybe bad as we check every batch in each epoch, ideally should check 
        # each epoch instead

        # for student_param_group, teacher_param_group in zip(student_optimizer.param_groups, teacher_optimizer.param_groups):
        #     student_param_group['lr'] = teacher_param_group['lr']

        logger.log_metric("train_loss", loss.item())
        for param_group in teacher_optimizer.param_groups:
            current_lr = param_group['lr']
        logger.log_metric("lr_teacher", current_lr)
        # for param_group in student_optimizer.param_groups:
        #     current_lr = param_group['lr']
        # logger.log_metric("lr_student", current_lr)

    if i % 1 == 0:

        # save_to_wav(teacher_output[0:1, 0:1, :].detach().numpy(), output_filename="train_sound_1.wav")
        # save_to_wav(teacher_output[0:1, 1:2, :].detach().numpy(), output_filename="train_sound_2.wav")

        # logger.log_train_soundfile("train_sound_1.wav", speaker=1, idx=i)
        # logger.log_train_soundfile("train_sound_2.wav", speaker=2, idx=i)

        eval(i)

        save_to_wav(clean_sound_teacher_output[0:1, 0:1, :].detach().numpy(), output_filename="teacher_train_clean.wav")
        # save_to_wav(clean_sound_student_output[0:1, 0:1, :].detach().numpy(), output_filename="student_train_clean.wav")
        save_to_wav(labels[0:1, 0:1, :].detach().numpy(), output_filename="train_true_label.wav")

        logger.log_custom_soundfile("teacher_train_clean.wav", f"train/teacher_clean_index{i}.wav")
        # logger.log_custom_soundfile("student_train_clean.wav", f"train/student_clean_index{i}.wav")
        logger.log_custom_soundfile("train_true_label.wav", "train/true_label.wav")

        # torch.save(student.state_dict(), "student.pth")
        torch.save(teacher.state_dict(), "teacher.pth")

        # logger.log_model("student.pth", "artifacts/student_latest.pth")
        logger.log_model("teacher.pth", "artifacts/teacher_latest.pth")

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, weights_only=True))
# model.eval()