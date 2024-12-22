import argparse
from load_config import load_config, Config
from load_models import load_models
import torch
from conv_tasnet_causal import ConvTasNet
from tqdm import tqdm
from neptuneLogger import NeptuneLogger
from wav_generator import save_to_wav
from Dataloader.Dataloader import EarsDataset,ConvTasNetDataLoader
import pickle
import time
import os
from typing import Tuple, Union
from torchmetrics.functional import signal_noise_ratio, scale_invariant_signal_noise_ratio
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
SAVE_MEMORY = False


def get_dataloaders(config: Config, data_path: str) -> Tuple[ConvTasNetDataLoader, ConvTasNetDataLoader]:
    max_samples = 1 if config.debug.overfit_run else None
    dataset_TRN = EarsDataset(data_dir=data_path, subset = 'train', normalize = False, max_samples = max_samples)
    train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=config.training_params.batch_size, shuffle=True)
    dataset_VAL = EarsDataset(data_dir=data_path, subset = 'valid', normalize = False, max_samples = max_samples)
    val_loader = ConvTasNetDataLoader(dataset_VAL, batch_size=config.training_params.batch_size, shuffle=True)
    return train_loader, val_loader

@torch.compile
def evaluate_model(model: ConvTasNet, val_loader: ConvTasNetDataLoader, loss_func: Union[signal_noise_ratio, scale_invariant_signal_noise_ratio]):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            if SAVE_MEMORY:
                inputs = inputs[:, :, :16000]
                labels = labels[:, :, :16000]
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            clean_sound_student_output = model(inputs)[:, 0:1, :]
            loss = -loss_func(clean_sound_student_output, labels)
            losses.append(loss)
    return sum(losses)/len(losses)

@torch.compile
def train_step_teacher(inputs: torch.Tensor, labels: torch.Tensor, teacher: ConvTasNet, optimizer: torch.optim.Optimizer, loss_func: Union[signal_noise_ratio, scale_invariant_signal_noise_ratio]):
    # assert inputs.device == labels.device == next(teacher.parameters()).device
    teacher.train()
    optimizer.zero_grad() 
    clean_sound_teacher_output = teacher(inputs)[:, 0:1, :]
    loss = -loss_func(clean_sound_teacher_output, labels)
    loss.backward()
    optimizer.step()
    return loss 

@torch.compile
def train_step_student(inputs: torch.Tensor, labels: torch.Tensor, student: ConvTasNet, teacher: ConvTasNet, optimizer: torch.optim.Optimizer, loss_func: Union[signal_noise_ratio, scale_invariant_signal_noise_ratio], teacher_predictions_factor: float):
    teacher.eval()
    student.train()
    optimizer.zero_grad()
    with torch.no_grad():
        teacher_out = teacher(inputs)[:, 0:1, :]
    student_output = student(inputs)[:, 0:1, :]
    target = teacher_predictions_factor * teacher_out + (1 - teacher_predictions_factor) * labels
    loss = -loss_func(student_output, target)
    loss.backward()
    optimizer.step()
    return loss

@torch.compile
def train_step_student_without_teacher(inputs: torch.Tensor, labels: torch.Tensor, student: ConvTasNet, optimizer: torch.optim.Optimizer, loss_func: Union[signal_noise_ratio, scale_invariant_signal_noise_ratio]):
    student.train()
    optimizer.zero_grad()
    student_output = student(inputs)[:, 0:1, :]
    loss = -loss_func(student_output.contiguous(), labels.contiguous())
    loss.backward()
    optimizer.step()
    return loss.detach()

def log_train_losses(logger: NeptuneLogger, train_losses: dict):
    for key, value in train_losses.items():
        logger.log_metric(f"train_loss/{key}", value)

def log_eval_metrics(logger: NeptuneLogger, teacher: ConvTasNet, student: ConvTasNet, val_loader: ConvTasNetDataLoader, loss_func: Union[signal_noise_ratio, scale_invariant_signal_noise_ratio]):
    eval_losses = {}
    eval_losses['teacher'] = evaluate_model(teacher, val_loader, loss_func)
    eval_losses['student'] = evaluate_model(student, val_loader, loss_func)
    for key, value in eval_losses.items():
        logger.log_metric(f"eval_loss/{key}", value)
    return eval_losses

def log_example_wavs(logger: NeptuneLogger, inputs: torch.Tensor, labels: torch.Tensor, teacher: ConvTasNet, student: ConvTasNet, epoch: int):
    if not os.path.exists("tmp"): os.mkdir("tmp")
    with torch.no_grad():
        teacher_out = teacher(inputs)[:, 0:1, :]
        student_out = student(inputs)[:, 0:1, :]
    save_to_wav(teacher_out.cpu().detach().numpy(), output_filename=f"tmp/teacher_out.wav")
    save_to_wav(student_out.cpu().detach().numpy(), output_filename=f"tmp/student_out.wav")
    save_to_wav(labels.cpu().detach().numpy(), output_filename=f"tmp/labels.wav")
    logger.log_custom_soundfile(f"tmp/teacher_out.wav", f"eval/teacher_out{epoch}.wav")
    logger.log_custom_soundfile(f"tmp/student_out.wav", f"eval/student_out{epoch}.wav")
    logger.log_custom_soundfile(f"tmp/labels.wav", f"eval/labels{epoch}.wav")

def step_scheduler(logger: NeptuneLogger, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, eval_losses: dict, teacher_optimizer: torch.optim.Optimizer, student_optimizer: torch.optim.Optimizer):
    scheduler.step(eval_losses['teacher'])
    for param_group in teacher_optimizer.param_groups:
        current_lr = param_group['lr']
    for student_param_group in student_optimizer.param_groups:
        student_param_group['lr'] = current_lr
    logger.log_metric("learning_rate", current_lr)

def save_models(logger: NeptuneLogger, teacher: ConvTasNet, student: ConvTasNet, raw_models_copy: list):
    if not os.path.exists("tmp"): os.mkdir("tmp")
    teacher_copy, _ = raw_models_copy[0]
    student_copy, _ = raw_models_copy[1]
    teacher_copy.to('cpu')
    student_copy.to('cpu')
    teacher_copy.load_state_dict(teacher.state_dict())
    student_copy.load_state_dict(student.state_dict())
    torch.save(teacher_copy.state_dict(), f"tmp/teacher.pth")
    torch.save(student_copy.state_dict(), f"tmp/student.pth")
    logger.log_model(f"tmp/teacher.pth", f"artifacts/teacher.pth") # Notice that this is not normalized, so might sound bad while being okay when normalized.
    logger.log_model(f"tmp/student.pth", f"artifacts/student.pth")

def train(config: Config):
    save_int_vals = True if config.training_params.epoch_to_turn_off_intermediate > 0 else False
    models = load_models([config.training_init.teacher_path, config.training_init.student_path], DEVICE, causal = [False, True], 
                         save_intermediate_values = [save_int_vals, save_int_vals], dropout = [config.training_params.dropout, config.training_params.dropout])
    logger = NeptuneLogger(project_name = config.neptune_project, api_token = config.neptune_api, test = config.debug.test_run)

    logger.log_metadata({
        "lr": config.training_params.lr,
        "optimizer": "adam",
        "scheduler": "Reduce on Plateau",
        "epoch": config.training_params.epochs,
        "batch_size": config.training_params.batch_size,
        "desc": config.run_name
    })

    teacher, _ = models[0]
    student, _ = models[1]
    
    teacher = teacher.to(DEVICE)
    student = student.to(DEVICE)
    
    train_loader, val_loader = get_dataloaders(config, config.data_path)
    loss_func = scale_invariant_signal_noise_ratio

    student_optimizer = torch.optim.Adam(student.parameters(), weight_decay=config.training_params.weight_decay)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), weight_decay=config.training_params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(teacher_optimizer, mode='min', factor = 0.5, patience = 3)

    raw_models_copy = load_models([config.training_init.teacher_path, config.training_init.student_path], DEVICE, causal = [False, True], save_intermediate_values = [save_int_vals, save_int_vals])
    save_models(logger, teacher, student, raw_models_copy)

    for i in tqdm(range(config.training_params.epochs)):
        start_time = time.time()
        for inputs, labels in train_loader:
            if SAVE_MEMORY:
                inputs = inputs[:, :, :16000]
                labels = labels[:, :, :16000]
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            train_losses = {}
            if config.training_init.train_teacher:
                train_losses['teacher'] = train_step_teacher(inputs, labels, teacher, teacher_optimizer, loss_func)
            if config.training_init.train_student:
                train_losses['student'] = train_step_student(inputs, labels, student, teacher, student_optimizer, loss_func, config.training_params.teacher_predictions_factor)
            if config.training_init.train_student_without_teacher:
                train_losses['student_without_teacher'] = train_step_student_without_teacher(inputs, labels, student, student_optimizer, loss_func)
            
            log_train_losses(logger, train_losses)
        eval_losses = log_eval_metrics(logger, teacher, student, val_loader, loss_func)
        logger.log_metric("time", time.time() - start_time)
        log_example_wavs(logger, inputs, labels, teacher, student, i)
        step_scheduler(logger, scheduler, eval_losses, teacher_optimizer, student_optimizer)
        save_models(logger, teacher, student, raw_models_copy)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    SAVE_MEMORY = config.debug.save_memory
    train(config)
