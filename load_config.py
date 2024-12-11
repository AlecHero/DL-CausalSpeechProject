from dataclasses import dataclass
import yaml
import os
from pathlib import Path

@dataclass
class TrainingInit:
    teacher_path: str
    student_path: str
    train_teacher: bool
    train_student: bool
    train_student_without_teacher: bool

@dataclass 
class TrainingParams:
    lr: float
    dropout: float
    weight_decay: float
    epochs: int
    epoch_to_turn_off_intermediate: int
    batch_size: int
    teacher_predictions_factor: float

@dataclass
class Debug:
    test_run: bool
    save_memory: bool
    overfit_run: bool

@dataclass
class Config:
    run_name: str
    debug: Debug
    training_init: TrainingInit
    training_params: TrainingParams

def logic_check(config_dict: dict):
    assert not all([config_dict['training_init']['train_student'], config_dict['training_init']['train_student_without_teacher']]), "You cannot train both student and student without teacher at the same time"

def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    logic_check(config_dict)
    
    debug = Debug(**config_dict['debug'])
    training_init = TrainingInit(**config_dict['training_init'])
    training_params = TrainingParams(**config_dict['training_params'])
    
    return Config(
        run_name=config_dict['run_name'],
        debug=debug,
        training_init=training_init,
        training_params=training_params
    )
