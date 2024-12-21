import os
import torch
from Dataloader.Dataloader import EarsDataset, ConvTasNetDataLoader

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure the environment variable $BLACKHOLE is set
blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The environment variable $BLACKHOLE is not set.")

# Define paths
data_dir = os.path.join(blackhole_path, "EARS-WHAM")
print(f"Dataset directory: {data_dir}")

# Test dataset initialization
try:
    dataset_TRN = EarsDataset(data_dir=data_dir, subset='train', normalize=False)
    print(f"Training dataset size: {len(dataset_TRN)}")
except Exception as e:
    print(f"Failed to initialize training dataset: {e}")

try:
    train_loader = ConvTasNetDataLoader(dataset_TRN, batch_size=1, shuffle=True)
    print(f"Train loader created successfully with batch size 1.")
except Exception as e:
    print(f"Failed to create train loader: {e}")

# Check if teacher model loads correctly
try:
    from conv_tasnet_causal import ConvTasNet
    teacher = torch.compile(torch.load("teacher_latest.pth"))
    teacher = teacher.to(device)
    print("Teacher model loaded and moved to device successfully.")
except Exception as e:
    print(f"Failed to load teacher model: {e}")

