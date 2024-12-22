import matplotlib.pyplot as plt
import pandas as pd
import glob
from typing import List, Tuple
from pathlib import Path
import numpy as np


def load_results(directory: str = 'results') -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all CSV result files from the specified directory.
    
    Args:
        directory: Directory containing the CSV files
        
    Returns:
        List of tuples containing (filename, dataframe)
    """
    csv_files = glob.glob(f'{directory}/*.csv')
    results = []
    
    max_epochs = 0
    for file in csv_files:
        df = pd.read_csv(file, header=None, names=['epoch', 'timestamp', 'value'])
        max_epochs = max(max_epochs, len(df))
    
    for file in csv_files:
        filename = Path(file).name
        df = pd.read_csv(file, header=None, names=['epoch', 'timestamp', 'value'])
        
        # Extend shorter sequences to match max length
        if len(df) < max_epochs:
            # Create new indices spaced by 1.6
            new_indices = np.linspace(0, max_epochs-1, len(df))
            # Interpolate values at new indices
            df['epoch'] = new_indices
            
        results.append((filename, df))
    
    return results


def plot_training_results(
    results: List[Tuple[str, pd.DataFrame]],
    save_path: str = 'training_results.png',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Create and save a plot comparing training results.
    
    Args:
        results: List of (filename, dataframe) tuples from load_results()
        save_path: Path where to save the plot
        figsize: Figure dimensions (width, height)
    """
    plt.figure(figsize=figsize)
    
    for filename, df in results:
        # Convert epoch to hours (assuming 24 hours total)
        hours = df['epoch'] * (24 / max(df['epoch']))
        plt.plot(hours, df['value'], marker='o', markersize=4, label=filename.replace(".csv", ""))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Hours')
    plt.ylabel('SI-SNR validation loss')
    plt.title('SI-SNR validation loss over 24h training on 1 A100 GPU.')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    results = load_results()
    plot_training_results(results)
