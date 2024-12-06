import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch

from load_models import *

def plot_spectrogram(audio, sr, title, ax):
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)

def visualize_results(results, sr=16000):
    for idx, (predictions, inputs, outputs) in enumerate(results):
        # Convert tensors to numpy
        inputs_np = inputs.squeeze().numpy()
        outputs_np = outputs.squeeze().numpy()

        fig, axes = plt.subplots(len(predictions) + 2, 1, figsize=(12, len(predictions) * 4))

        # Plot output spectrogram
        plot_spectrogram(outputs_np, sr, "Ground Truth Clean Speech", axes[1])

        # Plot predictions for each model
        for i, prediction in enumerate(predictions):
            prediction_np = prediction.squeeze().detach().numpy()
            # make more precise describtion of models later
            plot_spectrogram(prediction_np, sr, f"Model {i + 1} Prediction", axes[i + 2])
        
        # Plot input spectrogram
        plot_spectrogram(inputs_np, sr, "Noisy Speech", axes[0])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Generate results with mock data
    results = get_model_predictions_and_data(mock=True, save_memory=True, datapoints=1, deterministic=True)
    
    # Visualize spectrograms
    visualize_results(results, sr=16000)
