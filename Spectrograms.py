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
    models = ['Student_only_labels_dropout', 'Student_only_labels', 'Student_only_teacher', 'Student_partly_teacher', 'Student_only_teacher_e2e']
    for idx, (predictions, inputs, outputs) in enumerate(results):
        inputs_np = inputs.squeeze().numpy()
        outputs_np = outputs.squeeze().numpy()

        fig, axes = plt.subplots(len(predictions) + 2, 1, figsize=(12, len(predictions) * 4))

        # Plot noisy spectrogram
        plot_spectrogram(inputs_np, sr, "Noisy Speech", axes[0])

        # Plot clean spectrogram
        plot_spectrogram(outputs_np, sr, "Clean Speech", axes[1])

        # Plot predictions for each model
        for i, prediction in enumerate(predictions):
            prediction_np = prediction[0].squeeze().detach().numpy()
            plot_spectrogram(prediction_np, sr, f"{models[i]} Prediction", axes[i + 2])


        plt.tight_layout()

        # Save plot(s) in Plots folder
        plot_path = os.path.join('Plots', f"results_from_prediction{idx + 1}.png")
        plt.savefig(plot_path)

        plt.show()

if __name__ == "__main__":

    results = get_model_predictions_and_data(mock=False, save_memory=True, datapoints=10, deterministic=True)

    visualize_results(results, sr=16000)

