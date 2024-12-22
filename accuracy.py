from load_models import get_model_predictions_and_data
import torch
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
import numpy as np


if __name__ == "__main__":
    model_string = [
            "student_only_labels_dropout", 
            "student_only_labels", 
            "student_only_teacher",
            "student_partly_teacher",
            "student_only_teacher_e2e"
    ]
    datapoints = 100 # 593
    results = get_model_predictions_and_data(mock = False, save_memory = True, datapoints = datapoints, deterministic = True)
    
    model_results_snr = [[] for _ in range(len(model_string) + 2)]
    model_results_sdr = [[] for _ in range(len(model_string) + 2)]
    model_results_sisdr = [[] for _ in range(len(model_string) + 2)]
    
    for (predictions, inputs, outputs) in results:
        for i, (prediction, model_load_string) in enumerate(predictions):
            model_results_snr[i].append(SignalNoiseRatio()(prediction[0:1, :], outputs[0:1, :]))
            model_results_sdr[i].append(SignalDistortionRatio()(prediction[0:1, :], outputs[0:1, :]))
            model_results_sisdr[i].append(ScaleInvariantSignalDistortionRatio()(prediction[0:1, :], outputs[0:1, :]))
            
        model_results_snr[-2].append(SignalNoiseRatio()(inputs[0:1, :], outputs[0:1, :]))
        model_results_sdr[-2].append(SignalDistortionRatio()(inputs[0:1, :], outputs[0:1, :]))
        model_results_sisdr[-2].append(ScaleInvariantSignalDistortionRatio()(inputs[0:1, :], outputs[0:1, :]))
        
        model_results_snr[-1].append(SignalNoiseRatio()(outputs[0:1, :], outputs[0:1, :]))
        model_results_sdr[-1].append(SignalDistortionRatio()(outputs[0:1, :], outputs[0:1, :]))
        model_results_sisdr[-1].append(ScaleInvariantSignalDistortionRatio()(outputs[0:1, :], outputs[0:1, :]))

    for i in range(len(model_results_snr)):
        mean_snr = np.mean(model_results_snr[i])
        ci_snr = 1.96 * np.std(model_results_snr[i]) / np.sqrt(len(model_results_snr[i]))
        
        mean_sdr = np.mean(model_results_sdr[i]) 
        ci_sdr = 1.96 * np.std(model_results_sdr[i]) / np.sqrt(len(model_results_sdr[i]))
        
        mean_sisdr = np.mean(model_results_sisdr[i])
        ci_sisdr = 1.96 * np.std(model_results_sisdr[i]) / np.sqrt(len(model_results_sisdr[i]))
        
        print(f"Model {model_string[i] if i < len(model_string) else 'Input' if i == len(model_string) else 'Output'}:")
        print(f"  SNR: {mean_snr:.2f} ± {ci_snr:.2f}")
        print(f"  SDR: {mean_sdr:.2f} ± {ci_sdr:.2f}")
        print(f"SI-SDR: {mean_sisdr:.2f} ± {ci_sisdr:.2f}\n")
