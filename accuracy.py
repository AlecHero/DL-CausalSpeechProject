from load_models import get_model_predictions_and_data

import torch
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio, ScaleInvariantSignalDistortionRatio
from Dataloader.Dataloader import EarsDataset
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np

# Det er også en god ide at rapportere metrikker som improvements -
# hvis man beregner e.g. SI-SDR mellem noisy speech (netværkets input)
# og clean speech på hele test sættet vil man få en non-zero SI-SDR
# som kan tænkes som værende dit performance floor (hvad er SI-SDR
# hvis man ingen processing har, men bare outputter input direkte?).
# Så kan man rapportere SI-SDRi (SI-SDR improvement) som forskellen
# på SI-SDR floor og den gennemsnitlige SI-SDR man får ved at bruge
# netværket til at processere input.

def compute_metrics(results, save_path=None):
    baseline_metrics, prediction_metrics = [], []
    
    for (predictions, inputs, outputs) in results:
        snr_scores = SignalNoiseRatio()(inputs, outputs)
        sdr_scores = SignalDistortionRatio()(inputs, outputs)
        si_sdr_scores = ScaleInvariantSignalDistortionRatio()(inputs, outputs)
        baseline_scores = [snr_scores, sdr_scores, si_sdr_scores]
        
        num_models = len(predictions)
        prediction_scores = [0]*num_models
        for i in range(num_models):
            pred_snr_scores = SignalNoiseRatio()(inputs, predictions[i])
            pred_sdr_scores = SignalDistortionRatio()(inputs, predictions[i])
            pred_si_sdr_scores = ScaleInvariantSignalDistortionRatio()(inputs, predictions[i])
            
            prediction_scores[i] = [pred_snr_scores, pred_sdr_scores, pred_si_sdr_scores]
        
        baseline_metrics.append(baseline_scores)
        prediction_metrics.append(prediction_scores)
    
    if save_path:
        torch.save({
            "baseline_metrics": baseline_metrics,
            "prediction_metrics": prediction_metrics
        }, save_path)
    
    return baseline_metrics, prediction_metrics

if __name__ == "__main__":
    datapoints = 632
    results = get_model_predictions_and_data(
        mock = False,
        save_memory = True,
        datapoints = datapoints,
        deterministic = False
    )
    
    save_path = r"/zhome/f8/2/187151/DL-CausalSpeechProject/Plots/SNR_SDR"
    save_path = save_path + f"/metrics{datapoints}.pt"
    compute_metrics(results, save_path=save_path)