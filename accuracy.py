from load_models import get_model_predictions_and_data, get_train_dataset

from torchmetrics.audio import SignalNoiseRatio as SNR
from torchmetrics.audio import SignalDistortionRatio as SDR
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR
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

def accuracy_metrics(results, i):
    # for (predictions, inputs, outputs) in results:
    #     # predictions len = 3 (models) = denoised
    #     # inputs = noisy
    #     # outputs = clean
        
    #     si_sdr_scores = SI_SDR()(inputs, outputs)
    #     sdr_scores = SDR()(inputs, outputs)
    #     snr_scores = SNR()(inputs, outputs)

    sdr = SDR()
    snr = SNR()
    si_sdr = SI_SDR()
    
    predictions, inputs, outputs = results[0]
    
    si_sdr_scores = si_sdr(inputs, outputs)
    sdr_scores = sdr(inputs, outputs)
    snr_scores = snr(inputs, outputs)
    fig, ax snr.plot()
    
    
    return si_sdr_scores, sdr_scores, snr_scores
    

if __name__ == "__main__":
    results = get_model_predictions_and_data(
        mock = False,
        save_memory = True,
        datapoints = 2,
        deterministic = False
    )
    
    print(accuracy_metrics(results, 0))