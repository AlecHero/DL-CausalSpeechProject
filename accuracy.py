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
            pred_snr_scores = SignalNoiseRatio()(predictions[i], outputs)
            pred_sdr_scores = SignalDistortionRatio()(predictions[i], outputs)
            pred_si_sdr_scores = ScaleInvariantSignalDistortionRatio()(predictions[i], outputs)
            
            prediction_scores[i] = [pred_snr_scores, pred_sdr_scores, pred_si_sdr_scores]
        
        baseline_metrics.append(baseline_scores)
        prediction_metrics.append(prediction_scores)
    
    if save_path:
        torch.save({
            "baseline_metrics": baseline_metrics,
            "prediction_metrics": prediction_metrics
        }, save_path)
    
    return baseline_metrics, prediction_metrics


### PLOTTING
metric_names = ['SNR', 'SDR', 'SI-SDR']
model_names = ["student_only_labels", "student_only_teacher", "student_partly_teacher"]
num_metrics = 3

def plot_metrics(metrics_path, save_path=None):
    import seaborn as sns
    sns.set_theme()

    metrics = torch.load(metrics_path)
    baseline_metrics = torch.tensor(metrics["baseline_metrics"])
    prediction_metrics = torch.tensor(metrics["prediction_metrics"])
    
    num_models = prediction_metrics.shape[1]
    palette = sns.color_palette("Dark2")

    # metric per model
    for model_idx in range(num_models):
        fig, axes = plt.subplots(num_models, 1, constrained_layout=True)
        fig.set_size_inches(12, 6)
        fig.set_dpi(100)
        
        fig.suptitle(f"Metrics | Model: {model_names[model_idx]}")
        fig.supxlabel("Datapoints")
        fig.supylabel("Metric Specific Values")
        
        for metric_idx, ax in enumerate(axes):
            ax.set_title(f"{metric_names[metric_idx]} - values")
            ax.plot(prediction_metrics[:, :, model_idx][:, metric_idx], color=palette[metric_idx])
        
        if save_path:
            fig.savefig(save_path+fr"\metrics_model_{model_idx+1}_{model_names[model_idx]}")

    # improvement metric
    for metric_idx in range(num_metrics):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        fig.set_dpi(100)
        
        fig.suptitle(f"{metric_names[metric_idx]} - improvement values")
        fig.supxlabel("Datapoints")
        fig.supylabel("Metric Specific Values")
        
        for model_idx in range(num_metrics):
            base = baseline_metrics[:, metric_idx]
            pred = prediction_metrics[:, :, model_idx][:, metric_idx]
            improvement = base - pred
            ax.plot(improvement, label=model_names[model_idx])
            ax.set_xlim(0, 100)
            ax.legend()
        
        if save_path:
            fig.savefig(save_path+fr"\improvement_metric_{metric_names[metric_idx]}_1st100")

def get_mean_conf(data_path, printer=True):
    from scipy.stats import t
    import numpy as np

    metrics = torch.load(data_path)
    baseline_metrics = torch.tensor(metrics["baseline_metrics"])
    prediction_metrics = torch.tensor(metrics["prediction_metrics"])

    n, num_models, num_metrics = prediction_metrics.shape

    means = []
    conf_intervals = []

    for model_idx in range(num_models):
        for metric_idx in range(num_metrics):
            data = np.asarray(baseline_metrics[:, metric_idx] - prediction_metrics[:, metric_idx, model_idx])
            mean = np.mean(data)
            margin = t.ppf((1 + 0.95) / 2, n-1) * np.std(data, ddof=1) / np.sqrt(n)
            
            if printer:
                print(model_names[model_idx], "\t", round(mean, 1), "\t", round(margin, 2), "\t", metric_names[metric_idx])
            means.append(mean)
            conf_intervals.append(margin)
        if printer:
            print()

    return means, conf_intervals


def plot_conf(data_path):
    import numpy as np

    means, conf_intervals = get_mean_conf(data_path, printer=False)
    
    metrics = torch.load(data_path)
    baseline_metrics = torch.tensor(metrics["baseline_metrics"])
    prediction_metrics = torch.tensor(metrics["prediction_metrics"])

    n, num_models, num_metrics = prediction_metrics.shape
    
    means = np.array(means).reshape(num_models, num_metrics)
    conf_intervals = np.array(conf_intervals).reshape(num_models, num_metrics)
    
    x = np.arange(num_metrics)  # Metric indices
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_idx in range(num_models):
        ax.errorbar(x, means[model_idx], yerr=conf_intervals[model_idx], label=f"Model {model_idx + 1}", capsize=5, fmt='o')

    # Customize plot
    ax.set_title("Confidence Intervals for Metrics by Model")
    ax.set_xlabel("Metric Index")
    ax.set_ylabel("Mean Difference ± CI")
    ax.legend(title="Models")
    plt.xticks(x, [f"Metric {i+1}" for i in x])  # Optional: label metrics
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


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