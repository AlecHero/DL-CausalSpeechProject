from load_models import get_model_predictions_and_data
from eval import Loss, sisdr_to_sdr
import numpy as np


if __name__ == "__main__":
    model_string = ["only labels", "only teacher", "partly teacher", "e2e", "baseline", "baseline2"]
    datapoints = 593 # 593
    results = get_model_predictions_and_data(mock = False, save_memory = True, datapoints = datapoints, deterministic = True)
    loss_func = Loss()    
    ## TO USE:
    model_results = [[] for _ in range(len(results[0][0]) + 2)]
    for (predictions, inputs, outputs) in results:
        for i, (prediction, model_load_string) in enumerate(predictions):
            model_results[i].append(loss_func.sisnr(prediction, outputs))
        model_results[-2].append(loss_func.sisnr(inputs, outputs))
        model_results[-1].append(loss_func.sisnr(outputs, outputs))
    # results = [[12.82], [13.62], [13.98], [13.94]]
    
    for i, model_result in enumerate(model_results):
        mean_sisnr = np.mean(model_result)
        ci_sisnr = 1.96 * np.std(model_result) / np.sqrt(len(model_result))
        
        sdr_values = [sisdr_to_sdr(est, scaling_factor=1.047) for est in model_results[i]]
        mean_sdr = np.mean(sdr_values)
        ci_sdr = 1.96 * np.std(sdr_values) / np.sqrt(len(sdr_values))
        
        print(f"Model {model_string[i]} mean SI-SDR: {mean_sisnr:.2f} ± {ci_sisnr:.2f}")
        print(f"Model {model_string[i]} mean    SDR: {mean_sdr:.2f} ± {ci_sdr:.2f}\n")


