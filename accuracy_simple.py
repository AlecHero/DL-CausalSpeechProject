from load_models import get_model_predictions_and_data
import torch




def save_results(n_datapoints, save_path):
    datapoints = n_datapoints
    results = get_model_predictions_and_data(
        mock = False,
        save_memory = True,
        datapoints = datapoints,
        deterministic = False
    )

    if save_path: torch.save({ "results": results }, save_path)

if __name__ == "__main__":
    save_path = "/zhome/f8/2/187151/DL-CausalSpeechProject/results.pt"
    save_results