import os
from typing import Union, List
from Dataloader.Dataloader import EarsDataset, ConvTasNetDataLoader
from conv_tasnet_causal import ConvTasNet
import torch
from tqdm import tqdm
import random

num_sources = 2
enc_kernel_size = 16
enc_num_feats = 512
msk_kernel_size = 3
msk_num_feats = 128
msk_num_hidden_feats = 512
msk_num_layers = 8
msk_num_stacks = 3
msk_activate = 'sigmoid'
blackhole_path = os.getenv('BLACKHOLE')

def get_train_dataset(mock: bool = False):
    print("Loading dataset...")
    if mock:
        dataset_TRN = [[torch.rand(1, 1, 149158), torch.rand(1, 1, 149158)]]
    else:
        dataset_TRN = EarsDataset(data_dir=os.path.join(blackhole_path, "EARS-WHAM"), subset = 'train', normalize = False)
        print(dataset_TRN[0][0].shape, dataset_TRN[0][0].dtype)
    print("Dataset loaded")
    return dataset_TRN

def load_models():
    models_load_strings = ["models/student_only_labels_cpu.pth", "models/student_only_teacher_cpu.pth", "models/student_partly_teacher_cpu.pth"]
    models = []
    for model_load_string in tqdm(models_load_strings, desc = "Loading models..."):
        model = ConvTasNet(
            num_sources=num_sources,
            enc_kernel_size=enc_kernel_size, 
            enc_num_feats=enc_num_feats,
            msk_kernel_size=msk_kernel_size,
            msk_num_feats=msk_num_feats,
            msk_num_hidden_feats=msk_num_hidden_feats,
            msk_num_layers=msk_num_layers,
            msk_num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            causal = True,
            save_intermediate_values = True
        )
        model.load_state_dict(torch.load(model_load_string, weights_only = True))
        model.eval()
        models.append(model)
    return models

def get_model_predictions_and_data(
    models: Union[List[ConvTasNet], None] = None, 
    mock: bool = False, 
    datapoints: int = 1, 
    save_memory: bool = False, 
    deterministic: bool = False
    ) -> List[tuple]:
    """
    This function loads the models and returns the predictions and the data
    Both the predictions and outputs are only the clean data.
    Datapoints is the number of datapoints in which you want the predictions, inputs and outputs.
    The output size is then: (datapoints, 3), with element = length of models. 
    Each element, the inputs and the outputs will have the same shape: (1, 1, x), where x is the length of the audio clip.
    """
    assert datapoints > 0
    assert not (mock is True and datapoints > 1), f"If you want to mock, you should only want 1 datapoint, but you want {datapoints}"
    if models is None:
        models = load_models()
    dataset_TRN = get_train_dataset(mock = mock)
    assert len(dataset_TRN) >= datapoints
    indexes = random.sample(range(len(dataset_TRN)), datapoints) if not deterministic else range(datapoints)
    result = []
    for i in range(datapoints):
        inputs, outputs = dataset_TRN[indexes[i]]
        inputs.unsqueeze_(0)
        outputs.unsqueeze_(0)
        if save_memory:
            inputs, outputs = inputs[:, :, :16000], outputs[:, :, :16000]
        model_outputs = []
        with torch.no_grad():
            for model in tqdm(models, desc = f"Getting model predictions for {i}'th datapoint"):
                predictions, _ = model(inputs)
                predictions = predictions[:, 0:1, :] # Only the clean part
                model_outputs.append(predictions)
                assert predictions.shape == outputs.shape, f"Predictions and outputs have different shapes: {predictions.shape} and {outputs.shape}"
                assert predictions.shape == inputs.shape, f"Predictions and inputs have different shapes: {predictions.shape} and {inputs.shape}"
        result.append((model_outputs, inputs, outputs))

    return result

if __name__ == "__main__":
    # Settings when running on bsub should then be mock = False, save_memory = False, deterministic = _____ , datapoints > 1
    results = get_model_predictions_and_data(mock = False, save_memory = True, datapoints = 2, deterministic = False)
    
    ## TO USE:
    # for (predictions, inputs, outputs) in results:
    #     for prediction in predictions:
    #         do_something(prediction, inputs, outputs) # which all have the same shape

