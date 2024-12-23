import os
from typing import Union, List
from Dataloader.Dataloader import EarsDataset, ConvTasNetDataLoader
from conv_tasnet_causal import ConvTasNet
import torch
from tqdm import tqdm
import random
from wav_generator import save_to_wav
from typing import List, Optional

num_sources = 2
enc_kernel_size = 16
enc_num_feats = 512
msk_kernel_size = 3
msk_num_feats = 128
msk_num_hidden_feats = 512
msk_num_layers = 8
msk_num_stacks = 3
msk_activate = 'sigmoid'

def get_val_dataset(data_path: str, mock: bool = False):
    print("Loading dataset...")
    if mock:
        dataset_VAL = [[torch.rand(1, 149158), torch.rand(1, 149158)]]
    else:
        dataset_VAL = EarsDataset(data_dir=data_path, subset = 'train', normalize = False)
    print("Dataset loaded")
    return dataset_VAL

def load_models(models_load_strings: Union[List[str], None] = None, device: str = 'cpu', causal: Union[List[bool], None] = None, save_intermediate_values: Union[List[bool], None] = None, dropout: Union[List[float], None] = None) -> List[tuple[ConvTasNet, str]]:
    if models_load_strings is None:
        models_load_strings = [
            "models/student_only_labels_dropout.pth", 
            "models/student_only_labels.pth", 
            "models/student_only_teacher.pth",
            "models/student_partly_teacher.pth",
            "models/student_only_teacher_e2e.pth"
        ]
    models = []
    for i, model_load_string in enumerate(tqdm(models_load_strings, desc = "Loading models...")):
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
            causal = causal[i] if causal is not None else True,
            save_intermediate_values = save_intermediate_values[i] if save_intermediate_values is not None else True,
            dropout = dropout[i] if dropout is not None else 0.0
        )
        if model_load_string != "":
            try: 
                model.load_state_dict(torch.load(model_load_string, weights_only = True, map_location=torch.device(device)))
            except RuntimeError:
                checkpoint = torch.load(model_load_string, weights_only=True, map_location=torch.device(device))
                new_checkpoint_state_dict = {}
                for key, value in checkpoint.items():
                    new_key = key.replace('conv_layers.5.bias', 'conv_layers.6.bias').replace('conv_layers.5.weight', 'conv_layers.6.weight')
                    new_key = new_key.replace('_orig_mod.encoder.', "encoder.").replace('_orig_mod.decoder.', "decoder.").replace('_orig_mod.mask_generator.', "mask_generator.")
                    new_checkpoint_state_dict[new_key] = value
                model.load_state_dict(new_checkpoint_state_dict)
        model.eval()
        models.append((model, model_load_string))
    return models

def get_model_predictions_and_data(
    models: Union[List[ConvTasNet], None] = None, 
    mock: bool = False, 
    datapoints: int = 1, 
    save_memory: bool = False, 
    deterministic: bool = False,
    data_path: Optional[str] = None,
    ) -> List[tuple]:
    """
    This function loads the models and returns the predictions and the data
    Both the predictions and outputs are only the clean data.
    Datapoints is the number of datapoints in which you want the predictions, inputs and outputs.
    The output size is then: (datapoints, 3), with element = length of models. 
    Each element, the inputs and the outputs will have the same shape: (1, 1, x), where x is the length of the audio clip.
    """
    if data_path is None:
        blackhole_path = os.getenv('BLACKHOLE')
        data_path = os.path.join(blackhole_path, "EARS-WHAM")
    assert datapoints > 0
    assert not (mock is True and datapoints > 1), f"If you want to mock, you should only want 1 datapoint, but you want {datapoints}"
    if models is None:
        models = load_models()
    dataset_VAL = get_val_dataset(data_path = data_path, mock = mock)
    assert len(dataset_VAL) >= datapoints, f"Dataset has {len(dataset_VAL)} datapoints, but you want {datapoints}"
    indexes = random.sample(range(len(dataset_VAL)), datapoints) if not deterministic else range(datapoints)
    result = []
    for i in tqdm(range(datapoints), desc = f"Getting model predictions for {datapoints} datapoints"):
        inputs, outputs = dataset_VAL[indexes[i]]
        # min_len = min(len(inputs), len(outputs))
        # inputs, outputs = inputs[..., :min_len], outputs[..., :min_len]
        inputs.unsqueeze_(0) 
        outputs.unsqueeze_(0)
        if save_memory:
            inputs, outputs = inputs[:, :, :16000], outputs[:, :, :16000]
        model_outputs = []
        with torch.no_grad():
            for model, model_load_string in models:
                predictions, _ = model(inputs)
                predictions = predictions[:, 0:1, :] # Only the clean part
                rms = torch.sqrt(torch.mean(predictions**2))
                desired_rms = 0.03  # can change this, not a constant
                predictions = predictions * (desired_rms / (rms + 1e-9))
                predictions = torch.clamp(predictions, -0.9, 0.9)
                model_outputs.append((predictions, model_load_string))
                assert predictions.shape == outputs.shape, f"Predictions and outputs have different shapes: {predictions.shape} and {outputs.shape}"
                assert predictions.shape == inputs.shape, f"Predictions and inputs have different shapes: {predictions.shape} and {inputs.shape}"
            result.append((model_outputs, inputs, outputs))

    return result

if __name__ == "__main__":
    # Settings when running on bsub should then be mock = False, save_memory = False, deterministic = _____ , datapoints > 1
    results = get_model_predictions_and_data(mock = False, save_memory = True, datapoints = 4, deterministic = True)
    
    ## TO USE:
    for (predictions, inputs, outputs) in results:
        for i, (prediction, model_load_string) in enumerate(predictions):
            save_to_wav(prediction[0:1, :].cpu().detach().numpy(), output_filename=f"prediction_{i}d_{model_load_string.split('/')[-1]}.wav")
        inputs = inputs / (torch.max(torch.abs(inputs)) + 1e-9)
        outputs = outputs / (torch.max(torch.abs(outputs)) + 1e-9)
        save_to_wav(inputs[0:1, :].cpu().detach().numpy(), output_filename=f"prediction_input.wav")
        save_to_wav(outputs[0:1, :].cpu().detach().numpy(), output_filename=f"prediction_output.wav")

