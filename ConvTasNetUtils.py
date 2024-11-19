import torch
import math

def calculate_fmaps(weight: torch.Tensor) -> tuple[int, int, int]:
    num_outputs, num_inputs, *receptive_field_fmaps = weight.shape
    receptive_field = math.prod(receptive_field_fmaps)
    return num_inputs, num_outputs, receptive_field


def calculate_fan_in_out(weight: torch.Tensor) -> tuple[int, int]:
    fan_in, fan_out, receptive_fields = calculate_fmaps(weight)
    return fan_in * receptive_fields, fan_out * receptive_fields


def spectral_fan(fan_in: int, fan_out: int) -> float:
    """Parametrization 1 from https://arxiv.org/abs/2310.17813."""
    return fan_in**2 / min(fan_in, fan_out)


@torch.no_grad()
def spectral_normal_(weight: torch.Tensor, gain: float = 1) -> None:
    fan_in, fan_out = calculate_fan_in_out(weight)
    fan = spectral_fan(fan_in, fan_out)
    std = gain / math.sqrt(fan)
    weight.normal_(0, std)