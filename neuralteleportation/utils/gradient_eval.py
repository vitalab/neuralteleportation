from typing import Union

import torch
from numpy import number
from torch import Tensor
from torch.nn.modules.loss import _Loss

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def gradient_to_weight_norm(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                            loss_fn: _Loss, order: Union[str, number] = 'fro') -> Tensor:
    weights = model.get_weights()
    gradients = model.get_grad(data, target, loss_fn)

    # Compute the gradient/weight ratio where possible
    ratio = gradients / weights

    # Identify where the ratio is numerically unstable (division by 0-valued weights)
    nan_ratio_mask = torch.isnan(ratio)

    # Replace unstable values by statistically representative measures
    ratio[nan_ratio_mask] = ratio[~nan_ratio_mask].mean()

    # Compute the norm of the ratio and move result to CPU (to avoid cluttering GPU if fct is called repeatedly)
    return torch.norm(ratio, p=order)
