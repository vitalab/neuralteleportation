from numbers import Number
from typing import Union, Callable

import torch
from numpy import number
from torch import Tensor
from torch.nn.modules.loss import _Loss

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

GradientEvalFunc = Callable[[NeuralTeleportationModel, Tensor, Tensor, _Loss], Number]


def gradient_to_weight_norm(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                            loss_fn: _Loss, order: Union[str, number] = 'fro') -> Number:
    weights = model.get_weights()
    gradients = torch.stack([model.get_grad(data_batch, target_batch, loss_fn)
                             for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Compute the gradient/weight ratio where possible
    ratio = gradients / weights

    # Identify where the ratio is numerically unstable (division by 0-valued weights)
    nan_ratio_mask = torch.isnan(ratio)

    # Replace unstable values by statistically representative measures
    ratio[nan_ratio_mask] = ratio[~nan_ratio_mask].mean()

    # Compute the norm of the ratio and move result to CPU (to avoid cluttering GPU if fct is called repeatedly)
    return torch.norm(ratio, p=order).item()
