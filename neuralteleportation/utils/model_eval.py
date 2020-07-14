from numbers import Number
from typing import Union

import torch
from numpy import number
from torch import Tensor, optim

from neuralteleportation.experiments.optim_teleporting_gd import TeleportationTrainingConfig
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics


def weighted_grad_norm(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                       metrics: TrainingMetrics, config: TeleportationTrainingConfig,
                       order: Union[str, number] = 'fro') -> Number:
    weights = model.get_weights()
    gradients = torch.stack([model.get_grad(data_batch, target_batch, metrics.criterion)
                             for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Compute the gradient/weight ratio where possible
    ratio = gradients / weights

    # Identify where the ratio is numerically unstable (division by 0-valued weights)
    nan_ratio_mask = torch.isnan(ratio)

    # Replace unstable values by statistically representative measures
    ratio[nan_ratio_mask] = ratio[~nan_ratio_mask].mean()

    # Compute the norm of the ratio and move result to CPU (to avoid cluttering GPU if fct is called repeatedly)
    return torch.norm(ratio, p=order).item()


def loss_lookahead_diff(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                        metrics: TrainingMetrics, config: TeleportationTrainingConfig) -> Number:
    # Save the weights of the model before performing the lookahead
    model_weights = model.get_weights()

    # Initialize a new optimizer to perform lookahead
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    optimizer.zero_grad()

    # Compute loss at the teleported point
    loss = torch.stack([metrics.criterion(model(data_batch), target_batch)
                        for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Take a step using the gradient at the teleported point
    loss.backward()

    # Compute loss after the optimizer step
    lookahead_loss = torch.stack([metrics.criterion(model(data_batch), target_batch)
                                  for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Compute the difference between the lookahead loss and the original loss
    loss_diff = loss - lookahead_loss

    # Restore the weights of the model from before the lookahead
    model.set_weights(model_weights)

    return loss_diff.item()
