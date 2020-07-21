from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Number
from typing import Callable, Union

import torch
from numpy import number
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TeleportationTrainingConfig
from neuralteleportation.training.experiment_setup import get_optimizer_from_model_and_config


def teleport_model_to_optimize_metric(model: NeuralTeleportationModel, train_dataset: Dataset, metrics: TrainingMetrics,
                                      config: "OptimalTeleportationTrainingConfig", **kwargs) \
        -> NeuralTeleportationModel:
    print(f"Selecting best of {config.num_teleportations} random COBs "
          f"w.r.t. {config.optim_metric.__name__}")

    # Extract a single batch on which to compute gradients for each model to be compared
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    data, target = [], []
    for (data_batch, target_batch), _ in zip(dataloader, range(config.num_batches)):
        data.append(data_batch)
        target.append(target_batch)
    data = torch.stack(data).to(device=config.device)
    target = torch.stack(target).to(device=config.device)

    optimal_metric = config.optim_metric(model=model, data=data, target=target, metrics=metrics, config=config)
    model.cpu()  # Move model to CPU to avoid having 2 models on the GPU (to avoid possible CUDA OOM error)
    optimal_model = model

    for _ in range(config.num_teleportations):
        teleported_model = deepcopy(model).random_teleport(cob_range=config.cob_range,
                                                           sampling_type=config.cob_sampling)
        teleported_model.to(config.device)  # Move model back to chosen device before computing gradients
        metric = config.optim_metric(model=teleported_model, data=data, target=target, metrics=metrics, config=config)
        teleported_model.cpu()  # Move model back to CPU after computation is done (to avoid possible CUDA OOM error)
        if metric > optimal_metric:
            optimal_model = teleported_model
            optimal_metric = metric

    return optimal_model.to(config.device)


@dataclass
class OptimalTeleportationTrainingConfig(TeleportationTrainingConfig):
    teleport_fn: Callable = field(default=teleport_model_to_optimize_metric)
    num_teleportations: int = 10
    num_batches: int = 1
    optim_metric: Callable[..., Number] = None  # Required


def weighted_grad_norm(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                       metrics: TrainingMetrics, order: Union[str, number] = 'fro', **kwargs) -> Number:
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
                        metrics: TrainingMetrics, config: OptimalTeleportationTrainingConfig, **kwargs) -> Number:
    # Save the state of the model, prior to performing the lookahead
    state_dict = model.state_dict()

    # Initialize a new optimizer to perform lookahead
    optimizer = get_optimizer_from_model_and_config(model, config)
    optimizer.zero_grad()

    # Compute loss at the teleported point
    loss = torch.stack([metrics.criterion(model(data_batch), target_batch)
                        for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Take a step using the gradient at the teleported point
    loss.backward()

    # Compute loss after the optimizer step
    lookahead_loss = torch.stack([metrics.criterion(model(data_batch), target_batch)
                                  for data_batch, target_batch in zip(data, target)]).mean(dim=0)

    # Restore the state of the model prior to the lookahead
    model.load_state_dict(state_dict)

    # Compute the difference between the lookahead loss and the original loss
    return (loss - lookahead_loss).item()
