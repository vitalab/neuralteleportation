from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Sequence, List, Union

import torch
import torch.optim as optim
from numpy import number
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import test, train_epoch


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    teleport_every_n_epochs: int = 2
    num_teleportations: int = 10


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TeleportationTrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    for epoch in range(config.epochs):
        if (epoch % config.teleport_every_n_epochs) == 0 and epoch > 0:
            print(f"Applying {config.num_teleportations} random COB to compare gradients in training")
            models = _teleport_model(model, config)
            model = _select_optimal_model(models, train_dataset, metrics, config)

            # Force a new optimizer in case a teleportation of the original model was chosen
            optimizer = optim.SGD(model.parameters(), lr=config.lr)

        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)

        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    return model


def _teleport_model(model: nn.Module, config: TeleportationTrainingConfig) -> List[NeuralTeleportationModel]:
    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    model = NeuralTeleportationModel(network=model, input_shape=(1,) + config.input_shape)
    model.cpu()  # Move model to CPU before teleporting it (to avoid possible CUDA OOM error)

    # Include the non-teleported, original model as a possible model
    models = [model]

    # Teleport the model to obtain N different models corresponding to the same function
    models.extend(deepcopy(model).random_teleport() for _ in range(config.num_teleportations))

    return models


def _select_optimal_model(models: Sequence[NeuralTeleportationModel], train_dataset: Dataset,
                          metrics: TrainingMetrics, config: TeleportationTrainingConfig) -> nn.Module:
    # Extract a single batch on which to compute gradients for each model to be compared
    # TODO: Should we try accumulating the gradients over a whole epoch?
    data, target = next(iter(DataLoader(train_dataset, batch_size=config.batch_size)))
    data = data.to(device=config.device)
    target = target.to(device=config.device)

    # Select the model that maximizes the overall ratio between gradients and weights
    optimal_model = max(models,
                        key=lambda model: _compute_gradient_to_weight_norm(model, data, target, metrics.criterion,
                                                                           device=config.device))

    # Move model with optimal gradient back to chosen device before resuming training
    return optimal_model.to(config.device)


def _compute_gradient_to_weight_norm(model: NeuralTeleportationModel, data: Tensor, target: Tensor,
                                     loss_fn: _Loss, order: Union[str, number] = 'fro', device: str = 'cpu') \
        -> Tensor:
    model.to(device)  # Move model back to chosen device before computing gradients
    weights = model.get_weights()
    gradients = model.get_grad(data, target, loss_fn)
    model.cpu()  # Move model back to CPU after computation is done (to avoid possible CUDA OOM error)

    # Compute the gradient/weight ratio where possible
    ratio = gradients / weights

    # Identify where the ratio is numerically unstable (division by 0-valued weights)
    nan_ratio_mask = torch.isnan(ratio)

    # Replace unstable values by statistically representative measures
    ratio[nan_ratio_mask] = ratio[~nan_ratio_mask].mean()

    # Compute the norm of the ratio and move result to CPU (to avoid cluttering GPU if fct is called repeatedly)
    return torch.norm(ratio, p=order).cpu()


if __name__ == '__main__':
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.experiment_run import run_single_output_training

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    config = TeleportationTrainingConfig(input_shape=(3, 32, 32), device='cuda')
    run_single_output_training(train, get_cifar10_models(device='cuda'), config, metrics,
                               cifar10_train, cifar10_test, val_set=cifar10_val)
