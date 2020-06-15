import operator
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch.optim as optim
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import test, train_epoch
from neuralteleportation.utils.gradient_eval import gradient_to_weight_norm


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    teleport_every_n_epochs: int = 2
    num_teleportations: int = 10
    comparison_metric = (gradient_to_weight_norm, operator.gt)


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TeleportationTrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    for epoch in range(config.epochs):
        if (epoch % config.teleport_every_n_epochs) == 0 and epoch > 0:
            print(f"Applying {config.num_teleportations} random COB to compare gradients in training")
            model = optimize_model_gradients(model, train_dataset, metrics, config)

            # Force a new optimizer in case a teleportation of the original model was chosen
            optimizer = optim.SGD(model.parameters(), lr=config.lr)

        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)

        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    return model


def optimize_model_gradients(model: nn.Module, train_dataset: Dataset,
                             metrics: TrainingMetrics, config: TeleportationTrainingConfig) -> nn.Module:
    # Extract a single batch on which to compute gradients for each model to be compared
    # TODO: Should we try accumulating the gradients over a whole epoch?
    data, target = next(iter(DataLoader(train_dataset, batch_size=config.batch_size)))
    data = data.to(device=config.device)
    target = target.to(device=config.device)

    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)

    # Unpack the configuration for the metric to use to optimize gradients
    metric_func, metric_compare = config.comparison_metric

    optimal_metric = metric_func(model, data, target, metrics.criterion).cpu()
    model.cpu()  # Move model to CPU to avoid having 2 models on the GPU (to avoid possible CUDA OOM error)
    optimal_model = model

    for _ in range(config.num_teleportations):
        teleported_model = deepcopy(model).random_teleport()
        teleported_model.to(config.device)  # Move model back to chosen device before computing gradients
        metric = metric_func(teleported_model, data, target, metrics.criterion).cpu()
        teleported_model.cpu()  # Move model back to CPU after computation is done (to avoid possible CUDA OOM error)
        if metric_compare(metric, optimal_metric):
            optimal_model = teleported_model
            optimal_metric = metric

    return optimal_model.network.to(config.device)


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
