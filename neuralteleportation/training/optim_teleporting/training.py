from copy import deepcopy

import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics
from neuralteleportation.training.optim_teleporting.config import TeleportationTrainingConfig
from neuralteleportation.training.training import train_epoch, test


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TeleportationTrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    for epoch in range(config.epochs):
        if (epoch % config.teleport_every_n_epochs) == 0 and epoch > 0:
            print(f"Selecting best of {config.num_teleportations} random COBs "
                  f"w.r.t. {config.comparison_metric[0].__name__}")
            model = _select_optimal_model(model, train_dataset, metrics, config)

            # Force a new optimizer in case a teleportation of the original model was chosen
            optimizer = optim.SGD(model.parameters(), lr=config.lr)

        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)

        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    return model


def _select_optimal_model(model: nn.Module, train_dataset: Dataset,
                          metrics: TrainingMetrics, config: TeleportationTrainingConfig) -> nn.Module:
    # Extract a single batch on which to compute gradients for each model to be compared
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    data, target = [], []
    for (data_batch, target_batch), _ in zip(dataloader, range(config.num_batches)):
        data.append(data_batch)
        target.append(target_batch)
    data = torch.stack(data).to(device=config.device)
    target = torch.stack(target).to(device=config.device)

    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)

    # Unpack the configuration for the metric to use to optimize gradients
    metric_func, metric_compare = config.comparison_metric

    optimal_metric = metric_func(model, data, target, metrics, config)
    model.cpu()  # Move model to CPU to avoid having 2 models on the GPU (to avoid possible CUDA OOM error)
    optimal_model = model

    for _ in range(config.num_teleportations):
        teleported_model = deepcopy(model).random_teleport(cob_range=config.cob_range,
                                                           sampling_type=config.cob_sampling)
        teleported_model.to(config.device)  # Move model back to chosen device before computing gradients
        metric = metric_func(teleported_model, data, target, metrics, config)
        teleported_model.cpu()  # Move model back to CPU after computation is done (to avoid possible CUDA OOM error)
        if metric_compare(metric, optimal_metric):
            optimal_model = teleported_model
            optimal_metric = metric

    return optimal_model.network.to(config.device)
