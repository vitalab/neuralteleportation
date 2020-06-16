import random
from dataclasses import dataclass
from typing import Tuple

import torch.optim as optim
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import test, train_epoch


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    teleport_every_n_epochs: int = 2
    teleport_prob: float = 1.  # Always teleport by default when reaching `teleport_every_n_epochs`


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TeleportationTrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)

    for epoch in range(config.epochs):
        if (epoch % config.teleport_every_n_epochs) == 0 and epoch > 0:
            if random.random() < config.teleport_prob:
                print("Applying random COB to model in training")
                model.random_teleport(cob_range=config.cob_range, sampling_type=config.cob_sampling)

                # Initialze a new optimizer using the model's new parameters
                optimizer = optim.SGD(model.parameters(), lr=config.lr)
            else:
                print("Skipping COB")

        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch + 1, device=config.device)

        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    return model


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
