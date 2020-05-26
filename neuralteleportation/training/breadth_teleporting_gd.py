from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Sequence

import torch.optim as optim
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import test, train_epoch


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    starting_epoch: int = 1
    teleport_every_n_epochs: int = 2
    num_teleportations: int = 1


def train(model: Union[nn.Module, Tuple[str, nn.Module]], train_dataset: Dataset, metrics: TrainingMetrics,
          config: TeleportationTrainingConfig, val_dataset: Dataset = None, optimizer: Optimizer = None) \
        -> Dict[str, nn.Module]:
    # If the model is not named (at the first iteration), initialize its name based on its class
    if type(model) is tuple:
        model_name, model = model
    else:
        model_name = model.__class__.__name__

    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    stopping_epoch = min(config.starting_epoch + config.teleport_every_n_epochs, config.epochs + 1)
    for epoch in range(config.starting_epoch, stopping_epoch):
        print(f'Training epoch {epoch} for {model_name} ...')
        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)
        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    # Update new starting epoch for the next iteration of model training
    config.starting_epoch += config.teleport_every_n_epochs

    # Determine if training has reached its end
    # TODO Add test for convergence
    is_train_end = config.starting_epoch == config.epochs + 1

    if is_train_end:
        trained_models = {f'{model_name}_0': model}
    else:
        # Teleport the model and train each teleportation recursively
        trained_models = teleport_and_train((model_name, model), train_dataset, metrics, config, optimizer,
                                            val_dataset=val_dataset)

    return trained_models


def teleport_and_train(model: Tuple[str, nn.Module], train_dataset: Dataset, metrics: TrainingMetrics,
                       config: TeleportationTrainingConfig, optimizer: Optimizer, val_dataset: Dataset = None) \
        -> Dict[str, nn.Module]:
    model_name, model = model

    # Teleport the model to obtain N different models corresponding to the same function
    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    teleportation_model = NeuralTeleportationModel(network=deepcopy(model), input_shape=(1,) + config.input_shape)
    teleported_models = [deepcopy(teleportation_model.random_teleport()) for _ in range(config.num_teleportations)]

    # Call recursively the training algorithm on teleported models, with less epochs left to perform
    # The non-teleported model uses the previous training iterations' optimizer,
    # and the teleported models initialize new optimizers
    trained_models = train((f'{model_name}_0', model), train_dataset, metrics, deepcopy(config),
                           val_dataset=val_dataset, optimizer=optimizer)
    for idx, teleported_model in enumerate(teleported_models, 1):
        trained_teleportations = train((f'{model_name}_{idx}', teleported_model), train_dataset, metrics,
                                       deepcopy(config), val_dataset=val_dataset)
        trained_models.update(trained_teleportations)

    return trained_models


def run_models(models: Sequence[nn.Module], config: TeleportationTrainingConfig, metrics: TrainingMetrics,
               train_set: VisionDataset, test_set: VisionDataset, val_set: VisionDataset = None):
    for model in models:
        print(f"Training {model.__class__.__name__} model using multiple COB "
              f"every {config.teleport_every_n_epochs} epochs")
        trained_models = train(model, train_dataset=train_set, metrics=metrics, config=deepcopy(config),
                               val_dataset=val_set)
        for id, trained_model in trained_models.items():
            print("Testing {}: {} \n".format(id, test(trained_model, test_set, metrics, config)))
        print()


if __name__ == '__main__':
    from neuralteleportation.training.experiments_setup import (
        get_mnist_models, get_mnist_datasets, get_cifar10_models, get_cifar10_datasets
    )
    from neuralteleportation.metrics import accuracy

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on MNIST
    mnist_train, mnist_val, mnist_test = get_mnist_datasets()
    config = TeleportationTrainingConfig()
    run_models(get_mnist_models(), config, metrics, mnist_train, mnist_test, val_set=mnist_val)

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    config = TeleportationTrainingConfig(input_shape=(3, 32, 32))
    run_models(get_cifar10_models(), config, metrics, cifar10_train, cifar10_test, val_set=cifar10_val)
