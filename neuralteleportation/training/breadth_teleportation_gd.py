from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch.optim as optim
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
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
    teleportation_model = NeuralTeleportationModel(network=deepcopy(model), input_shape=config.input_shape)
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


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn

    mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

    models = [
        MLPCOB(),
        vgg16COB(num_classes=10, input_channels=1),
        resnet18COB(num_classes=10, input_channels=1),
    ]

    config = TeleportationTrainingConfig()
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    for model in models:
        print(f"Training {model.__class__.__name__} model using multiple COB "
              f"every {config.teleport_every_n_epochs} epochs")
        models = train(model, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_val)
        for id, model in models.items():
            print("Testing {}: {} \n".format(id, test(model, mnist_test, metrics, config)))
        print()
