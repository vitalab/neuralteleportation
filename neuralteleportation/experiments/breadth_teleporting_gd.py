from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple, Union

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
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    # Always move model to GPU before training
    model.cuda()

    stopping_epoch = min(config.starting_epoch + config.teleport_every_n_epochs, config.epochs + 1)
    for epoch in range(config.starting_epoch, stopping_epoch):
        print(f'Training epoch {epoch} for {model_name} ...')
        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)
        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))

    # Always move model off-GPU after training
    model.cpu()

    # Update new starting epoch for the next iteration of model training
    config.starting_epoch += config.teleport_every_n_epochs

    # Determine if training has reached its end
    # TODO Add test for convergence
    is_train_end = config.starting_epoch >= config.epochs + 1

    if is_train_end:
        trained_models = {f'{model_name}_0': model}
    else:
        # Teleport the model and train each teleportation recursively
        trained_models = _teleport_and_train((model_name, model), train_dataset, metrics, config, optimizer,
                                             val_dataset=val_dataset)

    return trained_models


def _teleport_and_train(model: Tuple[str, nn.Module], train_dataset: Dataset, metrics: TrainingMetrics,
                        config: TeleportationTrainingConfig, optimizer: Optimizer, val_dataset: Dataset = None) \
        -> Dict[str, nn.Module]:
    model_name, model = model

    # Teleport the model to obtain N different models corresponding to the same function
    # NOTE: The input shape passed to `NeuralTeleportationModel` must take into account the batch dimension
    teleportation_model = NeuralTeleportationModel(network=model, input_shape=(2,) + config.input_shape)
    teleported_models = [deepcopy(teleportation_model).random_teleport().network
                         for _ in range(config.num_teleportations)]

    # Call recursively the training algorithm on teleported models, with less epochs left to perform
    # The non-teleported model uses the previous training iterations' optimizer,
    # and the teleported models initialize new optimizers (with the new models' parameters)
    trained_models = train((f'{model_name}_0', model), train_dataset, metrics, deepcopy(config),
                           val_dataset=val_dataset, optimizer=optimizer)
    for idx, teleported_model in enumerate(teleported_models, 1):
        trained_teleportations = train((f'{model_name}_{idx}', teleported_model), train_dataset, metrics,
                                       deepcopy(config), val_dataset=val_dataset)
        trained_models.update(trained_teleportations)

    return trained_models


if __name__ == '__main__':
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.experiment_run import run_multi_output_training

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    config = TeleportationTrainingConfig(input_shape=(3, 32, 32), device='cuda')
    run_multi_output_training(train, get_cifar10_models(), config, metrics,
                              cifar10_train, cifar10_test, val_set=cifar10_val)
