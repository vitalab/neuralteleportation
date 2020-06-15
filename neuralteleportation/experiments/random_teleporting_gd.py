import random
from dataclasses import dataclass
from typing import Tuple
import numpy as np

import torch.optim as optim
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.training.training import test, train_epoch
from neuralteleportation.utils.logger import BaseLogger, TensorboardLogger, VisdomLogger, MultiLogger
from neuralteleportation.changeofbasisutils import get_available_cob_sampling_types


@dataclass
class TeleportationTrainingConfig(TrainingConfig):
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    teleport_every_n_epochs: int = 2
    teleport_prob: float = 1.  # Always teleport by default when reaching `teleport_every_n_epochs`
    cob_range: float = 0.5
    cob_sampling: str = "usual"
    vis_logger: BaseLogger = None


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TeleportationTrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    # Initialize an optimizer if there isn't already one
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    model = NeuralTeleportationModel(
        network=model, input_shape=(1,) + config.input_shape)
    if config.vis_logger is not None:
        config.vis_logger.add_text(
            "Config",
            "epochs: {}<br>"
            "teleport_every_n_epochs: {}<br>"
            "cob_range: {}<br>"
            "cob_sampling_type: {}<br>".format(
                config.epochs,
                config.teleport_every_n_epochs,
                config.cob_range,
                config.cob_sampling
            ),
        )

    for epoch in range(config.epochs):
        if (epoch % config.teleport_every_n_epochs) == 0 and epoch > 0:
            if random.random() < config.teleport_prob:
                print("Applying random COB to model in training")
                model.random_teleport(
                    cob_range=config.cob_range, sampling_type=config.cob_sampling)

                # Initialze a new optimizer using the model's new parameters
                optimizer = optim.SGD(model.parameters(), lr=config.lr)
            else:
                print("Skipping COB")

        train_epoch(model, metrics.criterion, optimizer,
                    train_loader, epoch + 1, device=config.device, config=config)

        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))
            if config.vis_logger is not None:
                config.vis_logger.add_scalar("val_loss", val_res["loss"], epoch)
                config.vis_logger.add_scalar("val_accuracy", val_res["accuracy"], epoch)

            if np.isnan(val_res["loss"]) or np.isnan(val_res["accuracy"]):
                print("Stopping: Loss NaN!")
                break

    return model


if __name__ == '__main__':
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.experiment_run import run_single_output_training

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    cob_ranges = [0.7, 1.2]
    cob_samplings = get_available_cob_sampling_types()
    teleport_every_n_epochs = [1, 2, 5, 10]
    for sampling_type in cob_samplings:
        for cob_range in cob_ranges:
            for n in teleport_every_n_epochs:
                env_name = "teleport_{}_{}_every_{}".format(sampling_type, cob_range, n)
                print("Starting: ", env_name)
                config = TeleportationTrainingConfig(
                    input_shape=(3, 32, 32),
                    device='cuda',
                    cob_range=cob_range,
                    cob_sampling=sampling_type,
                    teleport_every_n_epochs=n,
                    epochs=20,
                    vis_logger=VisdomLogger(env=env_name)
                )
                run_single_output_training(train, get_cifar10_models(device='cuda'), config, metrics,
                                           cifar10_train, cifar10_test, val_set=cifar10_val)
