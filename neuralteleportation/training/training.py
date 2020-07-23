from collections import defaultdict
from typing import Sequence, Callable, Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neuralteleportation.training.config import TrainingConfig, TrainingMetrics, TeleportationTrainingConfig
from neuralteleportation.training.experiment_setup import (
    get_optimizer_from_model_and_config,
    get_lr_scheduler_from_optimizer_and_config,
)
from neuralteleportation.utils.optimtools import get_optimizer_lr, update_optimizer_params


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None, lr_scheduler=None) -> nn.Module:
    if optimizer is None:
        optimizer = get_optimizer_from_model_and_config(model, config)

    lr_scheduler_interval = None
    if config.lr_scheduler is not None:
        lr_scheduler_interval = config.lr_scheduler[1]

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_batches, drop_last=config.drop_last_batch)

    for epoch in range(config.epochs):
        if (isinstance(config, TeleportationTrainingConfig)
                and (epoch % config.every_n_epochs) == 0
                and epoch > 0):
            model = config.teleport_fn(model, train_dataset, metrics, config)
            # Force a new optimizer in case the model was swapped as a result of the teleportations
            # We need to recreate the optimizer with the new model's parameters and update it
            # with the previous optimizer's parameters otherwise any changes to the old optimizer will be lost
            old_optimizer_state = optimizer.state_dict()
            optimizer = get_optimizer_from_model_and_config(model, config)
            if lr_scheduler:
                # Similar to the optimizer, the lr scheduler needs to be updated after its recreation.
                old_scheduler_state = lr_scheduler.state_dict()
                lr_scheduler = get_lr_scheduler_from_optimizer_and_config(optimizer, config)
                lr_scheduler.load_state_dict(old_scheduler_state)
            # update the optimizer, because for certain LrSchedulers, when they are recreated,
            # they overwrite the previous parameters set in the optimizer (c.f OneCycleLR)
            optimizer = update_optimizer_params(optimizer, old_optimizer_state)
        if lr_scheduler:
            print("Current LR: ", get_optimizer_lr(optimizer))
        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch,
                    device=config.device, config=config, lr_scheduler=lr_scheduler)

        if val_dataset:
            if config.comet_logger:
                with config.comet_logger.validate():
                    val_res = test(model, val_dataset, metrics, config)
            else:
                val_res = test(model, val_dataset, metrics, config)

            print("Validation: {}".format(val_res))
            if np.isnan(val_res["loss"]) or np.isnan(val_res["accuracy"]):
                print("Stopping: Loss NaN!")
                if config.exp_logger:
                    config.exp_logger.add_text(
                        "Info", "Stopped due to Loss NaN.")
                break
            if config.exp_logger is not None:
                config.exp_logger.add_scalar(
                    "val_loss", val_res["loss"], epoch)
                config.exp_logger.add_scalar(
                    "val_accuracy", val_res["accuracy"], epoch)
        if lr_scheduler and lr_scheduler_interval == "epoch":
            lr_scheduler.step()

    return model


def train_epoch(model: nn.Module, criterion: _Loss, optimizer: Optimizer, train_loader: DataLoader, epoch: int,
                device: str = 'cpu', progress_bar: bool = True, config: TrainingConfig = None, lr_scheduler=None) -> None:
    lr_scheduler_interval = None
    if config.lr_scheduler is not None:
        lr_scheduler_interval = config.lr_scheduler[1]
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if progress_bar:
            output = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                              (batch_idx + 1) *
                                                                              train_loader.batch_size,
                                                                              len(train_loader.dataset),
                                                                              100. * batch_idx /
                                                                              len(train_loader),
                                                                              loss.item())
            pbar.set_postfix_str(output)
        step = (len(train_loader.dataset) * epoch) + batch_idx * len(data)
        if config.comet_logger:  # TODO Add ``batch_idx % 500 == 0`` in case we make too many calls to the Comet API
            config.comet_logger.log_metric("loss", loss.item())
        if batch_idx % 500 == 0 and config.exp_logger:
            if config.exp_logger:
                config.exp_logger.add_scalar("train_loss", loss.item(), step)
        if lr_scheduler and lr_scheduler_interval == "step":
            lr_scheduler.step()
    pbar.update()
    pbar.close()


def test(model: nn.Module, dataset: Dataset, metrics: TrainingMetrics, config: TrainingConfig) -> Dict[str, Any]:
    test_loader = DataLoader(dataset, batch_size=config.batch_size)
    model.eval()
    results = defaultdict(list)
    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for i, (data, target) in pbar:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            results['loss'].append(metrics.criterion(output, target).item())

            if metrics is not None:
                batch_results = compute_metrics(
                    metrics.metrics, y=target, y_hat=output, to_tensor=False)
                for k in batch_results.keys():
                    results[k].append(batch_results[k])

            pbar.update()
            pbar.set_postfix(loss=pd.DataFrame(results['loss']).mean().values,
                             accuracy=pd.DataFrame(results['accuracy']).mean().values)

    pbar.close()
    reduced_results = dict(pd.DataFrame(results).mean())
    if config.comet_logger:
        config.comet_logger.log_metrics(reduced_results)
    return reduced_results


def compute_metrics(metrics: Sequence[Callable[[Tensor, Tensor], Tensor]], y_hat: Tensor, y: Tensor,
                    prefix: str = '', to_tensor: bool = True) -> Dict[str, Any]:
    results = {}
    for metric in metrics:
        m = metric(y_hat, y)
        if to_tensor:
            m = torch.tensor(m)
        results[prefix + metric.__name__] = m
    return results


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    from torch.nn.modules import Flatten
    import torch.nn as nn

    mnist_train = MNIST('/tmp', train=True, download=True,
                        transform=transforms.ToTensor())
    mnist_val = MNIST('/tmp', train=False, download=True,
                      transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True,
                       transform=transforms.ToTensor())

    model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    config = TrainingConfig()
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    train(model, train_dataset=mnist_train, metrics=metrics,
          config=config, val_dataset=mnist_val)
    print(test(model, mnist_test, metrics, config))
