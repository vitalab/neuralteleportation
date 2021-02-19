from collections import defaultdict
from statistics import mean
from typing import Sequence, Callable, Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neuralteleportation.training.config import TrainingConfig, TrainingMetrics, TeleportationTrainingConfig
from neuralteleportation.training.experiment_setup import (
    get_optimizer_from_model_and_config,
    get_lr_scheduler_from_optimizer_and_config, get_teleportation_epochs,
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
                and epoch in get_teleportation_epochs(config)):
            model = config.teleport_fn(model=model, train_dataset=train_dataset, metrics=metrics, config=config)
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
        train_epoch(model, metrics, optimizer, train_loader, epoch,
                    device=config.device, config=config, lr_scheduler=lr_scheduler)

        if val_dataset:
            if config.logger:
                with config.logger.validate():
                    val_res = test(model, val_dataset, metrics, config)
            else:
                val_res = test(model, val_dataset, metrics, config)

            print("Validation: {}".format(val_res))
            if np.isnan(val_res["loss"]) or np.isnan(val_res["accuracy"]):
                print("Stopping: Loss NaN!")
                if config.logger:
                    config.logger.add_text(
                        "Info", "Stopped due to Loss NaN.")
                break
            if config.logger is not None:
                config.logger.add_scalar(
                    "val_loss", val_res["loss"], epoch)
                config.logger.add_scalar(
                    "val_accuracy", val_res["accuracy"], epoch)
        if lr_scheduler and lr_scheduler_interval == "epoch":
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(metrics=val_res["accuracy"])
            else:
                lr_scheduler.step()

    if config.logger is not None:
        config.logger.flush()

    return model


def train_epoch(model: nn.Module, metrics: TrainingMetrics, optimizer: Optimizer, train_loader: DataLoader, epoch: int,
                device: str = 'cpu', progress_bar: bool = True, config: TrainingConfig = None, lr_scheduler=None) -> None:
    lr_scheduler_interval = None
    if config.lr_scheduler is not None:
        lr_scheduler_interval = config.lr_scheduler[1]
    
    # Init data structures to keep track of the metrics at each batch
    metrics_by_batch = {metric.__name__: [] for metric in metrics.metrics}
    metrics_by_batch.update(loss=[])
    
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        if batch_idx == config.max_batch:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = metrics.criterion(output, target)
        metrics_by_batch["loss"].append(loss.item())
        for metric in metrics.metrics:
            metrics_by_batch[metric.__name__].append(metric(output, target))
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
        if lr_scheduler and lr_scheduler_interval == "step":
            lr_scheduler.step()
    pbar.update()
    pbar.close()

    # Log the mean of each metric at the end of the epoch
    if config is not None and config.logger is not None:
        reduced_metrics = {metric: mean(values_by_batch) for metric, values_by_batch in metrics_by_batch.items()}
        config.logger.log_metrics(reduced_metrics, epoch=epoch)
        for metric_name, value in reduced_metrics.items():
            config.logger.add_scalar(metric_name, value, epoch)


def test(model: nn.Module, dataset: Dataset,
         metrics: TrainingMetrics, config: TrainingConfig,
         eval_mode: bool = True) -> Dict[str, Any]:
    test_loader = DataLoader(dataset, batch_size=config.batch_size)
    if eval_mode:
        model.eval()
    results = defaultdict(list)
    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for i, (data, target) in pbar:
            if i == config.max_batch:
                break
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
    if config.logger is not None:
        config.logger.log_metrics(reduced_results, epoch=0)
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
