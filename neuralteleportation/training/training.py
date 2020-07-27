from collections import defaultdict
from typing import Sequence, Callable, Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neuralteleportation.training.config import TrainingConfig, TrainingMetrics, TeleportationTrainingConfig
from neuralteleportation.training.experiment_setup import get_optimizer_from_model_and_config


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None) -> nn.Module:
    if optimizer is None:
        optimizer = get_optimizer_from_model_and_config(model, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle_batches)

    for epoch in range(config.epochs):
        if (isinstance(config, TeleportationTrainingConfig)
                and (epoch % config.every_n_epochs) == 0
                and epoch > 0):
            model = config.teleport_fn(model=model, train_dataset=train_dataset, metrics=metrics, config=config)
            # Force a new optimizer in case the model was swapped as a result of the teleportations
            optimizer = get_optimizer_from_model_and_config(model, config)

        train_epoch(model, metrics, optimizer, train_loader, epoch, device=config.device, config=config)

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

    return model


def train_epoch(model: nn.Module, metrics: TrainingMetrics, optimizer: Optimizer, train_loader: DataLoader, epoch: int,
                device: str = 'cpu', progress_bar: bool = True, config: TrainingConfig = None) -> None:
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = metrics.criterion(output, target)
        evaluated_metrics = {metric.__name__: metric(output, target) for metric in metrics.metrics}
        evaluated_metrics["loss"] = loss.item()
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
        if (not config.log_every_n_batch) or batch_idx % config.log_every_n_batch == 0:
            if config.comet_logger:
                config.comet_logger.log_metrics(evaluated_metrics)
            if config.exp_logger:
                if config.exp_logger:
                    for metric_name, value in evaluated_metrics.items():
                        config.exp_logger.add_scalar(f"train_{metric_name}", value, step)
    pbar.update()
    pbar.close()


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
