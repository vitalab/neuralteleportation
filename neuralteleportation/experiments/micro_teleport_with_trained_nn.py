from collections import defaultdict
from typing import Sequence, Callable

import pandas as pd
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from models.model_zoo import vggcob, resnetcob
from tqdm import tqdm

from neuralteleportation.training.config import TrainingConfig, TrainingMetrics


def train(model: nn.Module, train_dataset: Dataset, metrics: TrainingMetrics, config: TrainingConfig,
          val_dataset: Dataset = None, optimizer: Optimizer = None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    for epoch in range(1, config.epochs + 1):
        train_epoch(model, metrics.criterion, optimizer, train_loader, epoch, device=config.device)
        if val_dataset:
            val_res = test(model, val_dataset, metrics, config)
            print("Validation: {}".format(val_res))


def train_epoch(model: nn.Module, criterion: _Loss, optimizer: Optimizer, train_loader: DataLoader, epoch: int,
                device: str = 'cpu', progress_bar: bool = True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if progress_bar and batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model: nn.Module, dataset: Dataset, metrics: TrainingMetrics, config: TrainingConfig):
    test_loader = DataLoader(dataset, batch_size=config.batch_size)
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            results['loss'].append(metrics.criterion(output, target).item())

            if metrics is not None:
                batch_results = compute_metrics(metrics.metrics, y=target, y_hat=output, to_tensor=False)
                for k in batch_results.keys():
                    results[k].append(batch_results[k])

    results = pd.DataFrame(results)
    return dict(results.mean())


def compute_metrics(metrics: Sequence[Callable[[Tensor, Tensor], Tensor]], y_hat: Tensor, y: Tensor,
                    prefix: str = '', to_tensor: bool = True):
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
    # from torch.nn.modules import Flatten
    import torch.nn as nn

    trans = []
    trans.append(transforms.Resize(size=224))
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    mnist_train = MNIST('/tmp', train=True, download=True, transform=trans)
    mnist_val = MNIST('/tmp', train=False, download=True, transform=trans)
    mnist_test = MNIST('/tmp', train=False, download=True, transform=trans)

    # model = torch.nn.Sequential(
    #     Flatten(),
    #     nn.Linear(784, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 10)
    # )

    model = vggcob.vgg11COB(input_channels=1);

    config = TrainingConfig()
    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    train(model, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_val)
    print(test(model, mnist_test, metrics, config))