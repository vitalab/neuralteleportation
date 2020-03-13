"""
This file defines the core research contribution
"""
from argparse import ArgumentParser, Namespace
from typing import Callable, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class PlSystem(pl.LightningModule):

    def __init__(self, network: nn.Module, train_dataset: Dataset, test_dataset: Dataset, hparams: Namespace,
                 loss_fn: Callable, val_dataset: Dataset = None, metrics: Sequence[Callable] = None):
        """

        Args:
            network: nn.Module, network to train
            train_dataset: torch.utils.data.Dataset, training dataset
            val_dataset: torch.utils.data.Dataset, validation dataset
            test_dataset: torch.utils.data.Dataset, test dataset
            hparams: Namespace, hyperparams from argparser
            loss_fn: Callable, torch.nn.functional loss function
            metrics: Sequence[Callable], list of metrics to evaluate model
        """
        super(PlSystem, self).__init__()

        self.network = network
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.hparams = hparams

        if metrics:
            self.metric_names = [m.__name__ for m in self.metrics]

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        if self.metrics:
            metrics = self.compute_metrics(y_hat, y)
            progress_bar = metrics
            tensorboard_logs.update(metrics)

        return_dict = {'loss': loss,
                       'log': tensorboard_logs}

        if self.metrics:
            return_dict.update(metrics)
            return_dict.update({'progress_bar': progress_bar})

        return return_dict

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)

        return_dict = {'val_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y, prefix='val_')
            return_dict.update(metrics)
        return return_dict

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss}

        if self.metrics:
            average_metrics = {}
            for name in self.metric_names:
                average_metrics['val_' + name] = torch.stack([x['val_' + name] for x in outputs]).mean()
            logs.update(average_metrics)

        return_dict = {'avg_val_loss': avg_loss,
                       'progress_bar': logs,
                       'log': logs}
        if self.metrics:
            return_dict.update(average_metrics)

        return return_dict

    def testing_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)

        return_dict = {'test_loss': loss}
        if self.metrics:
            metrics = self.compute_metrics(y_hat, y, prefix='test_')
            return_dict.update(metrics)

        return return_dict

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss}

        if self.metrics:
            average_metrics = {}
            for name in self.metric_names:
                average_metrics['test_' + name] = torch.stack([x['test_' + name] for x in outputs]).mean()
            logs.update(average_metrics)

        return_dict = {'avg_test_loss': avg_loss,
                       'log': logs}
        if self.metrics:
            return_dict.update(average_metrics)

        return return_dict

    def compute_metrics(self, y_hat, y, prefix='', to_tensor=True):
        metrics = {}
        for metric in self.metrics:
            m = metric(y_hat, y)
            if to_tensor:
                m = torch.tensor(m)
            metrics[prefix + metric.__name__] = m

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Initial learning rate',
                            dest='lr')
        return parser


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from neuralteleportation.model import NeuralTeleportationModel
    from pytorch_lightning import Trainer
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.layers import Flatten
    from neuralteleportation.layer_utils import patch_module

    args = ArgumentParser(add_help=False)
    args = PlSystem.add_model_specific_args(args)
    params = args.parse_args()

    mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

    cnn_model = torch.nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    network = NeuralTeleportationModel()
    print(network.get_weights())
    model = PlSystem(network=network, train_dataset=mnist_train, val_dataset=mnist_val,
                     test_dataset=mnist_test, hparams=params, loss_fn=nn.CrossEntropyLoss(), metrics=[accuracy])

    model = patch_module(model)

    print(model)

    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=1)
    trainer.fit(model)

    print(network.get_weights())
