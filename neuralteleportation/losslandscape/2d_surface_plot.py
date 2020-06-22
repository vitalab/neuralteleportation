from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.training import test, train
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics


def generate_random_2d_vector(weights: torch.Tensor) -> torch.Tensor:
    """
        Generates a random vector of size equals to the weights of the model.
    """
    direction = torch.randn(weights.size())
    normalize_direction(direction, weights)
    return direction


def normalize_direction(direction: torch.Tensor, weights: torch.Tensor):
    """
        Apply a filter normalization to the direction vector.
        d <- d/||d|| * ||w||

        This process is a inplace operation.
    """
    assert len(direction) == len(weights), "Direction must have the same size as model weights"
    w = weights.detach().cpu()
    direction.mul_(w) / direction.norm()
    # for d, w in zip(direction, weights):
    #     d.mul_(w.norm()) / d.norm()


def generate_contour(model: NeuralTeleportationModel, directions: torch.Tensor,
                     surface: torch.Tensor, trainset: Dataset, metric: TrainingMetrics,
                     config: TrainingConfig) -> torch.Tensor:
    """
        Generate a tensor containing the loss values from a given model.
    """
    w = model.get_weights()
    loss = []
    acc = []
    delta, eta = directions
    for _, x in enumerate(surface[0]):
        for _, y in enumerate(surface[1]):
            print("Evaluating [{:.3f}, {:.3f}]".format(x.item(), y.item()))
            x, y = x.to(config.device), y.to(config.device)
            # L (w + alpha*delta + beta*eta)
            model.set_weights(w + (delta * x + eta * y).to(config.device))
            results = test(model, trainset, metric, config)
            loss.append(results['loss'])
            # acc.append(results['accuracy'])

    return loss, acc


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data.dataloader import DataLoader
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_cifar10_models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = TrainingConfig(
        lr=1e-3,
        epochs=2,
        batch_size=32,
        device=device
    )

    models = get_cifar10_models()
    trainset, valset, testset = get_cifar10_datasets()
    # trainset.data = trainset.data[:config.batch_size*3]
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

    for m in models:
        model = NeuralTeleportationModel(m, input_shape=(config.batch_size, 3, 32, 32)).to(device)
        w1 = model.get_weights().detach().cpu()

        train(model, trainset, metric, config)
        model.random_teleport()
        config.epochs = 3
        w2 = model.get_weights().detach().cpu()
        train(model, trainset, metric, config)

        w3 = model.get_weights().detach().cpu()
        delta, eta = generate_random_2d_vector(w3), generate_random_2d_vector(w3)

        # calculate angle between the two direction vectors.
        angle = torch.acos(torch.dot(delta, eta) / (delta.norm() * eta.norm()).item())
        angle = np.degrees(angle.detach().cpu().numpy())
        print("angle between direction is {}".format(angle))

        x = torch.linspace(-1, 1, 10)
        y = torch.linspace(-1, 1, 10)
        shape = x.shape if y is None else (len(x), len(y))
        surface = torch.stack((x, y))

        loss, _ = generate_contour(model, [delta, eta], surface, trainset, metric, config)
        loss = np.array(loss)
        loss = np.resize(loss, shape)

        plt.figure()
        plt.contourf(x, y, loss)
    plt.show()
