import argparse
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, BatchNorm2dCOB, LinearCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB, UpsampleCOB

from neuralteleportation.training.training import test, train
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_cifar10_models
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.metrics import accuracy


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=2)

    return parser.parse_args()


def test_model_with(model: nn.Module, testset: Dataset,
                    metric: TrainingMetrics, config: TrainingConfig,
                    rept: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    loss_avg = []
    acc_avg = []
    for _ in range(rept):
        m = NeuralTeleportationModel(model, input_shape=(config.batch_size, 3, 32, 32)).to(device)
        w_o, cob_o = m.get_params()
        m.random_teleport()
        w_t, cob_t = m.get_params()

        m.set_params(weights=w_o, cob=cob_o)
        res = test(m, testset, metric, config)
        loss1, acc1 = res['loss'], res['accuracy']

        m.set_params(weights=w_t, cob=cob_t)
        res = test(m, testset, metric, config)
        loss2, acc2 = res['loss'], res['accuracy']

        loss_avg.append(np.abs(loss1 - loss2))
        acc_avg.append(np.abs(acc1 - acc2))

        print("==========================================")
        print("Loss and accuracy diff with set/get was")
        print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
        print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
        print("==========================================")

    return np.mean(loss_avg), np.mean(acc_avg)


def test_model_without(model: nn.Module, testset: Dataset,
                       metric: TrainingMetrics, config: TrainingConfig,
                       rept: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    loss_avg = []
    acc_avg = []
    for _ in range(rept):
        m = NeuralTeleportationModel(model, input_shape=(config.batch_size, 3, 32, 32)).to(device)

        res = test(m, testset, metric, config)
        loss1, acc1 = res['loss'], res['accuracy']

        m.random_teleport()

        res = test(m, testset, metric, config)
        loss2, acc2 = res['loss'], res['accuracy']

        loss_avg.append(np.abs(loss1 - loss2))
        acc_avg.append(np.abs(acc1 - acc2))

        print("==========================================")
        print("Loss and accuracy diff without set/get was")
        print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
        print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
        print("==========================================")

    return np.mean(loss_avg), np.mean(acc_avg)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, _, testset = get_cifar10_datasets()

    args = argument_parser()

    metric = TrainingMetrics(
        criterion=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = TrainingConfig(
        batch_size=32,
        epochs=1,
        device=device
    )

    model = get_cifar10_models()[0]

    print("==========================================")
    print()
    print("==========================================")
    print("===========Testing w/o set/get============")
    print("==========================================")
    print()
    print("==========================================")

    loss, acc = test_model_without(model, testset, metric, config, args.t)

    print("==========================================")
    print("Loss and accuracy avg diff without set/get was")
    print("Loss diff was: {:.6e}".format(np.mean(loss)))
    print("Acc diff was: {:.6e}".format(np.mean(acc)))
    print("==========================================")

    print()
    print("==========================================")
    print("===========Testing with set/get===========")
    print("==========================================")
    print()
    print("==========================================")

    loss, acc = test_model_with(model, testset, metric, config, args.t)

    print("==========================================")
    print("Loss and accuracy avg diff with set/get was")
    print("Loss diff was: {:.6e}".format(np.mean(loss)))
    print("Acc diff was: {:.6e}".format(np.mean(acc)))
    print("==========================================")
