import argparse
import torch
import torch.nn as nn
import numpy as np

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.layers.neuron import Conv2dCOB, BatchNorm2dCOB, LinearCOB
from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB

from neuralteleportation.training.training import test, train
from neuralteleportation.training.experiment_setup import get_mnist_datasets
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.metrics import accuracy


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = Conv2dCOB(1, 64, 3)
        self.relu1 = ReLUCOB()
        self.bn1 = BatchNorm2dCOB(64)
        self.flatten = FlattenCOB()
        self.lin1 = LinearCOB(43264, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, _, testset = get_mnist_datasets()

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
    model = NeuralTeleportationModel(Net4(), input_shape=(config.batch_size, 1, 28, 28)).to(device)
    # train(model, trainset, metrics=metric, config=config)
    loss_avg = []
    acc_avg = []

    print("==========================================")
    print("===========Testing w/o set/get============")
    print("==========================================")
    print()
    print("==========================================")
    for _ in range(args.t):
        res = test(model, testset, metric, config)
        loss1, acc1 = res['loss'], res['accuracy']

        model.random_teleport()
        res = test(model, testset, metric, config)
        loss2, acc2 = res['loss'], res['accuracy']

        loss_avg.append(np.abs(loss1 - loss2))
        acc_avg.append(np.abs(acc1 - acc2))

        print("==========================================")
        print("Loss and accuracy diff without set/get was")
        print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
        print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
        print("==========================================")

    print("==========================================")
    print("Loss and accuracy diff without set/get was")
    print("Loss diff was: {:.6e}".format(np.mean(loss_avg)))
    print("Acc diff was: {:.6e}".format(np.mean(acc_avg)))
    print("==========================================")

    print()
    print("==========================================")
    print("===========Testing with set/get===========")
    print("==========================================")
    print()
    print("==========================================")
    loss_avg = []
    acc_avg = []
    for _ in range(args.t):
        model = NeuralTeleportationModel(Net4(), input_shape=(config.batch_size, 1, 28, 28)).to(device)
        w_o = model.get_weights()
        model.random_teleport()
        w_t = model.get_weights()

        model.set_weights(w_o)
        res = test(model, testset, metric, config)
        loss1, acc1 = res['loss'], res['accuracy']

        model.set_weights(w_t)
        res = test(model, testset, metric, config)
        loss2, acc2 = res['loss'], res['accuracy']

        loss_avg.append(np.abs(loss1 - loss2))
        acc_avg.append(np.abs(acc1 - acc2))

        print("==========================================")
        print("Loss and accuracy diff with set/get was")
        print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
        print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
        print("==========================================")

    print("==========================================")
    print("Loss and accuracy diff without set/get was")
    print("Loss diff was: {:.6e}".format(np.mean(loss_avg)))
    print("Acc diff was: {:.6e}".format(np.mean(acc_avg)))
    print("==========================================")