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
        self.conv1 = Conv2dCOB(1, 2, 3)
        self.relu1 = ReLUCOB()
        self.bn1 = BatchNorm2dCOB(2)
        self.flatten = FlattenCOB()
        self.lin1 = LinearCOB(1352, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        return self.lin1(self.flatten(x))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, _, testset = get_mnist_datasets()


    metric = TrainingMetrics(
        criterion=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = TrainingConfig(
        batch_size=8192,
        epochs=1,
        device=device
    )
    model = NeuralTeleportationModel(Net4(), input_shape=(config.batch_size, 1, 28, 28)).to(device)
    # train(model, trainset, metrics=metric, config=config)

    print("==========================================")
    res = test(model, trainset, metric, config)
    loss1, acc1 = res['loss'], res['accuracy']

    model.random_teleport()
    res = test(model, trainset, metric, config)
    loss2, acc2 = res['loss'], res['accuracy']

    print("==========================================")
    print("Loss and accuracy diff without set/get was")
    print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
    print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
    print("==========================================")

    # model = NeuralTeleportationModel(Net4(), input_shape=(config.batch_size, 1, 28, 28)).to(device)
    w_o = model.get_weights()
    model.random_teleport()
    w_t = model.get_weights()

    model.set_weights(w_o)
    res = test(model, trainset, metric, config)
    loss1, acc1 = res['loss'], res['accuracy']

    model.set_weights(w_t)
    res = test(model, trainset, metric, config)
    loss2, acc2 = res['loss'], res['accuracy']

    print("==========================================")
    print("Loss and accuracy diff with set/get was")
    print("Loss diff was: {:.6e}".format(np.abs(loss1 - loss2)))
    print("Acc diff was: {:.6e}".format(np.abs(acc1 - acc2)))
    print("==========================================")
