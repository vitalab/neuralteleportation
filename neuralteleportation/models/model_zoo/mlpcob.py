from functools import reduce
from operator import mul

import torch
from torch import nn

from neuralteleportation.layers.activation import ReLUCOB, ELUCOB, LeakyReLUCOB, TanhCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB

__all__ = ['MLPCOB']


class MLPCOB(nn.Module):
    """Multi-layer Perceptron model class.

    Args:
        input_shape (list of 3 ints) - the channels, height and width of the input
        num_classes (int) - number of classification classes
        hidden_layers (list of ints) - the number of neurons in each hidden layer

    """

    def __init__(self, input_shape, num_classes, hidden_layers=(500, 500, 500, 500, 500), activation="relu"):
        super().__init__()

        # Create the input and hidden layers
        layers_dim = ((reduce(mul, input_shape),) + tuple(hidden_layers))
        layers = []
        for idx in range(len(hidden_layers)):
            layers.append(LinearCOB(layers_dim[idx], layers_dim[idx + 1]))

            if activation == 'elu':
                layers.append(ELUCOB())
            elif activation == 'leakyrelu':
                layers.append(LeakyReLUCOB())
            elif activation == 'tanh':
                layers.append(TanhCOB())
            else:
                layers.append(ReLUCOB())

        self.net = torch.nn.Sequential(
            FlattenCOB(),
            *layers,
            LinearCOB(hidden_layers[-1], num_classes)
        )

    def forward(self, input):
        return self.net(input)


if __name__ == '__main__':
    from tests.model_test import test_teleport

    mlp = MLPCOB(input_shape=(1, 28, 28), num_classes=10, activation='tanh')
    test_teleport(mlp, (1, 1, 28, 28), verbose=True)
