from functools import reduce
from operator import mul

import torch
from torch import nn

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB


class MLPCOB(nn.Module):
    """Multi-layer Perceptron model class.

    Args:
        input_shape (list of 3 ints) - the channels, height and width of the input
        hidden_layers (list of ints) - the number of neurons in each hidden layer
        num_classes (int) - number of classification classes

    """

    def __init__(self, input_shape=(1, 28, 28), hidden_layers=(128,), num_classes=10):
        super().__init__()

        # Create the input and hidden layers
        layers_dim = ((reduce(mul, input_shape),) + tuple(hidden_layers))
        layers = []
        for idx in range(len(hidden_layers)):
            layers.append(LinearCOB(layers_dim[idx], layers_dim[idx + 1]))
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

    mlp = MLPCOB()
    test_teleport(mlp, (1, 1, 28, 28), verbose=True)
