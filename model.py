import numpy as np
import torch
import torch.nn as nn
from layers import LinearCOB, Flatten


def get_random_cob(range, size):
    samples = np.random.randint(0, 2, size=size)
    cob = np.zeros_like(samples, dtype=np.float)
    cob[samples == 1] = np.random.uniform(low=1 / range, high=1, size=samples.sum())
    cob[samples == 0] = np.random.uniform(low=1, high=range, size=(len(samples) - samples.sum()))

    return cob


class NeuralTeleportationModel(nn.Module):
    SUPPORTED_LAYERS = [LinearCOB]

    def __init__(self, input_dim=784, hidden=(500,), output_dim=10, output_activation=None, use_bias=False):
        """

        Args:
            input_dim: int, number of input neurons
            hidden: squenece, tuple of number of hidden neurons
            output_dim: int, number of output neurons
            output_activation: nn.Module, activation function
        """

        super(NeuralTeleportationModel, self).__init__()

        self.use_bias = use_bias

        self.layers = [input_dim] + list(hidden) + [output_dim]
        self.net = nn.Sequential()

        # Input layer
        self.net.add_module('flatten_layer', Flatten())
        self.net.add_module('input_layer', LinearCOB(self.layers[0], self.layers[1], bias=use_bias))
        self.net.add_module('relu_{}'.format(0), nn.ReLU())

        # Hidden layers
        for i in range(1, len(self.layers) - 2):
            self.net.add_module('layer_{}'.format(i), LinearCOB(self.layers[i], self.layers[i + 1], bias=use_bias))
            self.net.add_module('relu_{}'.format(i), nn.ReLU())

        # Output layers
        self.net.add_module('output_layer', LinearCOB(self.layers[-2], self.layers[-1], bias=use_bias))
        if output_activation:
            self.net.add_module('output_activation', output_activation)

    def is_layer_supported(self, layer):
        return type(layer) in self.SUPPORTED_LAYERS

    def forward(self, x):
        return self.net(x)

    def get_change_of_basis(self, basis_range=10):
        """
          Returns list of np.arrays of change of basis per neuron
        """
        cob = []
        # xi=random()
        # cob.append(np.ones(self.layers[0])*xi)
        cob.append(np.ones(self.layers[0]))
        for i in range(1, len(self.layers) - 1):
            cob.append(get_random_cob(basis_range, size=self.layers[i]))
        # cob.append(np.ones(self.layers[-1])*xi)
        cob.append(np.ones(self.layers[-1]))
        return cob

    def get_supported_layers(self):
        return [l for l in self.net.children() if self.is_layer_supported(l)]

    def apply_change_of_basis(self, cob_range=10, device='cpu'):
        """
          Applies change of basis to each of the linear layer weights
        """
        cob = self.get_change_of_basis(cob_range)
        linear_layers = [l for l in self.net.children() if self.is_layer_supported(l)]
        # for k, layer in enumerate(linear_layers):
        #     w = cob[k + 1][..., None] / cob[k][None, ...]
        #
        #     layer.weight = torch.nn.Parameter(layer.weight * torch.Tensor(w, device=device), requires_grad=True)
        #     # divide by xi for bias
        #     if layer.bias:
        #         layer.bias = torch.nn.Parameter(layer.bias * torch.tensor(cob[k + 1], dtype=torch.float32),
        #                                         requires_grad=True)

        for k, layer in enumerate(self.get_supported_layers()):
            layer.apply_cob(prev_cob=cob[k], next_cob=cob[k + 1])

    def reset_weights(self):
        """Reset all *resetable* layers."""

        def reset(m):
            for m in self.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.apply(reset)

    def get_weights(self, flat=False):
        w = []
        # linear_layers = [l for l in self.net.children() if type(l) == nn.Linear]

        for k, layer in enumerate(self.get_supported_layers()):
            # print(layer.weight)
            w.append(layer.weight.flatten())

            #TODO fix set weights for bias before uncommenting this

            # if self.use_bias:
            #     grad.append(layer.bias.flatten())

        if flat:
            return torch.cat(w)
        else:
            return w

    def set_weights(self, weights):
        counter = 0
        # linear_layers = [l for l in self.net.children() if type(l) == nn.Linear]

        for k, layer in enumerate(self.get_supported_layers()):
            shape = layer.weight.shape
            w = weights[counter:counter + np.prod(shape)].reshape(shape)
            counter += np.prod(shape)
            layer.weight = torch.nn.Parameter(w, requires_grad=True)

    def get_grad(self, data, target, loss_fn, flat=False):
        grad = []
        output = self.net(data)
        loss = loss_fn(output, target)
        loss.backward()

        # linear_layers = [l for l in self.net.children() if type(l) == nn.Linear]

        for k, layer in enumerate(self.get_supported_layers()):
            grad.append(layer.weight.grad.flatten())

            # if self.use_bias:
            #     grad.append(layer.bias.grad.flatten())

        if flat:
            return torch.cat(grad)
        else:
            return grad


if __name__ == '__main__':
    model = NeuralTeleportationModel(4, (10,), 4)
    print(model)
    x = torch.Tensor([[[2, 2], [4, 5]]])
    print(model(x))
    print(model.get_weights())

    model.apply_change_of_basis()
    print(model(x))
    print(model.get_weights())

    cob = get_random_cob(10, 12)
