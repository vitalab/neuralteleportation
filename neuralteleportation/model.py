import numpy as np
import torch
import torch.nn as nn
from neuralteleportation.layers import LinearCOB, Conv2DCOB, Flatten
from neuralteleportation.layer_utils import patch_module


class NeuralTeleportationModel(nn.Module):
    SUPPORTED_LAYERS = [LinearCOB, Conv2DCOB]

    def __init__(self, input_dim=784, hidden=(500,), output_dim=10, output_activation=None, use_bias=False):
        """

        Args:
            input_dim: int, number of input neurons
            hidden: sequence, tuple of number of hidden neurons
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
        # self.net.add_module('output_layer', LinearCOB(self.layers[-2], self.layers[-1], bias=use_bias, output=True))
        if output_activation:
            self.net.add_module('output_activation', output_activation)

    def forward(self, x):
        return self.net(x)

    def is_layer_supported(self, layer):
        return type(layer) in self.SUPPORTED_LAYERS

    def get_change_of_basis(self, basis_range=10):
        """
          Returns list of np.arrays of change of basis per neuron
        """
        # cob.append(np.ones(self.layers[0]))
        # for i in range(1, len(self.layers) - 1):
        #     cob.append(get_random_cob(basis_range, size=self.layers[i]))
        # cob.append(np.ones(self.layers[-1]))

        cob = []
        layers = self.get_supported_layers()
        cob.append(layers[0].get_input_cob())
        for i in range(len(layers) - 1):  # Hidden layers
            cob.append(layers[i].get_cob(range=basis_range))
        cob.append(layers[-1].get_output_cob())

        return cob

    def get_supported_layers(self):
        return [l for l in self.net.children() if self.is_layer_supported(l)]

    def apply_change_of_basis(self, cob_range=10, device='cpu'):
        """
          Applies change of basis to each of the linear layer weights
        """
        cob = self.get_change_of_basis(cob_range)

        for k, layer in enumerate(self.get_supported_layers()):
            layer.apply_cob(prev_cob=cob[k], next_cob=cob[k + 1])

    def reset_weights(self):
        """Reset all layers."""

        def reset(m):
            for m in self.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.apply(reset)

    def get_weights(self, flat: bool = True):
        """
            Return model weights
        Args:
            flat: bool, if true weights are returned as flatten torch tensor

        Returns:
            torch.Tensor or list containing model weights

        """
        w = []

        for k, layer in enumerate(self.get_supported_layers()):
            w.append(layer.weight.flatten())

            if self.use_bias:
                w.append(layer.bias.flatten())

        if flat:
            return torch.cat(w)
        else:
            return w

    def set_weights(self, weights):
        counter = 0

        for k, layer in enumerate(self.get_supported_layers()):
            shape = layer.weight.shape
            nb_params = np.prod(shape)
            w = torch.tensor(weights[counter:counter + nb_params].reshape(shape))
            layer.weight = torch.nn.Parameter(w, requires_grad=True)
            counter += nb_params

            if self.use_bias:
                shape = layer.bias.shape
                nb_params = np.prod(shape)
                b = torch.tensor(weights[counter:counter + nb_params].reshape(shape))
                layer.bias = torch.nn.Parameter(b, requires_grad=True)
                counter += nb_params

    def get_grad(self, data, target, loss_fn, flat=True):
        grad = []
        output = self.net(data)
        loss = loss_fn(output, target)
        loss.backward()

        for k, layer in enumerate(self.get_supported_layers()):
            grad.append(layer.weight.grad.flatten())

            if self.use_bias:
                grad.append(layer.bias.grad.flatten())

        if flat:
            return torch.cat(grad)
        else:
            return grad

    # def apply_change_of_basis(self, cob_range=10, device='cpu'):
    #     layers = self.get_supported_layers()
    #
    #     cob = layers[0].get_input_cob()
    #     for layer in layers:
    #         cob = layer.apply_cob(prev_cob=cob)

    # def get_change_of_basis(self, basis_range=10):
    #     """
    #       Returns list of np.arrays of change of basis per neuron
    #     """
    #     cob = []
    #
    #     layers = self.get_supported_layers()
    #     cob.append(layers[0].get_input_cob())
    #     for layer in layers:
    #         cob.append(layer.generate_cob())
    #
    #     return cob


if __name__ == '__main__':
    model = NeuralTeleportationModel(4, (10,), 4)
    # print(model)
    # x = torch.rand(size=(1,4))
    # pred1 = model(x).detach().numpy()
    # w1 = model.get_weights().detach().numpy()
    #
    # print(w1)
    #
    # cob = model.get_change_of_basis()
    # print(cob)
    #
    # model.apply_change_of_basis()
    #
    # w2 = model.get_weights().detach().numpy()
    # print(w2)
    #
    # pred2 = model(x).detach().numpy()
    #
    # print(pred1)
    # print(pred2)

    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5, 2),
    )

    cob_module = patch_module(test_module)

    print(test_module)
    print(cob_module)
