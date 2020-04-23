import torch.nn as nn
import torch
from neuralteleportation.utils import get_random_cob
import numpy as np


class NeuralTeleportationLayerMixin(object):
    def apply_cob(self, prev_cob, next_cob):
        raise NotImplemented


class Flatten(torch.nn.Module, NeuralTeleportationLayerMixin):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

    def apply_cob(self, prev_cob, next_cob):
        pass


class NeuronLayerMixin(NeuralTeleportationLayerMixin):
    def get_cob(self, basis_range=10):
        raise NotImplementedError

    def get_output_cob(self):
        raise NotImplementedError

    def get_input_cob(self):
        raise NotImplementedError

    def get_nb_params(self):
        nb_params = np.prod(self.weight.shape)
        if self.bias is not None:
            nb_params += np.prod(self.bias.shape)

        return nb_params

    def get_weights(self):
        # Maybe move this to support more layers like dropout, Batchnorm...
        if self.bias is not None:
            return self.weight.flatten(), self.bias.flatten()
        else:
            return self.weight.flatten(),

    def set_weights(self, weights):
        counter = 0
        w_shape = self.weight.shape
        w_nb_params = np.prod(w_shape)
        w = torch.tensor(weights[counter:counter + w_nb_params].reshape(w_shape))
        self.weight = torch.nn.Parameter(w, requires_grad=True)
        counter += w_nb_params

        if self.bias is not None:
            b_shape = self.bias.shape
            b_nb_params = np.prod(b_shape)
            b = torch.tensor(weights[counter:counter + b_nb_params].reshape(b_shape))
            self.bias = torch.nn.Parameter(b, requires_grad=True)


# add/concat
class MergeLayersMixin(NeuralTeleportationLayerMixin):
    pass


class ActivationLayerMixin(NeuralTeleportationLayerMixin):
    def __init__(self):
        self.cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = prev_cob

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.cob:
            self.cob = torch.zeros(input.shape[1])
        return self.cob * super().forward(input / self.cob)
