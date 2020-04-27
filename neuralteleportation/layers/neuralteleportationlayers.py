import torch.nn as nn
import torch
from neuralteleportation.utils import get_random_cob
import numpy as np
from torch.nn.modules import Flatten


class NeuralTeleportationLayerMixin(object):
    def apply_cob(self, prev_cob, next_cob):
        raise NotImplemented


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


class FlattenCOB(Flatten, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass


class ActivationLayerMixin(NeuralTeleportationLayerMixin):
    def __init__(self):
        self.cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = prev_cob

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.cob:
            self.cob = torch.ones(input.shape[1])
        return self.cob * super().forward(input / self.cob)


class BatchNorm2dCOB(nn.BatchNorm2d, NeuronLayerMixin):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.prev_cob = None
        self.next_cob = None

    def apply_cob(self, prev_cob, next_cob):

        self.prev_cob = torch.tensor(prev_cob)
        self.next_cob = torch.tensor(next_cob)

        # print(self.running_mean.shape)
        # print(self.running_var.shape)

        # print(self.weight.shape)
        # print(self.bias.shape)

        w = torch.tensor(next_cob, dtype=torch.float32)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)
        # self.running_var = nn.Parameter(self.running_var * w, requires_grad=False)

        b = torch.tensor(next_cob, dtype=torch.float32)
        # self.running_mean = torch.nn.Parameter(self.running_mean * b, requires_grad=False)

        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

        # print(self.weight.shape)
        # print(self.bias.shape)

    def get_cob(self, basis_range=10):
        """
        Returns:
            cob for the output neurons
        """
        return get_random_cob(range=basis_range, size=self.num_features)

    def get_input_cob(self):
        return np.ones(shape=self.num_features)

    def get_output_cob(self):
        return np.ones(shape=self.num_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.prev_cob is None:
            self.prev_cob = torch.ones(input.shape[1])
        if self.next_cob is None:
            self.next_cob = torch.ones(input.shape[1])

        cob1_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.prev_cob = self.prev_cob.view(cob1_shape).float()

        next_cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.next_cob = self.next_cob.view(next_cob_shape).float()

        return super().forward(input / self.prev_cob)


if __name__ == '__main__':
    layer = BatchNorm2dCOB(5)

    print(layer.get_weights())
    cob = layer.get_cob()
    prev_cob = layer.get_input_cob()
    print(cob)
    print(prev_cob)
    layer.apply_cob(cob, prev_cob)
    print(layer.get_weights())
