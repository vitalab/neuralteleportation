from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import Flatten


class NeuralTeleportationLayerMixin(object):
    def apply_cob(self, prev_cob, next_cob):
        raise NotImplemented


class NeuronLayerMixin(NeuralTeleportationLayerMixin):
    w: torch.Tensor
    b: torch.Tensor

    def get_cob(self, basis_range=10):
        """
        Returns a random change of basis for the output features of the layer.

        Args:
            basis_range (int): range for the change of basis.

        Returns:
            nd.array of change of basis.
        """
        raise NotImplementedError

    def get_output_cob(self):
        """
        Get change of basis for the layer if it is an output layer.

        Returns:
            Ones of size of the output features.
        """
        raise NotImplementedError

    def get_input_cob(self):
        """
        Get change of basis for the layer if it is an input layer.

        Returns:
            Ones of size of the input features.
        """
        raise NotImplementedError

    @property
    def input_cob_size(self):
        raise NotImplementedError

    @property
    def cob_size(self):
        raise NotImplementedError

    def get_nb_params(self):
        """
        Get the number of parameters in the layer (weight and bias).

        Returns:
            int, number of parameters in the layer.
        """
        nb_params = np.prod(self.weight.shape)
        if self.bias is not None:
            nb_params += np.prod(self.bias.shape)

        return nb_params

    def get_weights(self, flatten=True, bias=True) -> Tuple[torch.Tensor, ...]:
        """
         Get the weights from the layer.

        Returns:
            tuple of torch.Tensor

        """
        if self.bias is not None and bias:
            if flatten:
                return self.weight.flatten(), self.bias.flatten()
            else:
                return self.weight, self.bias
        else:
            if flatten:
                return self.weight.flatten(),
            else:
                return self.weight

    def set_weights(self, weights: torch.Tensor):
        """
            Set weights for the layer.

        Args:
            weights (torch.Tensor): weights to apply to the model.
        """
        counter = 0
        w_shape = self.weight.shape
        w_nb_params = np.prod(w_shape)
        w = weights[counter:counter + w_nb_params].reshape(w_shape)
        self.weight = torch.nn.Parameter(w, requires_grad=True)
        counter += w_nb_params

        if self.bias is not None:
            b_shape = self.bias.shape
            b_nb_params = np.prod(b_shape)
            b = weights[counter:counter + b_nb_params].reshape(b_shape)
            self.bias = torch.nn.Parameter(b, requires_grad=True)

    @staticmethod
    def calculate_cob(weights, target_weights, prev_cob, concat=True):
        """
            Compute the cob to teleport from weights to the target_weights
        """
        raise NotImplementedError


class FlattenCOB(Flatten, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass


class DropoutCOB(nn.Dropout, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass


class Dropout2dCOB(nn.Dropout2d, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass
