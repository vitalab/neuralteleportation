import torch.nn as nn
import torch
from neuralteleportation.utils import get_random_cob
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


# class AbstractCOBLayer:
#
#     def apply_cob(self, prev_cob):
#         raise NotImplemented
#
#     def get_cob(self):
#         raise NotImplementedError
#
#     def generate_cob(self, range=10, output=False):
#         raise NotImplementedError
#
#     def set_cob(self, cob):
#         raise NotImplementedError
#
#     def get_input_cob(self):
#         raise NotImplementedError
#
#
# class LinearCOB(nn.Linear, AbstractCOBLayer):
#
#     def __init__(self, in_features: int, out_features: int, bias=True, output=False):
#         super().__init__(in_features, out_features, bias)
#         self.is_output = output
#         self.generate_cob(output=output)
#
#     def apply_cob(self, prev_cob):
#         w = torch.tensor(self.cob[..., None] / prev_cob[None, ...], dtype=torch.float32)
#         self.weight = nn.Parameter(self.weight * w, requires_grad=True)
#
#         if self.bias is not None:
#             self.bias = torch.nn.Parameter(self.bias * torch.tensor(self.cob, dtype=torch.float32), requires_grad=True)
#
#         return self.cob
#
#     def generate_cob(self, range=10, output=False):
#         """
#
#         Returns:
#             cob for the output neurons
#         """
#         if output or self.is_output:
#             self.cob = np.ones(shape=self.out_features)
#         else:
#             self.cob = get_random_cob(range=range, size=self.out_features)
#
#         return self.cob
#
#     def get_input_cob(self):
#         """
#         Get cob if input layer
#         """
#         return np.ones(shape=self.in_features)
#
#     def get_cob(self):
#         return self.cob
#
#     def set_cob(self, cob):
#         # TODO check size
#         self.cob = cob
#
#
# class Conv2DCOB(nn.Conv2d):
#
#     def apply_cob(self, prev_cob, next_cob):
#         shape = self.weight.shape
#         for i in range(shape[0]):
#             if self.bias is not None:
#                 self.bias[i] *= next_cob[i]
#             for j in range(shape[1]):
#                 self.weight[i, j] *= next_cob[i] / prev_cob[j]
#
#     def get_cob(self, range, input=False):
#         """
#
#         Returns:
#             cob for the output feature maps
#         """
#         if input:
#             return get_random_cob(range=range, size=self.in_channels)
#         return get_random_cob(range=range, size=self.out_channels)


class AbstractCOBLayer:

    def apply_cob(self, prev_cob, next_cob):
        raise NotImplemented

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


class LinearCOB(nn.Linear, AbstractCOBLayer):

    def apply_cob(self, prev_cob, next_cob):

        if len(prev_cob) != self.in_features:  # if previous layer is Conv2D
            hw = self.in_features // len(prev_cob)  # size of feature maps
            cob = []
            for i, c in enumerate(prev_cob):
                cob.extend(np.repeat(c, repeats=hw).tolist())
            prev_cob = np.array(cob)

        w = torch.tensor(next_cob[..., None] / prev_cob[None, ...], dtype=torch.float32)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias * torch.tensor(next_cob, dtype=torch.float32), requires_grad=True)

    def get_cob(self, basis_range=10):
        """
        Returns:
            cob for the output neurons
        """
        return get_random_cob(range=basis_range, size=self.out_features)

    def get_input_cob(self):
        return np.ones(shape=self.in_features)

    def get_output_cob(self):
        return np.ones(shape=self.out_features)


class Conv2DCOB(nn.Conv2d, AbstractCOBLayer):

    def apply_cob(self, prev_cob, next_cob):
        shape = self.weight.shape
        for i in range(shape[0]):
            if self.bias is not None:
                self.bias[i] *= next_cob[i]
            for j in range(shape[1]):
                self.weight[i, j] *= next_cob[i] / prev_cob[j]

    def get_cob(self, basis_range=10):
        """
        Returns:
            cob for the output neurons
        """
        return get_random_cob(range=basis_range, size=self.out_channels)

    def get_input_cob(self):
        return np.ones(shape=self.in_channels)

    def get_output_cob(self):
        return np.ones(shape=self.out_channels)
