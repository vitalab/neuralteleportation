import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportationlayers import NeuronLayerMixin
from neuralteleportation.changeofbasisutils import get_random_cob


class LinearCOB(nn.Linear, NeuronLayerMixin):

    def apply_cob(self, prev_cob, next_cob):

        if len(prev_cob) != self.in_features:  # if previous layer is Conv2D, duplicate cob for each feature map.
            feature_map_size = self.in_features // len(prev_cob)  # size of feature maps
            cob = []
            for i, c in enumerate(prev_cob):
                cob.extend(np.repeat(c, repeats=feature_map_size).tolist())
            prev_cob = np.array(cob)

        w = torch.tensor(next_cob[..., None] / prev_cob[None, ...], dtype=torch.float32).type_as(self.weight)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias is not None:
            b = torch.tensor(next_cob, dtype=torch.float32).type_as(self.bias)
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

    def get_cob(self, basis_range=10):
        return get_random_cob(range=basis_range, size=self.out_features)

    def get_input_cob(self):
        return np.ones(shape=self.in_features)

    def get_output_cob(self):
        return np.ones(shape=self.out_features)


class Conv2dCOB(nn.Conv2d, NeuronLayerMixin):

    def apply_cob(self, prev_cob, next_cob):
        w = torch.tensor(next_cob[..., None] / prev_cob[None, ...], dtype=torch.float32)[..., None, None]
        self.weight = nn.Parameter(self.weight * w.type_as(self.weight), requires_grad=True)

        if self.bias is not None:
            b = torch.tensor(next_cob, dtype=torch.float32).type_as(self.bias)
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

    def get_cob(self, basis_range=10):
        """
        Returns:
            cob for the output feature maps.
        """
        return get_random_cob(range=basis_range, size=self.out_channels)

    def get_input_cob(self):
        return np.ones(shape=self.in_channels)

    def get_output_cob(self):
        return np.ones(shape=self.out_channels)


class ConvTranspose2dCOB(nn.ConvTranspose2d, NeuronLayerMixin):

    def apply_cob(self, prev_cob, next_cob):
        w = torch.tensor(next_cob[None, ...] / prev_cob[..., None], dtype=torch.float32)[..., None, None]
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias is not None:
            b = torch.tensor(next_cob, dtype=torch.float32)
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

    def get_cob(self, basis_range=10):
        """
        Returns:
            cob for the output feature maps.
        """
        return get_random_cob(range=basis_range, size=self.out_channels)

    def get_input_cob(self):
        return np.ones(shape=self.in_channels)

    def get_output_cob(self):
        return np.ones(shape=self.out_channels)


class BatchNorm2dCOB(nn.BatchNorm2d, NeuronLayerMixin):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.prev_cob = None
        self.next_cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.prev_cob = torch.tensor(prev_cob)
        self.next_cob = torch.tensor(next_cob)

        w = torch.tensor(next_cob, dtype=torch.float32).type_as(self.weight)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias is not None:
            b = torch.tensor(next_cob, dtype=torch.float32).type_as(self.bias)
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

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

        cob1_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.prev_cob = self.prev_cob.view(cob1_shape).float().type_as(input)

        return super().forward(input / self.prev_cob)


class BatchNorm1dCOB(nn.BatchNorm1d, NeuronLayerMixin):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.prev_cob = None
        self.next_cob = None

    def apply_cob(self, prev_cob, next_cob):

        self.prev_cob = torch.tensor(prev_cob)
        self.next_cob = torch.tensor(next_cob)

        w = torch.tensor(next_cob, dtype=torch.float32).type_as(self.weight)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias is not None:
            b = torch.tensor(next_cob, dtype=torch.float32).type_as(self.bias)
            self.bias = torch.nn.Parameter(self.bias * b, requires_grad=True)

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

        return super().forward(input / self.prev_cob)
