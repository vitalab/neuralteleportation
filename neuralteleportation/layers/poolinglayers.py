import torch.nn as nn
import torch

from neuralteleportation.layers.neuralteleportationlayers import NeuralTeleportationLayerMixin
from neuralteleportation.utils import get_random_cob
import numpy as np
from torch.nn.modules import Flatten

class MaxPool2dCOB(nn.MaxPool2d, NeuralTeleportationLayerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cob is None:
            self.cob = torch.ones(input.shape[1])

        cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.cob = self.cob.view(cob_shape).float().type_as(input)

        return self.cob * super().forward(input / self.cob)


class AdaptiveAvgPool2dCOB(nn.AdaptiveAvgPool2d, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.cob = None
    #
    # def apply_cob(self, prev_cob, next_cob):
    #     self.cob = torch.tensor(prev_cob)
    #
    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     if self.cob is None:
    #         self.cob = torch.ones(input.shape[1])
    #
    #     cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
    #     self.cob = self.cob.view(cob_shape).float().type_as(input)
    #
    #     return self.cob * super().forward(input / self.cob)


class AvgPool2dCOB(nn.AvgPool2d, NeuralTeleportationLayerMixin):
    def apply_cob(self, prev_cob, next_cob):
        pass

class UpsampleCOB(nn.Upsample, NeuralTeleportationLayerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cob is None:
            self.cob = torch.ones(input.shape[1])

        cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.cob = self.cob.view(cob_shape).float()

        return self.cob * super().forward(input / self.cob)