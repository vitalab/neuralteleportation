import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportationlayers import NeuralTeleportationLayerMixin


class MaxPool2dCOB(nn.MaxPool2d, NeuralTeleportationLayerMixin):
    """
        Wrapper for the MaxPool2d change of basis layer.
        Max pooling is positive scale invariant. It is necessary to un-apply and re-apply the change of basis for the
        opperation.
    """
    cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cob is None:
            self.cob = torch.ones(input.shape[1])

        cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.cob = self.cob.view(cob_shape).float().type_as(input)

        return self.cob * super().forward(input / self.cob)


class AdaptiveAvgPool2dCOB(nn.AdaptiveAvgPool2d, NeuralTeleportationLayerMixin):
    """
        Wrapper for the AdaptiveAvgPool2d change of basis layer.
        Average pooling is scale invariant, nothing needs to be done to the layer.
    """
    def apply_cob(self, prev_cob, next_cob):
        pass


class AvgPool2dCOB(nn.AvgPool2d, NeuralTeleportationLayerMixin):
    """
        Wrapper for the AvgPool2d change of basis layer.
        Average pooling is scale invariant, nothing needs to be done to the layer.
    """
    def apply_cob(self, prev_cob, next_cob):
        pass


class UpsampleCOB(nn.Upsample, NeuralTeleportationLayerMixin):
    cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cob is None:
            self.cob = torch.ones(input.shape[1])

        cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.cob = self.cob.view(cob_shape).float()

        return self.cob * super().forward(input / self.cob)
