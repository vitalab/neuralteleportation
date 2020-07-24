import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin, COBForwardMixin


class MaxPool2dCOB(COBForwardMixin, NeuralTeleportationLayerMixin, nn.MaxPool2d):
    """Wrapper for the MaxPool2d change of basis layer.

    Max pooling is positive scale invariant. It is necessary to un-apply and re-apply the change of basis for the
    operation.
    """
    cob_field = 'cob'
    reshape_cob = True

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.cob = prev_cob

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.cob * self.base_layer().forward(self, input / self.cob)


class AdaptiveAvgPool2dCOB(NeuralTeleportationLayerMixin, nn.AdaptiveAvgPool2d):
    """Wrapper for the AdaptiveAvgPool2d change of basis layer.

    Average pooling is scale invariant, nothing needs to be done to the layer.
    """

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass


class AvgPool2dCOB(NeuralTeleportationLayerMixin, nn.AvgPool2d):
    """Wrapper for the AvgPool2d change of basis layer.

    Average pooling is scale invariant, nothing needs to be done to the layer.
    """

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass


class UpsampleCOB(COBForwardMixin, NeuralTeleportationLayerMixin, nn.Upsample):
    """Wrapper for the Upsample change of basis layer.

    Upsampling is positive scale invariant. It is necessary to un-apply and re-apply the change of basis for the
    opperation.
    """
    cob_field = 'cob'
    reshape_cob = True

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.cob = prev_cob

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.cob * self.base_layer().forward(self, input / self.cob)
