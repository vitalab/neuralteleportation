import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin, COBForwardMixin


class ActivationLayerMixin(COBForwardMixin, NeuralTeleportationLayerMixin):
    cob_field = 'cob'

    def apply_cob(self, prev_cob: np.ndarray, next_cob: np.ndarray):
        self.cob = torch.tensor(prev_cob)

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.cob * self.base_layer().forward(self, input / self.cob)


class ReLUCOB(ActivationLayerMixin, nn.ReLU):
    reshape_cob = True
