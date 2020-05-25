import numpy as np
from torch import nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin


class DropoutCOB(NeuralTeleportationLayerMixin, nn.Dropout):

    def apply_cob(self, prev_cob: np.ndarray, next_cob: np.ndarray):
        pass


class Dropout2dCOB(NeuralTeleportationLayerMixin, nn.Dropout2d):

    def apply_cob(self, prev_cob: np.ndarray, next_cob: np.ndarray):
        pass
