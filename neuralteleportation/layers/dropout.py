import numpy as np
import torch
from torch import nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin


class DropoutCOB(NeuralTeleportationLayerMixin, nn.Dropout):

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass


class Dropout2dCOB(NeuralTeleportationLayerMixin, nn.Dropout2d):

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass
