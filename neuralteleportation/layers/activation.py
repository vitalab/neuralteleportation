import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin, COBForwardMixin


class ActivationLayerMixin(COBForwardMixin, NeuralTeleportationLayerMixin):
    cob_field = 'cob'

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.cob = prev_cob

    def _forward(self, input: torch.Tensor) -> torch.Tensor:

        activation_input = input / self.cob
        activation_output = self.base_layer().forward(self, activation_input)
        output = self.cob * activation_output
        # print("cob activation input:", input)
        # print("cob activation cob: ", self.cob)
        # print("activation input:", activation_input)
        # print("activation output:", activation_output)
        # print("cob activation output: ", output)
        return output

        # return self.cob * self.base_layer().forward(self, input / self.cob)




class ReLUCOB(ActivationLayerMixin, nn.ReLU):
    reshape_cob = True


class TanhCOB(ActivationLayerMixin, nn.Tanh):
    reshape_cob = True


class SigmoidCOB(ActivationLayerMixin, nn.Sigmoid):
    reshape_cob = True


class IdentityCOB(ActivationLayerMixin, nn.Identity):
    reshape_cob = True
