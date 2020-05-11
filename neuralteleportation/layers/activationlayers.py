import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportationlayers import NeuralTeleportationLayerMixin


class ActivationLayerMixin(NeuralTeleportationLayerMixin):
    cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = prev_cob

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.cob:
            self.cob = torch.ones(input.shape[1])
        return self.cob * super().forward(input / self.cob)


class ReLUCOB(nn.ReLU, ActivationLayerMixin):
    cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cob is None:
            self.cob = torch.ones(input.shape[1])

        cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
        self.cob = self.cob.view(cob_shape).float().type_as(input)

        return self.cob * super().forward(input / self.cob)
