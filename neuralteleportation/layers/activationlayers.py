import torch
import torch.nn as nn

from neuralteleportation.layers.layers_v3 import ActivationLayerMixin


class ReLUCOB(nn.ReLU, ActivationLayerMixin):
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
