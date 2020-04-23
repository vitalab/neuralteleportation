import torch.nn as nn
import torch
from neuralteleportation.layers.layers_v3 import MergeLayersMixin


class Concat(nn.Module, MergeLayersMixin):
    def forward(self, input1, input2):
        return torch.cat([input1, input2], dim=1)


class Add(nn.Module, MergeLayersMixin):
    def __init__(self):
        super().__init__()
        self.cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.cob = torch.tensor(prev_cob)

    def forward(self, input1, input2):
        if self.cob is None:
            self.cob = torch.ones(input1.shape[1])

        cob1_shape = (input1.shape[1],) + tuple([1 for _ in range(input1.dim() - 2)])
        self.cob = self.cob.view(cob1_shape).float()

        return input1 + self.cob * input2
