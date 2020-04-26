from typing import Union, Tuple, List

import torch.nn as nn
import torch
from neuralteleportation.layers.layers_v3 import MergeLayersMixin


class Concat(nn.Module, MergeLayersMixin):
    def forward(self, tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: Union[str, None] = 1):
        return torch.cat(tensors, dim=dim)

    def apply_cob(self, prev_cob, next_cob):
        pass


class Add(nn.Module, MergeLayersMixin):
    def __init__(self):
        super().__init__()
        self.prev_cob = None
        self.next_cob = None

    def apply_cob(self, prev_cob, next_cob):
        self.prev_cob = torch.tensor(prev_cob)
        self.next_cob = torch.tensor(next_cob)

    def forward(self, input1, input2):
        if self.prev_cob is None:
            self.prev_cob = torch.ones(input2.shape[1])
        if self.next_cob is None:
            self.next_cob = torch.ones(input2.shape[1])

        cob1_shape = (input2.shape[1],) + tuple([1 for _ in range(input2.dim() - 2)])
        self.prev_cob = self.prev_cob.view(cob1_shape).float()

        next_cob_shape = (input2.shape[1],) + tuple([1 for _ in range(input2.dim() - 2)])
        self.next_cob = self.next_cob.view(next_cob_shape).float()
        # print("Add, ", self.prev_cob.flatten(), ', ', self.next_cob.flatten())

        return torch.add(input1, self.next_cob * input2 / self.prev_cob)
        # return input1 + self.next_cob * input2 / self.prev_cob
