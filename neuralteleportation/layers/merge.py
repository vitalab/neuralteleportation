from typing import Union

import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.layers.neuralteleportation import NeuralTeleportationLayerMixin


class Concat(NeuralTeleportationLayerMixin, nn.Module):
    """Implementation of concatenation layer for teleportation that wraps torch.cat.

    IMPORTANT NOTE: The order of parameters to the forward method is important. Input1 must be computed
    before input2 in the network containing this layer. This is because this layers change of basis in the
    concatenation of the previous layers in the order determined by the graph. The forward method must respect that
    order.

    Example:
        Class network(nn.Module):
            def forward(x):
                x1 = layer1(x) #Computed First
                x2 = layer2(x1) #Computed second

                x3 = self.Concat(x1, x2) # x1 comes before x2.
    """

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def forward(self, *args, dim: Union[int, None] = 1):
        return torch.cat(list(args), dim=dim)


class Add(NeuralTeleportationLayerMixin, nn.Module):
    """Implementation of Add layer for teleportation that is equivalent to += in the forward method.

    IMPORTANT NOTE: The order of parameters to the forward method is important. Input1 must be computed
    before input2 in the network containing this layer. This is because we must apply the change of basis of
    the previous layer in the forward method.

    Example:
        Class network(nn.Module):
            def forward(x):
                x1 = layer1(x) #Computed First
                x2 = layer2(x1) #Computed second

                x3 = Add(x1, x2) # x1 comes before x2.
    """
    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        self.prev_cob = prev_cob
        self.next_cob = next_cob

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def forward(self, input1, input2):
        if self.prev_cob is None:
            self.prev_cob = torch.ones(input2.shape[1])
        if self.next_cob is None:
            self.next_cob = torch.ones(input2.shape[1])

        cob1_shape = (input2.shape[1],) + tuple([1 for _ in range(input2.dim() - 2)])
        self.prev_cob = self.prev_cob.view(cob1_shape).float().type_as(input1)

        next_cob_shape = (input2.shape[1],) + tuple([1 for _ in range(input2.dim() - 2)])
        self.next_cob = self.next_cob.view(next_cob_shape).float().type_as(input1)

        return torch.add(input1, self.next_cob * input2 / self.prev_cob)
