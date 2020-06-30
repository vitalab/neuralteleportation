import copy
import inspect

import torch
import torch.nn as nn
from torch.nn.modules import Flatten

from neuralteleportation.layers.activation import ReLUCOB, TanhCOB, SigmoidCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB, Conv2dCOB, ConvTranspose2dCOB, BatchNorm1dCOB, \
    BatchNorm2dCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB, AvgPool2dCOB

# Mapping from nn.Modules to COB layers.
COB_LAYER_DICT = {nn.Linear: LinearCOB,
                  nn.Conv2d: Conv2dCOB,
                  nn.ReLU: ReLUCOB,
                  nn.Tanh: TanhCOB,
                  nn.Sigmoid: SigmoidCOB,
                  nn.ConvTranspose2d: ConvTranspose2dCOB,
                  nn.AvgPool2d: AvgPool2dCOB,
                  nn.MaxPool2d: MaxPool2dCOB,
                  nn.BatchNorm2d: BatchNorm2dCOB,
                  nn.BatchNorm1d: BatchNorm1dCOB,
                  Flatten: FlattenCOB}


def swap_model_modules_for_COB_modules(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace normal layers with COB layers."""
    if not inplace:
        module = copy.deepcopy(module)
    _swap_cob_layers(module)
    return module


def _get_args_dict(fn, args, kwargs):
    """Get args in the form of a dict to re-create exactly the same layers."""
    args_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    return {**dict(zip(args_names, args)), **kwargs}


def _swap_cob_layers(module: torch.nn.Module) -> None:
    """
    Recursively iterate over the children of a module and replace them if
    they have an equivalent Cob layer. This function operates in-place.
    """

    for name, child in module.named_children():
        params = {k: v for k, v in child.__dict__.items() if k in inspect.getfullargspec(child.__init__).args}
        new_module = COB_LAYER_DICT[child.__class__](**params)

        if new_module is not None:
            module.add_module(name, new_module)

        # recursively apply to child
        _swap_cob_layers(child)
