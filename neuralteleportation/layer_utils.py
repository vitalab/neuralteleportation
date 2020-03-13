import copy

import torch

from neuralteleportation.layers import LinearCOB, Conv2DCOB


def patch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace layers with COB layers
    """
    if not inplace:
        module = copy.deepcopy(module)
    _patch_cob_layers(module)
    return module


def _patch_cob_layers(module: torch.nn.Module) -> None:
    """
    Recursively iterate over the children of a module and replace them if
    they are a Cob layer. This function operates in-place.
    """

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            new_module = LinearCOB(in_features=child.in_features, out_features=child.out_features,
                                   bias=(child.bias is not None))
        elif isinstance(child, torch.nn.Conv2d):
            new_module = Conv2DCOB(in_channels=child.in_channels, out_channels=child.out_channels,
                                   kernel_size=child.kernel_size, stride=child.stride,
                                   padding=child.padding, dilation=child.dilation, groups=child.groups,
                                   bias=(child.bias is not None), padding_mode=child.padding_mode)
        else:
            new_module = None

        if new_module is not None:
            module.add_module(name, new_module)

        # recursively apply to child
        _patch_cob_layers(child)
