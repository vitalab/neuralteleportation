import torch
import torch.nn as nn

from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules


def test_swap_module():
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    cob_module = swap_model_modules_for_COB_modules(test_module, inplace=False)

    assert not any(
        type(module) in [nn.Linear, nn.Conv2d] for module in cob_module.modules()
    )


if __name__ == '__main__':
    test_swap_module()
