import torch
from neuralteleportation.layer_utils import patch_module
from neuralteleportation.model import NeuralTeleportationModel

def test_patch_modulde():
    test_input  = torch.rand((1,10))
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5, 2),
    )

    cob_module = patch_module(test_module)

    print(test_module)
    print(cob_module)


    # assert not any(
    #     type(module) in NeuralTeleportationModel.SUPPORTED_LAYERS for module in cob_module.modules()
    # )


if __name__ == '__main__':
    test_patch_modulde()
