from typing import Sequence, Type

from torch import nn


def test_generic_models(modules: Sequence[Type[nn.Module]], verbose: bool = True):
    from tests.model_test import test_teleport

    input_shape = (1, 1, 28, 28)

    for module in modules:
        model = module()
        print("-----------------------------------------------------------")
        print("Testing model: {}".format(module.__name__))
        try:
            diff_avg = test_teleport(model, input_shape, verbose=verbose)
            print("{} model passed with avg diff: {}".format(module.__name__, diff_avg))
        except Exception as e:
            print("Teleportation failed for model: {} with error {}".format(module.__name__, e))

    print("All tests are done")
