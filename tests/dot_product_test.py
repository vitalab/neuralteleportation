# This script validates that the scalar product between a line defined by a set of weights and its teleporation
# and the gradient calculated a that point is null. It does this for many sampling types
# (see change get_random_cob in changeofbaseutils.py)

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

import numpy as np
import torch

red = "\033[31m"
reset = "\033[0m"


def test_dot_product(network, input_shape=(1, 1, 28, 28)) -> None:
    r"""
    This method tests the scalar produt between the teleporation line and the gradient for nullity

    Args:
        network : the model to which we wish to assigne weights

        input_shape :   the shape of the input
                        (it will be used by the networkgrapher of the model,
                        the values is not important for the test at hand)
    """

    model = NeuralTeleportationModel(network=network, input_shape=input_shape)
    x = torch.rand(input_shape, dtype=torch.float)
    y = torch.rand((1, 1, 10, 10), dtype=torch.float)
    w1 = model.get_weights().detach().numpy()
    sampling_types = ['usual', 'symmetric', 'negative', 'zero']

    loss_func = torch.nn.MSELoss()

    grad = model.get_grad(x, y, loss_func, zero_grad=False)

    for sampling_type in sampling_types:
        model.random_teleport(cob_range=0.0001, sampling_type=sampling_type)
        w2 = model.get_weights().detach().numpy()

        # Normalized scalar product
        dot_prod = np.dot(grad, (w2 - w1))/(np.linalg.norm(grad)*np.linalg.norm((w2 - w1)))

        # Arbitrary precision threshold for nullity comparison
        tol = 1e-5
        failed = (not np.allclose(dot_prod, 0, atol=tol))

        print(f'The result of the scalar product between the gradient and a micro-teleporation vector is: '
              f'{red * failed}{np.round(abs(dot_prod), int(abs(np.log10(tol))))}',
              f' (!=0 => FAILED!)' * failed,
              f'{reset}',
              f' using {sampling_type} sampling type',
              sep='')


if __name__ == '__main__':
    import torch.nn as nn
    from torch.nn.modules import Flatten
    from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules

    cnn_model = torch.nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    cnn_model = swap_model_modules_for_COB_modules(cnn_model)
    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    test_dot_product(network=mlp_model)

    test_dot_product(network=cnn_model)
