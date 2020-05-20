# This script validates that the scalar product between a line defined by a set of weights and its teleporation
# and the gradient calculated a that point is null. It does this for many sampling types
# (see change get_random_cob in changeofbaseutils.py)

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

import numpy as np
import torch

red = "\033[31m"
reset = "\033[0m"


# Test the initial assignation of weights to a model
def test_set_weights(network, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape)
    w1 = model.get_weights().detach().numpy()
    model = NeuralTeleportationModel(network, input_shape)

    model.set_weights(w1)
    w3 = model.get_weights().detach().numpy()

    assert np.allclose(w1, w3)


# Test the dot produt between the teleporation line and the gradient for nullity
def test_dot_product(network, input_shape=(1, 1, 28, 28)) -> None:
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
              f' using {sampling_type} sampling type', sep='')


def test_reset_weights(network, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape=input_shape)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)


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

    test_set_weights(network=mlp_model)
    test_dot_product(network=mlp_model)
    test_reset_weights(network=mlp_model)

    test_set_weights(network=cnn_model)
    test_dot_product(network=cnn_model)
    test_reset_weights(network=cnn_model)
