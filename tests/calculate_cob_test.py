import numpy as np
import torch

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def test_calculate_cob(network, model_name=None, input_shape=(1, 1, 28, 28), noise=False, verbose=False):
    """
        Test if the correct change of basis can be calculated for a random teleportation.

    Args:
        network (nn.Module): Network to be tested
        model_name (str): The name or label assigned to differentiate the model
        input_shape (tuple): Input shape of network
        noise (bool): whether to add noise to the target weights before optimisation.
        verbose (bool): whether to display sample ouputs during the test
    """
    model_name = model_name or network.__class__.__name__
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    w1 = model.get_weights(concat=False, flatten=False, bias=False)
    model.random_teleport()
    w2 = model.get_weights(concat=False, flatten=False, bias=False)

    if noise:
        for w in w2:
            w += torch.rand(w.shape) * 0.001

    cob = model.get_cob()
    calculated_cob = model.calculate_cob(w1, w2)

    error = (cob - calculated_cob).abs().mean()

    if verbose:
        print("Cob: ", cob.flatten()[:10])
        print("Calculated cob: ", calculated_cob.flatten()[:10])
        print("cob error ", (calculated_cob - cob).flatten()[:10])
        print("cob error : ", error)

    assert np.allclose(cob.detach().numpy(), calculated_cob.detach().numpy(), atol=1e-6),\
        "Calculate cob FAILED for " + model_name + " model."

    print("Calculate cob successful for " + model_name + " model.")


def test_calculate_cob_weights(network, model_name=None, input_shape=(1, 1, 28, 28), noise=False, verbose=True):
    """
        Test if a cob can be calculated and applied to a network to teleport the network from the initial weights to
        the targets weights.

    Args:
        network (nn.Module): Network to be tested
        model_name (str): The name or label assigned to differentiate the model
        input_shape (tuple): Input shape of network
        noise (bool): whether to add noise to the target weights before optimisation.
        verbose (bool): whether to display sample ouputs during the test
    """
    model_name = model_name or network.__class__.__name__
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    initial_weights = model.get_weights()
    w1 = model.get_weights(concat=False, flatten=False, bias=False)

    model.random_teleport()
    c1 = model.get_cob()
    model.random_teleport()
    c2 = model.get_cob()

    target_weights = model.get_weights()
    w2 = model.get_weights(concat=False, flatten=False, bias=False)

    if noise:
        for w in w2:
            w += torch.rand(w.shape) * 0.001

    calculated_cob = model.calculate_cob(w1, w2)

    model.initialize_cob()
    model.set_weights(initial_weights)
    model.teleport(calculated_cob, reset_teleportation=True)

    calculated_weights = model.get_weights()

    error = (calculated_weights - initial_weights).abs().mean()

    if verbose:
        print("weights: ", target_weights.flatten())
        print("Calculated cob weights: ", calculated_weights.flatten())
        print("Weight error ", error)
        print("C1: ", c1.flatten()[:10])
        print("C2: ", c2.flatten()[:10])
        print("C1 * C2: ", (c1 * c2).flatten()[:10])
        print("Calculated cob: ", calculated_cob.flatten()[:10])

    assert np.allclose(calculated_weights.detach().numpy(), target_weights.detach().numpy()), \
        "Calculate cob and weights FAILED for " + model_name + " model with error: " + str(error.item())

    print("Calculate cob and weights successful for " + model_name + " model.")


def test_calculate_ones(network, model_name=None, input_shape=(1, 1, 28, 28), noise=False, verbose=False):
    """
        Test if the correct change of basis can be calculated for a cob of ones.

    Args:
        network (nn.Module): Network to be tested
        model_name (str): The name or label assigned to differentiate the model
        input_shape (tuple): Input shape of network
        noise (bool): whether to add noise to the target weights before optimisation.
        verbose (bool): whether to display sample ouputs during the test
    """
    model_name = model_name or network.__class__.__name__
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    model.initialize_cob()

    w1 = model.get_weights(concat=False, flatten=False, bias=False)
    _w1 = model.get_weights(concat=False, flatten=False, bias=False)

    if noise:
        for w in _w1:
            w += torch.rand(w.shape) * 0.001

    cob = model.get_cob()
    calculated_cob = model.calculate_cob(w1, _w1)

    error = (cob - calculated_cob).abs().mean()

    if verbose:
        print("Cob: ", cob.flatten()[:10])
        print("Calculated cob: ", calculated_cob.flatten()[:10])
        print("cob error ", (calculated_cob - cob).flatten()[:10])
        print("cob error : ", error)

    assert np.allclose(cob, calculated_cob), "Calculate cob (ones) FAILED for " + model_name + " model."

    print("Calculate cob (ones) successful for " + model_name + " model.")


if __name__ == '__main__':
    import torch.nn as nn
    from torch.nn.modules import Flatten
    from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    test_calculate_cob(network=mlp_model, model_name="MLP", verbose=True)
    test_calculate_cob_weights(network=mlp_model, model_name="MLP", verbose=True)
    test_calculate_ones(network=mlp_model, model_name="MLP", verbose=True)
