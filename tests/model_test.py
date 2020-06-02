from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def test_set_weights(network: nn.Module, input_shape: Tuple = (1, 1, 28, 28), model_name: str = None):
    """
        test_set_weights checks if method set_weights() in NeuralTeleportationModel works
    Args:
        network (nn.Module): Network to test
        input_shape (tuple): Input shape of network
        model_name (str): The name or label assigned to differentiate the model
    """
    model_name = model_name or network.__class__.__name__

    model = NeuralTeleportationModel(network, input_shape)
    w1 = model.get_weights()

    model.reset_weights()
    model.set_weights(w1)
    w2 = model.get_weights()

    assert np.allclose(w1.detach().numpy(), w2.detach().numpy())
    print("Weights set successfully for " + model_name + " model.")


def test_teleport(network: nn.Module, input_shape: Tuple = (1, 1, 28, 28), verbose: bool = False,
                  atol: float = 1e-5, model_name: str = None):
    """
        Return mean of the difference between the weights of network and a random teleportation, and checks if
        teleportation has the same network function

    Args:
        network (nn.Module): Network to be tested
        input_shape (tuple): Input shape of network
        verbose (bool): Flag to print comparision between network and a teleportation
        atol (float): Absolute tolerance allowed between outputs to pass the test
        model_name (str): The name or label assigned to differentiate the model

    Returns:
        float with the average of the difference between the weights of the network and a teleportation
    """
    model_name = model_name or network.__class__.__name__
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)
    x = torch.rand(input_shape)
    pred1 = model(x).detach().numpy()
    w1 = model.get_weights().detach().numpy()

    model.random_teleport()

    pred2 = model(x).detach().numpy()
    w2 = model.get_weights().detach().numpy()

    diff_average = np.mean(np.abs((pred1 - pred2)))

    if verbose:
        print("Sample outputs: ")
        print("Pre teleportation: ", pred1.flatten()[:10])
        print("Post teleportation: ", pred2.flatten()[:10])
        print("Diff weight average: ", np.mean(np.abs((w1 - w2))))
        print("Diff prediction average: ", diff_average)

    # assert not np.allclose(w1, w2)
    assert np.all(~np.isclose(w1, w2))
    assert np.allclose(pred1, pred2, atol=atol), "Teleporation did not work for model {}. Average difference: {}". \
        format(model_name, diff_average)

    print("Teleportation successful for " + model_name + " model.")
    return diff_average


def test_reset_weights(network: nn.Module, input_shape: Tuple = (1, 1, 28, 28), model_name: str = None):
    """
        test_reset_weights checks if method reset_weights() in NeuralTeleportationModel works

    Args:
        network (nn.Module): Network to be tested
        input_shape (tuple): Input shape of network
        model_name (str): The name or label assigned to differentiate the model

    """
    model_name = model_name or network.__class__.__name__
    model = NeuralTeleportationModel(network, input_shape=input_shape)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    print("Reset weights successful for " + model_name + " model.")


def test_set_cob(network, model_name, input_shape=(1, 1, 28, 28), verbose=False):
    x = torch.rand(input_shape)
    model = NeuralTeleportationModel(network, input_shape=input_shape)
    model.random_teleport()
    w1 = model.get_weights()
    t1 = model.get_cob()
    pred1 = model(x)

    model.reset_weights()
    pred2 = model(x)

    model.set_change_of_basis(t1)
    model.set_weights(w1)
    model.apply_cob()

    pred3 = model(x)

    if verbose:
        print("Diff prediction average: ", (pred1 - pred3).mean())
        print("Pre teleportation: ", pred1.flatten()[:10])
        print("Post teleportation: ", pred3.flatten()[:10])

    assert not np.allclose(pred1.detach().numpy(), pred2.detach().numpy(), atol=1e-5)
    assert np.allclose(pred1.detach().numpy(), pred3.detach().numpy(), atol=1e-5), "Set cob/weights did not work."

    print("Set cob successful for " + model_name + " model.")


def test_calculate_cob(network, model_name, input_shape=(1, 1, 28, 28), noise=False, verbose=False):
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
        print("Cob: ", cob.flatten())
        print("Calculated cob: ", calculated_cob.flatten())
        print("cob error ", calculated_cob.flatten() - cob.flatten())
        print("cob error : ", error)

    assert np.allclose(cob, calculated_cob)

    print("Calculate cob successful for " + model_name + " model.")


# def test_calculate_cob2(network, input_shape=(1, 1, 28, 28), noise=False, verbose=True):
#     model = NeuralTeleportationModel(network=network, input_shape=input_shape)
#
#     _w1 = model.get_weights()
#     w1 = model.get_weights(concat=False, flatten=False, bias=False)
#     model.random_teleport()
#     w2 = model.get_weights(concat=False, flatten=False, bias=False)
#     _w2 = model.get_weights()
#
#     if noise:
#         for w in w2:
#             w += torch.rand(w.shape) * 0.001
#
#     calculated_cob = model.calculate_cob(w1, w2)
#
#     model.set_weights(_w1)
#     model.set_change_of_basis(calculated_cob)
#     model.teleport()
#
#     _w2_ = model.get_weights()
#
#     if verbose:
#         print("Cob: ", _w2.flatten()[:10])
#         print("Calculated cob: ", _w2_.flatten()[:10])
#
#     # assert np.allclose(_w2_, _w2)


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
        # nn.ReLU(),
        # nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    cnn_model = swap_model_modules_for_COB_modules(cnn_model)
    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    # test_set_weights(network=mlp_model, model_name="MLP")
    # test_teleport(network=mlp_model, model_name="MLP")
    # test_reset_weights(network=mlp_model, model_name="MLP")
    #
    # test_set_weights(network=cnn_model, model_name="Convolutional")
    # test_teleport(network=cnn_model, model_name="Convolutional")
    # test_reset_weights(network=cnn_model, model_name="Convolutional")
    #
    # test_set_cob(network=mlp_model, model_name="MLP")
    # test_set_cob(network=cnn_model, model_name="Convolutional")

    test_calculate_cob(network=mlp_model, model_name="MLP", verbose=True)
