import numpy as np
import torch

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def test_set_weights(network, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape)
    w1 = model.get_weights().detach().numpy()
    model = NeuralTeleportationModel(network, input_shape)
    w2 = model.get_weights().detach().numpy()

    model.set_weights(w1)
    w3 = model.get_weights().detach().numpy()

    # assert not np.allclose(w1, w2)
    assert np.allclose(w1, w3)


def test_teleport(network, input_shape=(1, 1, 28, 28), verbose=False):
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)
    x = torch.rand(input_shape)
    pred1 = model(x).detach().numpy()
    w1 = model.get_weights().detach().numpy()

    model.random_teleport()

    pred2 = model(x).detach().numpy()
    w2 = model.get_weights().detach().numpy()

    diff_average = (w1 - w2).mean()

    if verbose:
        print("Sample outputs: ")
        print("Pre teleportation: ", pred1.flatten()[:10])
        print("Post teleportation: ", pred2.flatten()[:10])
        print("Diff weight average: ", (w1 - w2).mean())
        print("Diff prediction average: ", (pred1 - pred2).mean())

    assert not np.allclose(w1, w2)
    assert np.allclose(pred1, pred2, atol=1e-5), "Teleporation did not work. Average difference: {}".format(diff_average)

    return diff_average


def test_reset_weights(network, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape=input_shape)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)


def test_calculate_cob(network, input_shape=(1, 1, 28, 28), noise=False, verbose=True):
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    w1 = model.get_weights(concat=False, flatten=False, bias=False)
    model.random_teleport()
    w2 = model.get_weights(concat=False, flatten=False, bias=False)

    if noise:
        for w in w2:
            w += torch.rand(w.shape) * 0.001

    cob = model.get_cob()
    calculated_cob = model.calculate_cob(w1, w2)

    if verbose:
        print("Cob: ", cob.flatten()[:10])
        print("Calculated cob: ", calculated_cob.flatten()[:10])

    # assert np.allclose(cob, calculated_cob)


def test_calculate_cob2(network, input_shape=(1, 1, 28, 28), noise=False, verbose=True):
    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    _w1 = model.get_weights()
    w1 = model.get_weights(concat=False, flatten=False, bias=False)
    model.random_teleport()
    w2 = model.get_weights(concat=False, flatten=False, bias=False)
    _w2 = model.get_weights()

    if noise:
        for w in w2:
            w += torch.rand(w.shape) * 0.001

    calculated_cob = model.calculate_cob(w1, w2)

    model.set_weights(_w1)
    model.set_change_of_basis(calculated_cob)
    model.teleport()

    _w2_ = model.get_weights()

    if verbose:
        print("Cob: ", _w2.flatten()[:10])
        print("Calculated cob: ", _w2_.flatten()[:10])

    # assert np.allclose(_w2_, _w2)




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

    # test_set_weights(network=mlp_model)
    # test_teleport(network=mlp_model)
    # test_reset_weights(network=mlp_model)
    #
    # test_set_weights(network=cnn_model)
    # test_teleport(network=cnn_model)
    # test_reset_weights(network=cnn_model)

    test_calculate_cob(mlp_model, verbose=True, noise=False)
    test_calculate_cob2(mlp_model, verbose=True, noise=False)

