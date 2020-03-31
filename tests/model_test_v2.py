from neuralteleportation.model_v2 import NeuralTeleportationModel
import numpy as np
import torch


def test_set_weights(network):
    model = NeuralTeleportationModel(network)
    w1 = model.get_weights().detach().numpy()
    model = NeuralTeleportationModel(network)
    w2 = model.get_weights().detach().numpy()

    model.set_weights(w1)
    w3 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    assert np.allclose(w1, w3)


def test_teleport(network):
    model = NeuralTeleportationModel(network=network)
    x = torch.rand((1, 1, 28, 28))
    pred1 = model(x).detach().numpy()
    w1 = model.get_weights().detach().numpy()

    model.teleport()

    pred2 = model(x).detach().numpy()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    assert np.allclose(pred1, pred2)


def test_reset_weights(network):
    model = NeuralTeleportationModel(network)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)


if __name__ == '__main__':
    import torch.nn as nn
    from neuralteleportation.layers import Flatten

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

    test_set_weights(network=mlp_model)
    test_teleport(network=mlp_model)
    test_reset_weights(network=mlp_model)

    test_set_weights(network=cnn_model)
    test_teleport(network=cnn_model)
    test_reset_weights(network=cnn_model)
