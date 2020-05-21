import numpy as np
import torch

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def test_set_weights(network, model_name, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape)
    w1 = model.get_weights().detach().numpy()

    model.set_weights(w1)
    w2 = model.get_weights().detach().numpy()

    assert np.allclose(w1, w2)
    print("Weights set successfully for " + model_name + " model.")


def test_teleport(network, model_name, input_shape=(1, 1, 28, 28), verbose=False):
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
    print("Teleportation successful for " + model_name + " model.")
    return diff_average


def test_reset_weights(network, model_name, input_shape=(1, 1, 28, 28)):
    model = NeuralTeleportationModel(network, input_shape=input_shape)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    print("Reset weights successful for " + model_name + " model.")


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

    test_set_weights(network=mlp_model, model_name="MLP")
    test_teleport(network=mlp_model, model_name="MLP")
    test_reset_weights(network=mlp_model, model_name="MLP")

    test_set_weights(network=cnn_model, model_name="Convolutional")
    test_teleport(network=cnn_model, model_name="Convolutional")
    test_reset_weights(network=cnn_model, model_name="Convolutional")

