import torch
import numpy as np

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def test_cuda_teleport(network, input_shape=(1, 1, 28, 28), verbose=False):

    network = network.cuda()

    model = NeuralTeleportationModel(network=network, input_shape=input_shape)

    x = torch.rand(input_shape).cuda()
    pred1 = model(x).cpu().detach().numpy()
    w1 = model.get_weights().cpu().detach().numpy()

    model.random_teleport()

    pred2 = model(x).cpu().detach().numpy()
    w2 = model.get_weights().cpu().detach().numpy()

    diff_average = (w1 - w2).mean()

    if verbose:
        print("Model on device: {}".format(next(network.parameters()).device))
        print("Sample outputs: ")
        print("Pre teleportation: ", pred1.flatten()[:10])
        print("Post teleportation: ", pred2.flatten()[:10])

    assert not np.allclose(w1, w2)
    assert np.allclose(pred1, pred2), "Teleporation did not work. Average difference: {}".format(diff_average)

    return diff_average


if __name__ == '__main__':
    import torch.nn as nn
    from torch.nn.modules import Flatten
    from neuralteleportation.layers.layer_utils import patch_module

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

    cnn_model = patch_module(cnn_model)

    test_cuda_teleport(network=cnn_model, verbose=True)