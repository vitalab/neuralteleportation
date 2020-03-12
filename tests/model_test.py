from neuralteleportation.model import NeuralTeleportationModel
import numpy as np
import torch


def test_set_weights(use_bias=True):
    model = NeuralTeleportationModel(use_bias=use_bias)
    w1 = model.get_weights().detach().numpy()
    model = NeuralTeleportationModel(use_bias=use_bias)
    w2 = model.get_weights().detach().numpy()

    model.set_weights(w1)
    w3 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    assert np.allclose(w1, w3)


def test_change_of_basis(use_bias=False, input_size=784):
    model = NeuralTeleportationModel(input_dim=input_size, use_bias=use_bias)
    x = torch.rand((1, 784))
    pred1 = model(x).detach().numpy()
    w1 = model.get_weights().detach().numpy()

    model.apply_change_of_basis()

    pred2 = model(x).detach().numpy()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)
    assert np.allclose(pred1, pred2)


def test_reset_weights(use_bias=True):
    model = NeuralTeleportationModel(use_bias=use_bias)
    w1 = model.get_weights().detach().numpy()
    model.reset_weights()
    w2 = model.get_weights().detach().numpy()

    assert not np.allclose(w1, w2)



if __name__ == '__main__':
    test_set_weights()
    test_change_of_basis()
    test_reset_weights()
