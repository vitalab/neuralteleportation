from _collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from tqdm import tqdm

from neuralteleportation.layers.activationlayers import ReLUCOB
from neuralteleportation.layers.neuralteleportationlayers import FlattenCOB
from neuralteleportation.layers.neuronlayers import LinearCOB
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training import test

mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

metrics = [accuracy]
loss = nn.CrossEntropyLoss()


def interpolate(vect_a, vect_b, alpha):
    return alpha * vect_a + (1.0 - alpha) * vect_b


def get_colors(alphas):
    """
        Get colors for interpolation plot
        ]-inf, 0[ -> red
        [0] -> green
        ]0, 1[ -> blue
        [1] -> green
        ]1, inf[ -> red
    Args:
        alphas:

    Returns:

    """
    c = np.chararray(shape=(len(alphas),))
    alphas = np.around(alphas, decimals=3)
    c[(alphas < 0) | (alphas > 1)] = 'r'
    c[(alphas >= 0) & (alphas <= 1)] = 'b'
    c[(alphas == 0) | (alphas == 1)] = 'g'
    c = c.astype('<U1')
    return c.tolist()


net = nn.Sequential(FlattenCOB(),
                    LinearCOB(784, 128),
                    ReLUCOB(),
                    LinearCOB(128, 128),
                    ReLUCOB(),
                    LinearCOB(128, 10),
                    )

# net = Net()

sample_input_shape = (1, 1, 28, 28)
model = NeuralTeleportationModel(network=net, input_shape=sample_input_shape)
print(model)

w1 = model.get_weights()
print(test(model, loss, metrics, mnist_test))
model.random_teleport()
w2 = model.get_weights()
print(test(model, loss, metrics, mnist_test))

print(w1)
print(w2)

distance = torch.dist(w1, w2)

alphas = np.arange(-1, 2, 0.1)
colors = get_colors(alphas)
improvement_key = 'accuracy'


for e in range(10):
    results = defaultdict(list)

    model.random_teleport()
    w2 = model.get_weights()

    best_weights = w1
    best_loss = float("-inf")

    for i in tqdm(range(len(alphas))):
        w = interpolate(w1, w2, alphas[i])
        model.set_weights(w)
        res = test(model, loss, metrics, mnist_test)

        if res[improvement_key] > best_loss:
            best_loss = res[improvement_key]
            best_weights = w

        for k in res.keys():
            results[k].append(res[k])

    print("Epoch: {}, loss = {}".format(e, best_loss))

    w1 = best_weights

    fig, axs = plt.subplots(1,2)
    plt.suptitle("Epoch {}".format(e))
    axs = axs.ravel()
    for i, k in enumerate(results.keys()):
        axs[i].set_title(k)
        axs[i].scatter(alphas, results[k], c=colors)

    plt.show(block=True)
