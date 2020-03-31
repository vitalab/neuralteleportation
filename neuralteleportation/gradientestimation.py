import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from neuralteleportation.metrics import accuracy
from neuralteleportation.model import NeuralTeleportationModel

from numpy import linalg as LA
import torch

mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

loss = nn.CrossEntropyLoss()

model = NeuralTeleportationModel()

data_loader = DataLoader(mnist_test, batch_size=len(mnist_test))

data, target = next(iter(data_loader))

grad = model.get_grad(data, target, loss)

w1 = model.get_weights()

model.teleport(cob_range=1.0001)
# model.teleport(cob_range=10)

w2 = model.get_weights()

distance = torch.dist(w1, w2)

print("Distance between weights: {}".format(distance))

tangent = w1 - w2

dot = tangent.dot(grad)

print("Dot product between tangent and gradient: {}".format(dot))
