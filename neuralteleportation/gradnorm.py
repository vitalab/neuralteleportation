import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from neuralteleportation.metrics import accuracy
from neuralteleportation.model import NeuralTeleportationModel

from numpy import linalg as LA

mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

metrics = [accuracy]
loss = nn.CrossEntropyLoss()



model = NeuralTeleportationModel()

data_loader = DataLoader(mnist_test, batch_size=len(mnist_test))

data, target = next(iter(data_loader))

print(data.shape)
print(target.shape)

for i in range(20):
    grad = model.get_grad(data, target, loss)
    w = model.get_weights()
    grad_norm = LA.norm(grad)

    print("Gradient: {}".format(grad_norm))
    print("Weight sum: {}".format(w.sum()))
    print("Weighted gradient {}".format(grad_norm / w.sum()))

    model.teleport()



