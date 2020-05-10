import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

from numpy import linalg as LA

from neuralteleportation.layers.neuralteleportationlayers import FlattenCOB
from neuralteleportation.layers.neuronlayers import LinearCOB
from neuralteleportation.layers.activationlayers import ReLUCOB
import torch.nn as nn

mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
mnist_val = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())

metrics = [accuracy]
loss = nn.CrossEntropyLoss()

net = nn.Sequential(FlattenCOB(),
                    LinearCOB(784, 128),
                    ReLUCOB(),
                    LinearCOB(128, 128),
                    ReLUCOB(),
                    LinearCOB(128, 10),
                    )

sample_input_shape = (1, 1, 28, 28)
model = NeuralTeleportationModel(network=net, input_shape=sample_input_shape)

data_loader = DataLoader(mnist_test, batch_size=len(mnist_test))

data, target = next(iter(data_loader))

print(data.shape)
print(target.shape)

w0 = model.get_weights()
for i in range(20):
    grad = model.get_grad(data, target, loss)
    w = model.get_weights()
    grad_norm = LA.norm(grad)

    print("------------------------------------------------")
    print("Gradient: {}".format(grad_norm))
    print("Weight sum: {}".format(w.sum()))
    print("Weighted gradient {}".format(grad_norm / w.sum()))

    model.set_weights(w0)
    model.random_teleport()



