import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training import train, test

# torch.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 3
batch_size = 128
metrics = [accuracy]
loss = nn.CrossEntropyLoss()

mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(FlattenCOB(),
                                 LinearCOB(784, 128),
                                 ReLUCOB(),
                                 LinearCOB(128, 128),
                                 ReLUCOB(),
                                 LinearCOB(128, 128),
                                 ReLUCOB(),
                                 LinearCOB(128, 10)
                                 )

    def forward(self, input):
        return self.net(input)


sample_input_shape = (1, 1, 28, 28)

model1 = Model().to(device)
model1 = NeuralTeleportationModel(network=model1, input_shape=sample_input_shape)
train(model1, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_test, metrics=metrics,
      epochs=epochs, device=device, batch_size=batch_size)

model2 = Model().to(device)
model2 = NeuralTeleportationModel(network=model2, input_shape=sample_input_shape)
train(model2, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_test, metrics=metrics,
      epochs=epochs, device=device, batch_size=batch_size)

print("Before")
w1 = model1.get_weights()
w2 = model2.get_weights()
diff = (w1.detach().cpu()-w2.detach().cpu()).abs().mean()
print(diff)

w1 = model1.get_weights(concat=False, flatten=False, bias=False)
w2 = model2.get_weights(concat=False, flatten=False, bias=False)
calculated_cob = model1.calculate_cob(w1, w2, concat=True)


model1.set_change_of_basis(calculated_cob)
model1.teleport()

print("After")
w1 = model1.get_weights()
w2 = model2.get_weights()
diff = (w1.detach().cpu()-w2.detach().cpu()).abs().mean()
print(diff)

w1 = model1.get_weights(concat=False, flatten=False, bias=False)
w2 = model2.get_weights(concat=False, flatten=False, bias=False)

for i in range(len(w1)):
    print('layer : ', i)
    print("w1  - w2 = ", (w1[i].detach().cpu()-w2[i].detach().cpu()).abs().sum())

