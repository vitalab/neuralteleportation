import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neuralteleportation.layers.activationlayers import ReLUCOB
from neuralteleportation.layers.neuralteleportationlayers import FlattenCOB
from neuralteleportation.layers.neuronlayers import LinearCOB
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.models.model_zoo.resnet import resnet18

# torch.manual_seed(1234)
from neuralteleportation.training import train, test

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
                                 LinearCOB(128, 10)
                                 )

    def forward(self, input):
        return self.net(input)


sample_input_shape = (1, 1, 28, 28)

model1 = NeuralTeleportationModel(network=Model(), input_shape=sample_input_shape)
optimizer = torch.optim.Adam(model1.parameters())
metrics = [accuracy]
loss = nn.CrossEntropyLoss()
train(model1, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_test, optimizer=optimizer, metrics=metrics,
      epochs=2)

model2 = NeuralTeleportationModel(network=Model(), input_shape=sample_input_shape)
optimizer = torch.optim.Adam(model2.parameters())
metrics = [accuracy]
loss = nn.CrossEntropyLoss()
train(model2, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_test, optimizer=optimizer, metrics=metrics,
      epochs=2)

w1 = model1.get_weights(concat=False, flatten=False, bias=False)

w2 = model2.get_weights(concat=False, flatten=False, bias=False)

calculated_cob = model1.calculate_cob(w1, w2, concat=True)


model1.set_change_of_basis(calculated_cob)
model1.teleport()

w2_ = model1.get_weights(concat=False, flatten=False, bias=False)

print("Before")
print(w2)
print("After")
print(w2_)

# model1.set_change_of_basis(torch.cat([t1, t2]))
# model1.teleport()
# # w1 = model1.get_weights(concat=False, flatten=False, bias=False)
# w1_ = model1.get_weights()
# print(test(model1, loss, metrics, mnist_test))
#
# w2 = model2.get_weights()
#
# print(w1_)
# print(w2)
# print((w1_ - w2).abs().mean())

############################

# # Get the initial set of weights and teleport.
# initial_weights = model.get_weights(concat=False, flatten=False, bias=False)
# model.random_teleport(cob_range=1)
#
# # Get second set of weights (Goal weights)
# target_weights = model.get_weights(concat=False, flatten=False, bias=False)
#
# # Get the change of basis that created this set of weights.
# target_cob = model.get_cob(concat=True)


# print("Target cob: ", target_cob)

# print(target_weights)

# print("Computed cob")
# cob = [torch.ones(initial_weights[0].shape[1])]
# for i in range(len(initial_weights)):
#     c = ((target_weights[i] / initial_weights[i]) * cob[-1][None])[:, 0]
#     # print(c)
#     cob.append(c)
#
# print(cob)

# cob = get_cob(initial_weights, target_weights)
#
# print("Target cob: ", target_cob.shape)
# print("cob: ", cob.shape)
#
# # print("Target cob: ", target_cob)
# # print("cob: ", cob)
#
# print("Cob diff:", (target_cob - cob))
# print("Cob diff:", (target_cob - cob).abs().mean())
