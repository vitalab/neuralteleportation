import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import MNIST

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

torch.manual_seed(10)

# net = nn.Sequential(FlattenCOB(),
#                     LinearCOB(784, 128),
#                     ReLUCOB(),
#                     LinearCOB(128, 128),
#                     ReLUCOB(),
#                     LinearCOB(128, 10)
#                     )
# sample_input_shape = (1, 1, 28, 28)
#
# mnist_train = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
# mnist_test = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
#
# optimizer = torch.optim.Adam(net.parameters())
# metrics = [accuracy]
# loss = nn.CrossEntropyLoss()
# train(net, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_test, optimizer=optimizer, metrics=metrics,
#       epochs=0)

net = nn.Sequential(LinearCOB(2, 3, bias=False),
                    ReLUCOB(),
                    LinearCOB(3, 1, bias=False)
                    )
sample_input_shape = (1, 1, 2)

model = NeuralTeleportationModel(network=net, input_shape=sample_input_shape, )

# Get the initial set of weights and teleport.
initial_weights = model.get_weights()
model.random_teleport(cob_range=1)

# Get second set of weights (Goal weights)
target_weights = model.get_weights()

# Get the change of basis that created this set of weights.
target_cob = model.get_cob(concat=True)

print("Target cob shape: ", target_cob.shape)

# Generate a new random cob
torch.manual_seed(20)
cob = model.generate_random_cob(cob_range=1)
print("cob shape: ", cob.shape)
print("cob grad_fn: ", cob.grad_fn)

history = []
cob_error_history = []

print("Inital error: ", (cob - target_cob).abs().mean().item())
print("Target cob: ", target_cob[0:10].data)
print("cob: ", cob[0:10].data)

"""
Optimize the cob to find the 'target_cob' that produced the original teleportation. 
"""

optimizer = optim.Adam([cob], lr=1e-3)
# scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)

for e in range(5000):
    # Reset the initial weights.
    model.set_weights(initial_weights)

    # Teleport with this cob
    model.set_change_of_basis(cob)
    model.teleport()

    # Get the new weights and calculate the loss
    weights = model.get_weights()
    loss = (weights - target_weights).square().mean()

    # Backwards pass
    # cob.retain_grad()
    loss.backward(retain_graph=True)

    # Calculate cob gradient and perform gradient descent step
    # grad = cob.grad
    # cob -= cob.grad * 0.01
    # cob.grad.zero_()

    grad = cob.grad

    optimizer.step()
    optimizer.zero_grad()

    # scheduler.step()

    history.append(loss.item())
    cob_error_history.append((cob - target_cob).square().mean().item())
    if e % 100 == 0:
        print("Step: {}, loss: {}, cob error: {}".format(e, loss.item(), (cob - target_cob).abs().mean().item()))

print("Final error: ", (cob - target_cob).abs().mean().item())
print("Target cob: ", target_cob[0:10].data)
print("cob: ", cob[0:10].data)

print("Inital weights: ", initial_weights[:10])
print("Target weights: ", target_weights[:10])
print("Weights: ", weights[:10])
print("Weight diff: ", (target_weights[:10] - weights[:10]).abs())

plt.figure()
plt.plot(history)
plt.title("Loss")
plt.show()

plt.figure()
plt.plot(cob_error_history)
plt.title("Cob error")
plt.show()
