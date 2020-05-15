import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch

from neuralteleportation.layers.activationlayers import ReLUCOB
from neuralteleportation.layers.neuralteleportationlayers import FlattenCOB
from neuralteleportation.layers.neuronlayers import LinearCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

torch.manual_seed(0)


net = nn.Sequential(FlattenCOB(),
                    LinearCOB(784, 128),
                    ReLUCOB(),
                    LinearCOB(128, 128),
                    ReLUCOB(),
                    LinearCOB(128, 10)
                    )

sample_input_shape = (1, 1, 28, 28)
model = NeuralTeleportationModel(network=net, input_shape=sample_input_shape, )

# Get the initial set of weights and teleport.
initial_weights = model.get_weights()
model.random_teleport()

# Get second set of weights (Goal weights)
target_weights = model.get_weights()

# Get the change of basis that created this set of weights.
target_cob = model.get_cob(concat=True)

print("Target cob shape: ", target_cob.shape)

# Generate a new random cob
cob = model.generate_random_cob()
print("cob shape: ", cob.shape)
print("cob grad_fn: ", cob.grad_fn)

history = []
cob_error_history = []

print("Inital error: ", (cob - target_cob).abs().mean())
print("Target cob: ", target_cob[0:10])
print("cob: ", cob[0:10])

"""
Optimize the cob to find the 'target_cob' that produced the original teleportation. 
"""

optimizer = optim.Adam([cob])

for e in range(10000):
    # Reset the initial weights.
    model.set_weights(initial_weights)

    # Teleport with this cob
    model.set_change_of_basis(cob)
    model.teleport()

    # Get the new weights and calculate the loss
    weights = model.get_weights()
    loss = (weights - target_weights).abs().mean()

    # Backwards pass
    # cob.retain_grad()
    loss.backward(retain_graph=True)

    # Calculate cob gradient and perform gradient descent step
    # grad = cob.grad
    # cob -= cob.grad * 0.01
    # cob.grad.zero_()

    optimizer.step()
    optimizer.zero_grad()

    history.append(loss.item())
    cob_error_history.append((cob - target_cob).abs().mean().item())
    if e % 100 == 0:
        print("Step: {}, loss: {}, cob error: {}".format(e, loss.item(), (cob - target_cob).abs().mean().item()))

print("Final error: ", (cob - target_cob).abs().mean())
print("Target cob: ", target_cob[0:10])
print("cob: ", cob[0:10])


plt.figure()
plt.plot(history)
plt.title("Loss")
plt.show()

plt.figure()
plt.plot(cob_error_history)
plt.title("Cob error")
plt.show()
