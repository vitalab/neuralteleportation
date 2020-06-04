import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import LinearCOB
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.training import train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = TrainingConfig(epochs=0, device=device)
metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

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
                                 LinearCOB(128, 10),
                                 )

    def forward(self, input):
        return self.net(input)

sample_input_shape = (1, 1, 28, 28)

model1 = Model().to(device)
model1 = NeuralTeleportationModel(network=model1, input_shape=sample_input_shape)
# model1.load_state_dict(torch.load('initial_model.pt'))
# model1.load_state_dict(torch.load('model1.pt'))
train(model1, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_test)
# torch.save(model1.state_dict(), 'model1.pt')
print(test(model1, mnist_test, metrics, config))

model2 = Model().to(device)
model2 = NeuralTeleportationModel(network=model2, input_shape=sample_input_shape)
# model2.load_state_dict(torch.load('initial_model.pt'))
# model2.load_state_dict(torch.load('model2.pt'))
train(model2, train_dataset=mnist_train, metrics=metrics, config=config, val_dataset=mnist_test)
# torch.save(model2.state_dict(), 'model2.pt')
print(test(model2, mnist_test, metrics, config))

# x = torch.rand(sample_input_shape)
x, y = mnist_train[0]

pred1 = model1(x.to(device))
pred2 = model2(x.to(device))

print("Model 1 prediction: ", pred1)
print("Model 2 prediction: ", pred2)
print("Pred diff", (pred1 - pred2).abs())

print("Initial: w1 - w2: ", (model1.get_weights() - model2.get_weights()).abs())

print("Before")
w1 = model1.get_weights()
w2 = model2.get_weights()
diff = (w1.detach().cpu() - w2.detach().cpu()).abs().mean()
print("Initial weight diff :", diff)

w1 = model1.get_weights(concat=False, flatten=False, bias=False)
w2 = model2.get_weights(concat=False, flatten=False, bias=False)
calculated_cob = model1.calculate_cob(w1, w2, concat=True)

model1.set_change_of_basis(calculated_cob)
model1.teleport()

print("After")
w1 = model1.get_weights()
w2 = model2.get_weights()
diff = (w1.detach().cpu() - w2.detach().cpu()).abs().mean()
print("Predicted weight diff :", diff)

w1 = model1.get_weights(concat=False, flatten=False, bias=False)
w2 = model2.get_weights(concat=False, flatten=False, bias=False)

for i in range(len(w1)):
    print('layer : ', i)
    print("w1  - w2 = ", (w1[i].detach().cpu() - w2[i].detach().cpu()).abs().sum())
    print("w1: ", w1[i].detach().cpu().flatten()[:10])
    print("w2: ", w2[i].detach().cpu().flatten()[:10])
