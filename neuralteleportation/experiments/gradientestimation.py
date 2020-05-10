from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from neuralteleportation.metrics import accuracy
from neuralteleportation.models.model_zoo.vgg import vgg16, vgg11
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from neuralteleportation.models.test_models.test_models import Net
from neuralteleportation.training import train
import torch.nn
from neuralteleportation.layers.neuralteleportationlayers import FlattenCOB
from neuralteleportation.layers.neuronlayers import LinearCOB, Conv2dCOB
from neuralteleportation.layers.activationlayers import ReLUCOB
import torch.nn as nn

# transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
transform = transforms.ToTensor()

mnist_train = MNIST('/tmp', train=True, download=True, transform=transform)
mnist_val = MNIST('/tmp', train=False, download=True, transform=transform)
mnist_test = MNIST('/tmp', train=False, download=True, transform=transform)

loss = nn.CrossEntropyLoss()
TRAIN = False

# net = nn.Sequential(FlattenCOB(),
#                     LinearCOB(784, 128),
#                     ReLUCOB(),
#                     LinearCOB(128, 128),
#                     ReLUCOB(),
#                     LinearCOB(128, 10),
#                     )

net = Net()

device = 'cpu'

# net = vgg11()
# net.features[0] = Conv2dCOB(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# net.cuda()

sample_input_shape = (1, 1, 28, 28)
# sample_input_shape = (1, 1, 224, 224)
model = NeuralTeleportationModel(network=net, input_shape=sample_input_shape, device=device)

if TRAIN:
    optim = torch.optim.Adam(model.parameters())
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()
    train(model, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_val, optimizer=optim, metrics=metrics,
          epochs=2, device=device)

net.eval()
w1 = model.get_weights()
model.random_teleport(cob_range=0.000001)
w2 = model.get_weights()
distance = torch.dist(w1, w2)
print("Distance between weights: {}".format(distance))
tangent = (w1-w2) / torch.norm(w1 - w2)
random_tangent = torch.rand(tangent.shape)
random_tangent = random_tangent / torch.norm(random_tangent)


batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
dot_results = []
random_dot_results = []
num_itters = 100
for b in batch_sizes:
    d = []
    d_random = []
    for n in tqdm(range(num_itters)):
        data_loader = DataLoader(mnist_test, batch_size=b, shuffle=True)
        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        grad = model.get_grad(data, target, loss)
        grad = grad / torch.norm(grad)
        dot = tangent.dot(grad).abs()
        dot_rand = random_tangent.dot(grad).abs()
        d.append(dot.cpu().detach().numpy())
        d_random.append(dot_rand.cpu().detach().numpy())

    d = np.array(d)
    d_random = np.array(d_random)
    dot_results.append(d)
    random_dot_results.append(d_random)
    print("Batch size: {} - Dot product between tangent and gradient: {}, random: {}".format(b,
                                                                                             d.mean(),
                                                                                             d_random.mean()))


dot_results = np.array(dot_results)
random_dot_results = np.array(random_dot_results)
print(dot_results.shape)
print(dot_results.std(axis=-1))

plt.figure()
plt.title("Dot product between tangent and gradient ({})".format('trained' if TRAIN else 'untrained'))
plt.errorbar(batch_sizes, dot_results.mean(axis=-1), yerr=dot_results.std(axis=-1)*2, fmt='o', ecolor='g', capsize=2, label='tangent', c='b')
# plt.errorbar(batch_sizes, random_dot_results.mean(axis=-1), yerr=random_dot_results.std(axis=-1)*2, fmt='o', ecolor='g', capsize=2, label='random', c='r')
# plt.scatter(batch_sizes, random_dot_results.mean(axis=-1), label='random', c='r')
# plt.xscale("log")
plt.xlabel('Batch size')
plt.ylabel("Dot product result")
plt.xticks(batch_sizes, batch_sizes)
plt.legend()
plt.show()

