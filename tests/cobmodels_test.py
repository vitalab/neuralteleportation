"""
Test for models constructed with layers Conv2dCOB, LinearCOB, FlattenCOB, ReLUCOB, MaxPool2dCOB, BatchNorm2dCOB,
ConvTranspose2dCOB, BatchNorm1dCOB
"""

import torch
import torch.nn as nn

from neuralteleportation.layers.activation import ReLUCOB, TanhCOB, SigmoidCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, ConvTranspose2dCOB, BatchNorm2dCOB, \
    BatchNorm1dCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.conv2 = Conv2dCOB(6, 16, 5)
        self.fc1 = LinearCOB(16 * 20 * 20, 120)
        self.fc2 = LinearCOB(120, 84)
        self.fc3 = LinearCOB(84, 10)
        self.flatten = FlattenCOB()
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.tanhcob = TanhCOB()
        self.relu3 = ReLUCOB()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.tanhcob(self.fc1(x))
        x = self.relu3(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.pool1 = MaxPool2dCOB(kernel_size=2)
        self.pool2 = MaxPool2dCOB(kernel_size=2)
        self.conv2 = Conv2dCOB(6, 16, 5)
        self.flatten = FlattenCOB()
        self.sigmoid = SigmoidCOB()
        self.relu1 = ReLUCOB()

    def forward(self, x):
        x = self.pool1(self.sigmoid(self.conv1(x)))
        x = self.pool2(self.relu1(self.conv2(x)))
        return x


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.conv2 = Conv2dCOB(6, 3, 5)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.bn1 = BatchNorm2dCOB(6)
        self.bn2 = BatchNorm2dCOB(3)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.conv2 = Conv2dCOB(6, 3, 5)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.bn1 = BatchNorm2dCOB(6)
        self.bn2 = BatchNorm2dCOB(3)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class ConvTransposeNet(nn.Module):
    def __init__(self):
        super(ConvTransposeNet, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.conv2 = ConvTranspose2dCOB(6, 3 // 2, kernel_size=2, stride=2)
        self.relu1 = ReLUCOB(inplace=True)
        self.relu2 = ReLUCOB(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            FlattenCOB(),
            LinearCOB(784, 128),
            BatchNorm1dCOB(128),
            ReLUCOB(),
            LinearCOB(128, 10)
        )

    def forward(self, input):
        return self.net(input)


if __name__ == '__main__':
    from tests.model_test import test_teleport

    models = [MLP, Net, Net2, Net3, Net4, ConvTransposeNet]
    input_shape = (1, 1, 28, 28)

    for model in models:
        try:
            mod = model()
            mod.eval()
            diff_avg = test_teleport(mod, input_shape)
            print("{} model passed with avg diff: {}".format(mod, diff_avg))
        except Exception as e:
            print("Teleportation failed for model: {} with error {}".format(mod, e))

    print("All tests are done")
