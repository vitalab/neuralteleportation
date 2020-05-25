import torch.nn as nn

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.merge import Add
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB


class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3)
        self.conv4 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()
        self.add = Add()
        self.flatten = FlattenCOB()
        self.fc1 = LinearCOB(3 * 24 * 24, 10)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))

        # x2 += x1
        x2 = self.add(x1, x2)

        x3 = self.relu3(self.conv3(x2))
        x4 = self.conv4(x3)
        x = self.flatten(x4)
        x = self.relu4(self.fc1(x))
        return x


class ResidualNet2(nn.Module):
    def __init__(self):
        super(ResidualNet2, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv4 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()
        self.add1 = Add()
        self.add2 = Add()
        self.flatten = FlattenCOB()
        self.fc1 = LinearCOB(2028, 10)

        self.relu5 = ReLUCOB()

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))

        # x2 += x1
        x2 = self.add1(x1, x2)
        x2 = self.relu5(x2)

        x3 = self.relu3(self.conv3(x2))
        x3 = self.add2(x1, x3)

        x4 = self.conv4(x3)
        x = self.flatten(x4)
        x = self.relu4(self.fc1(x))
        return x


class ResidualNet3(nn.Module):
    """
    This resnet has no conv after the add. We therefore need to set the change of basis of the previous residual layer
    to ones because it is the ouput cob
    """

    def __init__(self):
        super(ResidualNet3, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.add1 = Add()
        self.relu3 = ReLUCOB()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu2(self.conv2(x1))

        # x2 += x1
        x2 = self.add1(x1, x2)
        x2 = self.relu3(x2)

        return x2


class ResidualNet4(nn.Module):
    """
    This resnet has no conv before resisdual connection.
    """

    def __init__(self):
        super(ResidualNet4, self).__init__()
        self.conv0 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv1 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.add1 = Add()
        self.relu3 = ReLUCOB()

    def forward(self, x):
        x = self.conv0(x)
        identity = x

        x1 = self.conv1(x)
        x2 = self.relu2(self.conv2(x1))

        # x2 += x1
        x2 = self.add1(identity, x2)
        x3 = self.conv3(x2)
        x3 = self.relu3(x3)

        return x2


class ResidualNet5(nn.Module):
    """
    This resnet has no operation in the residual connection.
    """

    def __init__(self):
        super(ResidualNet5, self).__init__()
        nb_channels = 3
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=nb_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1)
        self.conv4 = Conv2dCOB(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3)
        self.conv11 = Conv2dCOB(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()
        self.add1 = Add()
        self.add2 = Add()

        self.relu5 = ReLUCOB()

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x11 = self.relu4(self.conv11(x1))

        x2 = self.relu2(self.conv2(x1))
        x3 = self.relu3(self.conv3(x2))

        # x11 = self.relu4(self.conv11(x1))

        x3 = self.add2(x11, x3)

        x4 = self.conv4(x3)
        return x4


if __name__ == '__main__':
    from tests.generic_models_test import test_generic_models

    test_generic_models([ResidualNet, ResidualNet2, ResidualNet3, ResidualNet4, ResidualNet5])
