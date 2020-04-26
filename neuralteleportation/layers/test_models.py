import torch.nn as nn

from neuralteleportation.layers.activationlayers import ReLUCOB
from neuralteleportation.layers.layers_v3 import MaxPool2dCOB, AvgPool2dCOB, BatchNorm2dCOB, FlattenCOB
from neuralteleportation.layers.mergelayers import Add, Concat
from neuralteleportation.layers.neuronlayers import Conv2dCOB, LinearCOB, ConvTranspose2dCOB


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
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = Conv2dCOB(1, 6, 5)
        self.pool1 = MaxPool2dCOB(kernel_size=2)
        self.pool2 = MaxPool2dCOB(kernel_size=2)
        self.conv2 = Conv2dCOB(6, 16, 5)
        # self.fc1 = LinearCOB(16 * 4 * 4, 120)
        # self.fc2 = LinearCOB(120, 84)
        # self.fc3 = LinearCOB(84, 10)
        self.flatten = FlattenCOB()
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        # self.relu3 = ReLUCOB()
        # self.relu4 = ReLUCOB()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # x = self.flatten(x)
        # x = self.relu3(self.fc1(x))
        # x = self.relu4(self.fc2(x))
        # x = self.fc3(x)
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
    TODO FIX THE ORDERING OF OPERATIONS BEFORE RESIDUAL CONNECTION
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



class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=6, out_channels=3, kernel_size=3, padding=1)
        self.conv4 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        self.relu4 = ReLUCOB()
        self.concat1 = Concat()
        self.flatten = FlattenCOB()
        self.fc1 = LinearCOB(2352, 10)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))

        # x2 += x1
        x2 = self.concat1([x1, x2])

        x3 = self.relu3(self.conv3(x2))
        x4 = self.conv4(x3)
        x = self.flatten(x4)
        x = self.relu4(self.fc1(x))

        return x


class DenseNet2(nn.Module):
    def __init__(self):
        super(DenseNet2, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv4 = Conv2dCOB(in_channels=9, out_channels=1, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu2 = ReLUCOB()
        self.relu3 = ReLUCOB()
        # self.relu4 = ReLUCOB()
        self.concat1 = Concat()
        # self.flatten = FlattenCOB()
        # self.fc1 = LinearCOB(2352, 10)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))

        x3 = self.relu3(self.conv3(x2))
        x3 = self.concat1([x1, x2, x3])
        x4 = self.conv4(x3)
        # x = self.flatten(x4)
        # x = self.relu4(self.fc1(x))

        return x4


class CombinedModule(nn.Module):
    def __init__(self):
        super(CombinedModule, self).__init__()
        self.resnet = ResidualNet()
        self.densenet = DenseNet2()

    def forward(self, x):
        x = self.densenet(x)
        x = self.resnet(x)

        return x


class SplitConcatModel(nn.Module):
    def __init__(self):
        super(SplitConcatModel, self).__init__()
        self.conv1 = Conv2dCOB(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv21 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv22 = Conv2dCOB(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = Conv2dCOB(in_channels=6, out_channels=3, kernel_size=3, padding=1)
        self.relu1 = ReLUCOB()
        self.relu21 = ReLUCOB()
        self.relu22 = ReLUCOB()
        # self.relu3 = ReLUCOB()
        # self.relu4 = ReLUCOB()
        self.concat1 = Concat()
        # self.flatten = FlattenCOB()
        # self.fc1 = LinearCOB(2352, 10)

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x21 = self.relu21(self.conv21(x1))
        x22 = self.relu22(self.conv22(x1))

        x2 = self.concat1([x21, x22])
        x3 = self.conv3(x2)

        return x3


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
