"""
Code from torchvision.models.vgg modified with cob layers.
https://pytorch.org/docs/stable/torchvision/models.html
"""

import torch.nn as nn
from torch.hub import load_state_dict_from_url

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.dropout import DropoutCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, BatchNorm2dCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB, AdaptiveAvgPool2dCOB

__all__ = [
    'VGGCOB', 'vgg11COB', 'vgg11_bnCOB', 'vgg13COB', 'vgg13_bnCOB', 'vgg16COB', 'vgg16_bnCOB',
    'vgg19_bnCOB', 'vgg19COB',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGGCOB(nn.Module):

    def __init__(self, features, num_classes, init_weights=True):
        super(VGGCOB, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2dCOB((7, 7))
        self.flatten = FlattenCOB()
        self.classifier = nn.Sequential(
            LinearCOB(512 * 7 * 7, 4096),
            ReLUCOB(True),
            DropoutCOB(),
            LinearCOB(4096, 4096),
            ReLUCOB(True),
            DropoutCOB(),
            LinearCOB(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, input_channels=3):
    layers = []
    in_channels = input_channels
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2dCOB(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2dCOB(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2dCOB(v), ReLUCOB(inplace=True)]
            else:
                layers += [conv2d, ReLUCOB(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vggCOB(arch, cfg, batch_norm, pretrained, progress, input_channels=3, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGCOB(make_layers(cfgs[cfg], batch_norm=batch_norm, input_channels=input_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11COB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg11', 'A', False, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg11_bnCOB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg11_bn', 'A', True, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg13COB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg13', 'B', False, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg13_bnCOB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg13_bn', 'B', True, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg16COB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg16', 'D', False, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg16_bnCOB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg16_bn', 'D', True, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg19COB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
                input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg19', 'E', False, pretrained, progress, input_channels=input_channels, **kwargs)


def vgg19_bnCOB(pretrained=False, progress=True, input_channels=3, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        input_channels (int): number of input channels for the network.
    """
    return _vggCOB('vgg19_bn', 'E', True, pretrained, progress, input_channels=input_channels, **kwargs)


if __name__ == '__main__':
    from torchsummary import summary
    from tests.model_test import test_teleport

    vgg = vgg16COB(pretrained=False, input_channels=1, num_classes=10)
    vgg.eval()  # Put the model in eval to compute the outputs (will change if in train() because of dropout)
    summary(vgg, (1, 224, 224), device='cpu')
    test_teleport(vgg, (1, 1, 224, 224), verbose=True)
