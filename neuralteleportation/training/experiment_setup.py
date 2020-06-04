import functools
from typing import Tuple, List, Callable
from itertools import chain

import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import VisionDataset, MNIST, CIFAR10

from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg16COB

from neuralteleportation.models.model_zoo import resnetcob, vggcob, densenetcob
from neuralteleportation.models.model_zoo.resnetcob import __all__ as __resnets__
from neuralteleportation.models.model_zoo.vggcob import __all__ as __vggnets__
from neuralteleportation.models.model_zoo.densenetcob import __all__ as __densenets__


def get_mnist_datasets() -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    train_set = MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    val_set = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    test_set = MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    return train_set, val_set, test_set


def get_cifar10_datasets() -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    train_set = CIFAR10('/tmp', train=True, download=True, transform=transforms.ToTensor())
    val_set = CIFAR10('/tmp', train=False, download=True, transform=transforms.ToTensor())
    test_set = CIFAR10('/tmp', train=False, download=True, transform=transforms.ToTensor())
    return train_set, val_set, test_set


def to_device(func: Callable[[], List[nn.Module]]):
    @functools.wraps(func)
    def wrapper_to_device(device: str = 'cpu'):
        return [module.to(device) for module in func()]

    return wrapper_to_device


@to_device
def get_mnist_models() -> List[nn.Module]:
    return [
        MLPCOB(),
        vgg16COB(num_classes=10, input_channels=1),
        resnet18COB(num_classes=10, input_channels=1),
        densenet121COB(num_classes=10, input_channels=1),
    ]


@to_device
def get_cifar10_models() -> List[nn.Module]:
    return [
        vgg16COB(num_classes=10, input_channels=3),
        resnet18COB(num_classes=10, input_channels=3),
        densenet121COB(num_classes=10, input_channels=3),
    ]


def get_model_list() -> List[str]:
    return list(chain.from_iterable([__resnets__, __vggnets__, __densenets__]))


def get_model_from_string(model, num_classes: int, input_channels: int) -> nn.Module:
    if model in __resnets__:
        func = getattr(resnetcob, model)
    elif model in __densenets__:
        func = getattr(densenetcob, model)
    elif model in __vggnets__:
        func = getattr(vggcob, model)
    else:
        raise Exception("%s was not found in the model zoo" % model)
    model = func(pretrained=False, num_classes=num_classes, input_channels=input_channels)
    return model
