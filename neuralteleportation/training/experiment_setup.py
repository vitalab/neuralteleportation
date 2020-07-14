import functools
from typing import Tuple, List, Callable, Dict, Union, Any

import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import VisionDataset, MNIST, CIFAR10, CIFAR100

from neuralteleportation.models.model_zoo import resnetcob, vggcob, densenetcob
from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
from torch.nn.modules import Flatten
from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules


def _default_vision_transform(func: Callable[[Callable], Any]):
    default_transform = transforms.ToTensor()

    @functools.wraps(func)
    def wrapper_vision_dataset(transform: Callable = None):
        return func(transform if transform else default_transform)

    return wrapper_vision_dataset


@_default_vision_transform
def get_mnist_datasets(transform=None) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    train_set = MNIST('/tmp', train=True, download=True, transform=transform)
    val_set = MNIST('/tmp', train=False, download=True, transform=transform)
    test_set = MNIST('/tmp', train=False, download=True, transform=transform)
    return train_set, val_set, test_set


@_default_vision_transform
def get_cifar10_datasets(transform=None) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    train_set = CIFAR10('/tmp', train=True, download=True, transform=transform)
    val_set = CIFAR10('/tmp', train=False, download=True, transform=transform)
    test_set = CIFAR10('/tmp', train=False, download=True, transform=transform)
    return train_set, val_set, test_set


@_default_vision_transform
def get_cifar100_datasets(transform=None) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    train_set = CIFAR100('/tmp', train=True, download=True, transform=transform)
    val_set = CIFAR100('/tmp', train=False, download=True, transform=transform)
    test_set = CIFAR100('/tmp', train=False, download=True, transform=transform)
    return train_set, val_set, test_set


def _to_device(func: Callable[[], List[nn.Module]]):
    @functools.wraps(func)
    def wrapper_to_device(device: str = 'cpu'):
        return [module.to(device) for module in func()]

    return wrapper_to_device


@_to_device
def get_mnist_models() -> List[nn.Module]:
    return [
        MLPCOB(num_classes=10),
        vgg16COB(num_classes=10, input_channels=1),
        resnet18COB(num_classes=10, input_channels=1),
        densenet121COB(num_classes=10, input_channels=1),
    ]


@_to_device
def get_cifar10_models() -> List[nn.Module]:
    return [
        vgg16COB(num_classes=10, input_channels=3),
        resnet18COB(num_classes=10, input_channels=3),
        densenet121COB(num_classes=10, input_channels=3),
    ]


@_to_device
def get_cifar100_models() -> List[nn.Module]:
    return [
        vgg16COB(num_classes=100, input_channels=3),
        resnet18COB(num_classes=100, input_channels=3),
        densenet121COB(num_classes=100, input_channels=3),
    ]


def get_cifar10_MLP() -> nn.Module:
    mlp_model = nn.Sequential(
        Flatten(),
        nn.Linear(3072, 1536),
        nn.ReLU(),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)
    return mlp_model


def get_cifar100_MLP() -> nn.Module:
    mlp_model = nn.Sequential(
        Flatten(),
        nn.Linear(3072, 1536),
        nn.ReLU(),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 100)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)
    return mlp_model


def _get_model_factories() -> Dict[str, Union[Callable[..., nn.Module], nn.Module]]:
    model_modules = [resnetcob, densenetcob, vggcob]
    return {model_name: getattr(model_module, model_name)
            for model_module in model_modules
            for model_name in model_module.__all__}


def get_model_list() -> List[str]:
    return list(chain.from_iterable([__resnets__, __vggnets__, __densenets__]))

def get_model_names() -> List[str]:
    return list(_get_model_factories().keys())


def get_model_from_name(model_name: str, num_classes: int, input_channels: int) -> nn.Module:
    model_factories = _get_model_factories()
    if model_name not in model_factories:
        raise KeyError(f"{model_name} was not found in the model zoo")

    model_factory = model_factories[model_name]
    model = model_factory(pretrained=False, num_classes=num_classes, input_channels=input_channels)
    return model
