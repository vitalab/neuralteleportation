from typing import Tuple, List, Callable, Dict, Union, Any

import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset, MNIST, CIFAR10, CIFAR100

from neuralteleportation.models.model_zoo import mlpcob, resnetcob, vggcob, densenetcob
from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg16COB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel

__dataset_config__ = {"mnist": {"cls": MNIST, "input_channels": 1, "image_size": (28, 28), "num_classes": 10},
                      "cifar10": {"cls": CIFAR10, "input_channels": 3, "image_size": (32, 32), "num_classes": 10,
                                  "train_transform": transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010)),
                                  ]),
                                  "test_transform": transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010)),
                                  ])},
                      "cifar100": {"cls": CIFAR100, "input_channels": 3, "image_size": (32, 32), "num_classes": 100}}
__models__ = [MLPCOB, vgg16COB, resnet18COB, densenet121COB]

from neuralteleportation.training.config import TrainingConfig


def get_dataset_info(dataset_name: str, *tags: str) -> Dict[str, Any]:
    return {tag: __dataset_config__[dataset_name.lower()][tag] for tag in tags}


def get_dataset_subsets(dataset_name: str, transform=None) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    if transform is None:
        transform = transforms.ToTensor()
    dataset_conf = __dataset_config__[dataset_name.lower()]
    dataset_cls = dataset_conf["cls"]
    train_transform = dataset_conf["train_transform"] if "train_transform" in dataset_conf.keys() else transform
    test_transform = dataset_conf["test_transform"] if "test_transform" in dataset_conf.keys() else transform
    train_set = dataset_cls('/tmp', train=True, download=True, transform=train_transform)
    val_set = dataset_cls('/tmp', train=False, download=True, transform=test_transform)
    test_set = dataset_cls('/tmp', train=False, download=True, transform=test_transform)
    return train_set, val_set, test_set


def _get_model_factories() -> Dict[str, Union[Callable[..., nn.Module], nn.Module]]:
    model_modules = [mlpcob, resnetcob, densenetcob, vggcob]
    return {model_name: getattr(model_module, model_name)
            for model_module in model_modules
            for model_name in model_module.__all__}


def get_model_names() -> List[str]:
    return list(_get_model_factories().keys())


def get_model(dataset_name: str, model_name: str, device: str = 'cpu', **model_kwargs) -> NeuralTeleportationModel:
    # Look up if the requested model is available in the model zoo
    model_factories = _get_model_factories()
    if model_name not in model_factories:
        raise KeyError(f"{model_name} was not found in the model zoo")

    # Dynamically determine the parameters for initializing the model based on the dataset
    model_kwargs.update(get_dataset_info(dataset_name, "num_classes"))
    if "mlp" in model_name.lower():
        input_channels, image_size = get_dataset_info(dataset_name, "input_channels", "image_size").values()
        model_kwargs.update(input_shape=(input_channels, *image_size))
    else:
        model_kwargs.update(get_dataset_info(dataset_name, "input_channels"))

    # Instantiate the model
    model_factory = model_factories[model_name]
    model = model_factory(**model_kwargs)

    # Transform the base ``nn.Module`` to a ``NeuralTeleportationModel``
    input_channels, image_size = get_dataset_info(dataset_name, "input_channels", "image_size").values()
    model = NeuralTeleportationModel(network=model, input_shape=(2, input_channels, *image_size))

    return model.to(device)


def get_models_for_dataset(dataset_name: str) -> List[NeuralTeleportationModel]:
    return [get_model(dataset_name, model.__name__) for model in __models__]


def get_optimizer_from_model_and_config(model: nn.Module, config: TrainingConfig, new_lr: float = None) -> Optimizer:
    optimizer_name, optimizer_kwargs = config.optimizer
    if new_lr:
        optimizer_kwargs.update({"lr": new_lr})
    return getattr(optim, optimizer_name)(model.parameters(), **optimizer_kwargs)


def get_lr_scheduler_from_optimizer_and_config(optimizer: Optimizer, config: TrainingConfig):
    lr_scheduler_name, _, lr_scheduler_kwargs = config.lr_scheduler
    return getattr(optim.lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_kwargs)
