from pathlib import Path
from typing import Tuple, List, Callable, Dict, Union, Any, Sequence

import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset, MNIST, CIFAR10, CIFAR100

from neuralteleportation.models.model_zoo import mlpcob, resnetcob, vggcob, densenetcob
from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.models.model_zoo.vggcob import vgg16COB, vgg16_bnCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TeleportationTrainingConfig
from neuralteleportation.utils.optimtools import initialize_model

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
                      "cifar100": {"cls": CIFAR100, "input_channels": 3, "image_size": (32, 32), "num_classes": 100,
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
                                   ])}}
__models__ = [MLPCOB, vgg16COB, resnet18COB, densenet121COB, vgg16_bnCOB]


def get_dataset_info(dataset_name: str, *tags: str) -> Dict[str, Any]:
    return {tag: __dataset_config__[dataset_name.lower()][tag] for tag in tags}


def get_dataset_subsets(dataset_name: str, root: Path = "/tmp", download: bool = True, transform=None) \
        -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    if transform is None:
        transform = transforms.ToTensor()
    dataset_conf = __dataset_config__[dataset_name.lower()]
    dataset_cls = dataset_conf["cls"]
    train_transform = dataset_conf["train_transform"] if "train_transform" in dataset_conf.keys() else transform
    test_transform = dataset_conf["test_transform"] if "test_transform" in dataset_conf.keys() else transform
    train_set = dataset_cls(str(root), train=True, download=download, transform=train_transform)
    val_set = dataset_cls(str(root), train=False, download=download, transform=test_transform)
    test_set = dataset_cls(str(root), train=False, download=download, transform=test_transform)
    return train_set, val_set, test_set


def _get_model_factories() -> Dict[str, Union[Callable[..., nn.Module], nn.Module]]:
    model_modules = [mlpcob, resnetcob, densenetcob, vggcob]
    return {model_name: getattr(model_module, model_name)
            for model_module in model_modules
            for model_name in model_module.__all__}


def get_model_names() -> List[str]:
    return list(_get_model_factories().keys())


def get_model(dataset_name: str, model_name: str, device: str = 'cpu',
              initializer: Dict[str, Union[str, float]] = None, **model_kwargs) -> NeuralTeleportationModel:
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

    if "cifar" in dataset_name and ("resnet" in model_name):
        model_kwargs.update({"for_dataset": "cifar"})
    # Instantiate the model
    model_factory = model_factories[model_name]
    model = model_factory(**model_kwargs)
    # Initialize the model
    if initializer is not None:
        init_gain = None if "gain" not in initializer.keys() and initializer["type"] == "none" else initializer["gain"]
        init_non_linearity = None if "non_linearity" not in initializer.keys() else initializer["non_linearity"]
        model = initialize_model(model, init_type=initializer["type"], init_gain=init_gain, non_linearity=init_non_linearity)

    # Transform the base ``nn.Module`` to a ``NeuralTeleportationModel``
    input_channels, image_size = get_dataset_info(dataset_name, "input_channels", "image_size").values()
    model = NeuralTeleportationModel(network=model, input_shape=(2, input_channels, *image_size))

    return model.to(device)


def get_models_for_dataset(dataset_name: str) -> List[NeuralTeleportationModel]:
    return [get_model(dataset_name, model.__name__) for model in __models__]


def get_optimizer_from_model_and_config(model: nn.Module, config: TrainingConfig, lr: float = None) -> Optimizer:
    optimizer_name, optimizer_kwargs = config.optimizer
    if lr:
        optimizer_kwargs.update({"lr": lr})
    return getattr(optim, optimizer_name)(model.parameters(), **optimizer_kwargs)


def get_lr_scheduler_from_optimizer_and_config(optimizer: Optimizer, config: TrainingConfig):
    lr_scheduler_name, _, lr_scheduler_kwargs = config.lr_scheduler
    return getattr(optim.lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_kwargs)


def get_teleportation_epochs(config: TeleportationTrainingConfig) -> Sequence[int]:
    """Determines the epochs at the start of which to teleport the model, based on the configuration.

    If the ``teleport_only_once`` flag is active, the ``every_n_epochs`` parameter indicates the ONLY epoch at which to
    teleport. Otherwise, ``every_n_epochs`` is the period of the teleportations, and is used to determine at which
    epochs (between 0 and ``epochs``) to teleport.

    Args:
        config: Collection of hyperparameters regarding the training, and the teleportations to apply.

    Returns:
        Indices of the epochs at the start of which to teleport the model during training.
    """
    if config.teleport_only_once:
        epochs = [config.every_n_epochs]
    else:
        epochs = range(config.every_n_epochs, config.epochs + 1, config.every_n_epochs)
    return epochs
